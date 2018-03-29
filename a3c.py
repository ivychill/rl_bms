# -*- coding: utf-8 -*-
from bms_env import *
import os
import threading
# import multiprocessing
# import skimage
# from skimage import transform, color, exposure
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from time import sleep
# import logging.handlers

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# preprocess the screen image
def preprocess(processed_input):
    return processed_input

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


# ### Actor-Critic Network
class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.state_input = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            layer1 = slim.fully_connected(slim.flatten(self.state_input), 256, activation_fn=tf.nn.elu)
            layer2 = slim.fully_connected(slim.flatten(layer1), 256, activation_fn=tf.nn.elu)
            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(layer2, [0])
            step_size = tf.shape(self.state_input)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


class Worker():
    def __init__(self, name, s_size, a_size, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(SUMMARY_PATH + str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.actions = np.identity(a_size, dtype=bool).tolist()

        self.env = BmsEnv()
        # self.env = gym.wrappers.Monitor(self.env, DUMP_PATH)

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.state_input: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def sample(self, distribution):
        act = np.random.choice(a_size, p=distribution)
        return act

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        local_episode_count = 0
        total_steps = 0
        reward_sum = 0
        logger.warn("Starting worker %d" % (self.number))

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                done = False

                rnn_state = self.local_AC.state_init

                self.env.firstRun()
                observation = self.env.reset()

                while observation == None:
                    sleep(10)
                    logger.warn("firstRun failure.....")
                    self.env.firstRun()
                    observation = self.env.reset()

                state = preprocess(observation)

                while (not done):
                    # self.env.render()
                    # Take an action using probabilities from policy network output.
                    try:
                        a_dist, value, rnn_state = sess.run(
                            [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                            feed_dict={self.local_AC.state_input: [state],
                                       self.local_AC.state_in[0]: rnn_state[0],
                                       self.local_AC.state_in[1]: rnn_state[1]})
                        action = self.sample(a_dist[0])
                        sleep(0.5)
                        observation, reward, done, info = self.env.step(action)
                        state1 = preprocess(observation)
                        episode_buffer.append([state, action, reward, state1, done, value[0, 0]])
                        episode_values.append(value[0, 0])
                        episode_reward += reward
                        logger.debug('%s episode: %d, step: %d, state: %s, action: %d, reward: %d, done: %r'
                                     % (self.name, local_episode_count, episode_step_count, state, action, reward, done))
                        reward_sum += reward
                        total_steps += 1
                        episode_step_count += 1
                        state = state1
                    except Exception,e:
                        logger.debug('Exception: %s' %e)

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if local_episode_count % EPISODE_BATCH_SIZE == (EPISODE_BATCH_SIZE - 1):
                    mean_reward = np.mean(self.episode_rewards[-EPISODE_BATCH_SIZE:])
                    mean_length = np.mean(self.episode_lengths[-EPISODE_BATCH_SIZE:])
                    mean_value = np.mean(self.episode_mean_values[-EPISODE_BATCH_SIZE:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, local_episode_count)
                    self.summary_writer.flush()
                    logger.warn('%s, episode %d, average reward %f' % (self.name, local_episode_count, reward_sum / EPISODE_BATCH_SIZE))
                    if reward_sum / EPISODE_BATCH_SIZE >= 0.8:
                        logger.warn('%s task solved in %d episodes!' % (self.name, local_episode_count))
                        # self.env.close()
                        # gym.upload(DUMP_PATH, api_key = GYM_API_KEY)
                        # os._exit(1)
                    reward_sum = 0

                if local_episode_count % SAVE_INTERVAL == (SAVE_INTERVAL - 1) and self.name == 'worker_0':
                    logger.warn('save model at epoch %d' % (local_episode_count))
                    ckpt_file = os.path.join(self.model_path, 'a3c_tennis')
                    saver.save(sess, ckpt_file, (local_episode_count + 1) / SAVE_INTERVAL)

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                local_episode_count += 1


max_episode_length = 300
gamma = .99  # discount rate for advantage estimation and reward discounting
# # 8个状态依次为：自己飞机的z, speed, pitch, yaw；TD框左上角位置(x,y), TD框右侧的2个读数。
# s_size = 8
# 4个状态依次为：自己飞机的z, speed, pitch, yaw
s_size = 4
# 8个动作依次为：无,仰角上/中/下,扫描角度,扫描线数,TD框左/右
a_size = 8

load_model = True
MODEL_PATH = './model'
SUMMARY_PATH = './summary/train_'
EPISODE_BATCH_SIZE = 100
SAVE_INTERVAL = 100

tf.reset_default_graph()

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# with tf.device("/cpu:0"):
global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
trainer = tf.train.AdamOptimizer(learning_rate=2e-5)
master_network = AC_Network(s_size, a_size, 'global', None)  # Generate global network
# num_workers = multiprocessing.cpu_count()  # Set workers ot number of available CPU threads
num_workers = 1
workers = []
# Create worker classes
for i in range(num_workers):
    workers.append(Worker(i, s_size, a_size, trainer, MODEL_PATH, global_episodes))
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        logger.warn ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)