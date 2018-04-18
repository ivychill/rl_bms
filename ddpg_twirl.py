# -*- coding: utf-8 -*-

from ddpg import *


# Hyper Parameters:
REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
# REPLAY_START_SIZE = 2000
BATCH_SIZE = 64
GAMMA = 0.99
MODEL_PATH = './model'


class DDPG_TWIRL(DDPG):
    def __init__(self, env):
        self.name = 'DDPG' # name for uploading results
        self.environment = env
        # Randomly initialize actor network and critic network
        # with both their target networks
        # self.state_dim = env.observation_space.shape[0]
        self.state_dim = 16
        # self.action_dim = env.action_space.shape[0]
        self.action_dim = 3

        self.time_step = 0
        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(self.sess,self.state_dim,self.action_dim)
        self.critic_network = CriticNetwork(self.sess,self.state_dim,self.action_dim)
        
        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        # self.exploration_noise = OUNoise(self.action_dim)
        # self.exploration_noise = OUNoise()
        self.OU = OU()
        # loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(MODEL_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            logger.warn("Successfully loaded: %s" % (checkpoint.model_checkpoint_path))
        else:
            logger.error("Could not find old network weights")

        self.critic_cost = 0

    def train(self):
        # logger.debug("......enter tain......")
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch,[BATCH_SIZE,self.action_dim])

        # Calculate y_batch
        
        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch,next_action_batch)
        y_batch = []  
        for i in range(len(minibatch)): 
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch,[BATCH_SIZE,1])
        # Update critic by minimizing the loss L
        self.critic_cost = self.critic_network.train(y_batch,state_batch,action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch,action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch,state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    # def noise_action(self,state):
    #     # Select action a_t according to the current policy and exploration noise
    #     action = self.actor_network.action(state)
    #     noise = self.exploration_noise.noise(action)
    #     noise_action = action + noise
    #     clipped_noise_action = np.clip(noise_action, 0, 1)
    #     return clipped_noise_action

    def noise_action(self,state,epsilon):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.action(state)
        noise = np.zeros(self.action_dim)

        # added 2018-03-29
        cur_altitude = self.environment.altitude
        cur_speed = self.environment.speed
        cur_roll = self.environment.roll
        cur_pitch = self.environment.pitch
        cur_speed_vector = self.environment.speed_vector

        # x_mu = 0.66 + (self.environment.roll_start - cur_roll) * 0.4
        # x_mu = x_mu + cur_speed_vector * 1.0
        # proposed_speed_vector = (self.environment.altitude_start - cur_altitude)/100 * math.pi/180
        # logger.debug("proposed_speed_vector: %s" % (proposed_speed_vector))
        # y_mu = 0.8 + (proposed_speed_vector - cur_speed_vector) * 1.0
        x_mu = 0.66
        y_mu = 0.5
        z_mu = 0.5
        # z_mu = 0.25 - (self.environment.speed_start - cur_speed) * 0.01
        # z_mu = z_mu + (self.environment.roll_start - cur_roll) * 0.5
        # z_mu = z_mu - cur_speed_vector * 2
        logger.debug(" x_mu: %s , y_mu: %s, z_mu: %s " % (x_mu,y_mu,z_mu))
        # noise_action = [x_mu, y_mu, z_mu]
        noise[0] = epsilon * self.OU.function(action[0], x_mu, 1.00, 0.10)
        noise[1] = epsilon * self.OU.function(action[1], y_mu, 1.00, 0.10)
        # noise = epsilon * self.OU.function(action, x_mu, 1.00, 0.10)

        noise[2] = epsilon * self.OU.function(action[2], z_mu, 1.00, 0.10)
        noise_action = action + noise
        logger.debug("action: %s, noise: %s" % (action, noise))
        clipped_noise_action = np.clip(noise_action, 0, 1)
        return clipped_noise_action

    def action(self,state):
        # logger.debug("predict begin...")
        action = self.actor_network.action(state)
        # logger.debug("predict end...")
        return action

    def perceive(self,state,action,reward,next_state,done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state,action,reward,next_state,done)

        self.time_step = self.time_step + 1

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() >  REPLAY_START_SIZE:
            logger.debug("train...")
            self.train()

        #if self.time_step % 10000 == 0:
            #self.actor_network.save_network(self.time_step)
            #self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        # if done:
        #     self.exploration_noise.reset()

    def saveNetwork(self):
        # logger.warn("time step: %s, save model" % (self.time_step))
        ckpt_file = os.path.join(MODEL_PATH, 'DDPG')
        self.saver.save(self.sess, ckpt_file, global_step = self.time_step)