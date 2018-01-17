
from ddpg import *
from bms_env import *

is_training = True         #True means Train, False means simply Run
max_eps = 1000000
test_eps = 10
# np.random.seed(1337)

def playGame():
    discount = 1
    train_indicator = is_training
    env = BmsEnv()
    agent = DDPG(env)

    episode_count = max_eps
    step = 0
    best_reward = 0
    best_shaped_reward = 0

    def action(a_t_input): # action is the float between (0,1), if less than 0.5, not break
        if a_t_input < 0.5:
            action = 0
        else:
            action = 1
        return action

    logger.warn("Experiment Start...")
    for episode in range(episode_count):
        env.reset()
        env.get_flag_entrance()
        total_reward = 0.
        step_eps = 0.
        done = False
        s_t = np.zeros((64))
        x_t, _1, _2 = env.get_feedback()
        s_t = np.append(s_t[16:],x_t) # set the first three frames to be zero and stack the last frame to it
        logger.debug("s_t.shape: %s" % s_t.shape)
        logger.debug("s_t: %s" % s_t)

        # Take noisy actions during agent training
        if (train_indicator): # to make the first step
            a_t = agent.noise_action(s_t)
            action = action(a_t)
        else:
            a_t = agent.action(s_t)
            action = action(a_t)
        logger.debug("a_t: %s" % a_t)
        logger.debug("action: %s" % action)

        while not action: # while FlyCtrl not break, collect normal buffer
            env.step(action)
            time.sleep(0.1)
            x_t1, r_t, done = env.get_feedback()
            logger.debug("get_feedback: %s" % env.get_feedback())
            s_t1 = np.append(s_t[16:],x_t1) # stack continious 4 frames
            logger.debug("s_t1: %s" % s_t1)

            if (train_indicator):
                agent.perceive(s_t,a_t,r_t,s_t1,done) # save the transition in the buffer
                logger.debug("s_t: %s, a_t: %s, r_t: %s, s_t1: %s, done: %s" % (s_t, a_t, r_t, s_t1, done))

            if (train_indicator): # Should a_t be 1 anytime, it will break
                a_t = agent.noise_action(s_t)
                action = action(a_t)
            else:
                a_t = agent.action(s_t)
                action = action(a_t)
            logger.debug("a_t: %s" % a_t)
            logger.debug("action: %s" % action)

            s_t = s_t1
            step += 1
            step_eps += 1
            total_reward += r_t

        env.step(action) # this time action is 1
        time.sleep(0.1)
        s_t1, r_t, done = env.get_feedback() # r_t here is immediate reward
        logger.debug("get_feedback: %s" % env.get_feedback())
        # step_eps_last = step_eps

        r_t_last = 0
        while not done:
            time.sleep(0.1)
            _3, r_t_last, done = env.get_feedback()
            discount *= 0.99 # keep multiply the discout factor for each step

            # total_reward += r_t
            # if (np.mod(step, 100) == 0):
            #     logger.debug("episode: %s, step: %s, action: %s, reward: %s" % (episode, step_eps, a_t, r_t))
            # step += 1
            # step_eps += 1

        r_t += r_t_last * discount # the reward for the last frame to be stored in buffer
        logger.debug("r_t: %s" % r_t)
        agent.perceive(s_t, a_t, r_t, s_t1, done) # store the final frame info into buffer
        logger.debug("s_t: %s, a_t: %s, r_t: %s, s_t1: %s, done: %s" % (s_t, a_t, r_t, s_t1, done))

        logger.info("episode: %s, step_eps: %s, step: %s, reward: %s, replay buffer: %s" % (episode, step_eps, step, total_reward, agent.replay_buffer.count()))

        # Saving the best model.
        if total_reward >= best_reward:
            if (train_indicator):
                logger.info("Now we save model with reward %s, previous best reward %s shaped %s" % (total_reward, best_reward, best_shaped_reward))
                best_reward = total_reward
                agent.saveNetwork()

            # total_reward_test = 0
            # for i in xrange(test_eps):
            #     state_test = env.reset()
            #     done_test = False
            #     while not done_test:
            #         # env.render()
            #         action_test = agent.action(state_test)  # direct action for test
            #         state_test, reward_test, done_test, _ = env.step(action_test)
            #         total_reward_test += reward_test
            #         # logger.debug("test action: %s, reward: %s, total reward: %s" % (action_test, reward_test, total_reward_test))
            # ave_reward = total_reward_test / test_eps
            # logger.info("Episode: %s, Evaluation Average Reward: %s" % (episode, ave_reward))

    logger.warn("Finish...")

if __name__ == "__main__":
    playGame()