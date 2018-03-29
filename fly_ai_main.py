
from ddpg import *
from fly_env import *

is_training = True         #True means Train, False means simply Run
# np.random.seed(1337)


def playGame():
    train_indicator = is_training
    env = FlyEnv()
    agent = DDPG(env)

    epsilon = 1.0
    explore_count = 300000.0
    episode_count = 1000000
    step = 0
    best_reward = 0
    best_shaped_reward = 0

    # it take long to execute agent.noise_actionfirst time, so execute a dummy run.
    s_t0 = np.zeros((5 * 4))
    a_t = agent.noise_action(s_t0, epsilon)

    logger.warn("Experiment Start...")
    for episode in range(episode_count):
        total_reward = 0.
        step_eps = 0.
        done = False
        # x_t = env.reset()
        s_t = env.reset()
        logger.debug("s_t: %s" % s_t)

        while not done: # while FlyCtrl not break, collect normal buffer
            # Take noisy actions during agent training
            if (train_indicator):  # to make the first step
                epsilon -= 1.0 / explore_count
                epsilon = max(epsilon, 0.1)
                # a_t = agent.noise_action(s_t)
                a_t = agent.noise_action(s_t, epsilon)
            else:
                a_t = agent.action(s_t)
            # logger.debug("a_t: %s" % a_t)

            x_t1, r_t, done, _ = env.step(a_t)
            time.sleep(0.1)

            s_t1 = np.append(s_t[5:],x_t1) # stack continious 4 frames
            logger.debug("s_t1: %s" % s_t1)

            if (train_indicator):
                agent.perceive(s_t,a_t,r_t,s_t1,done) # save the transition in the buffer
                # logger.debug("s_t: %s, a_t: %s, r_t: %s, s_t1: %s, done: %s" % (s_t, a_t, r_t, s_t1, done))

            s_t = s_t1
            step += 1
            step_eps += 1
            total_reward += r_t

        # logger.debug("r_t: %s" % r_t)
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