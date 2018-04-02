
from ddpg import *
from fly_env import *

is_training = True         #True means Train, False means simply Run
# np.random.seed(1337)


def playGame():
    train_indicator = is_training
    env = FlyEnv()
    agent = DDPG(env)

    EPISODE_BATCH_SIZE = 100
    epsilon = 1.0
    explore_count = 300000.0
    episode_count = 1000000
    step = 0
    best_reward = 10
    reward_sum = 0

    # it take long to execute agent.noise_actionfirst time, so execute a dummy run.
    s_t0 = np.zeros((5 * 4))
    a_t = agent.noise_action(s_t0, epsilon)

    logger.warn("Experiment Start...")
    for episode in range(episode_count):
        episode_reward = 0.
        step_eps = 0.
        done = False
        # x_t = env.reset()
        s_t = env.reset()
        # logger.debug("s_t: %s" % s_t)

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

            if (train_indicator):
                agent.perceive(s_t,a_t,r_t,s_t1,done) # save the transition in the buffer
                # logger.debug("s_t: %s, a_t: %s, r_t: %s, s_t1: %s, done: %s" % (s_t, a_t, r_t, s_t1, done))
                logger.info("episode: %d, step_eps: %d, reward: %s" % (episode, step_eps, r_t))

            s_t = s_t1
            step += 1
            step_eps += 1
            episode_reward += r_t

        logger.info("episode: %d, step_eps: %d, step: %d, total reward: %s, replay buffer: %d" % (episode, step_eps, step, episode_reward, agent.replay_buffer.count()))

        reward_sum += episode_reward
        if episode % EPISODE_BATCH_SIZE == (EPISODE_BATCH_SIZE - 1):
            reward_average = reward_sum / EPISODE_BATCH_SIZE
            logger.warn('episode %d, average reward %f' % (episode, reward_average))
            if reward_average >= best_reward:
                if (train_indicator):
                    logger.info(
                        "Now we save model with reward %s, previous best reward %s" % (reward_average, best_reward))
                    best_reward = episode_reward
                    agent.saveNetwork()

            reward_sum = 0

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