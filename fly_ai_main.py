
from ddpg_horizontal_circle import *
from fly_env_horizontal_circle import *

is_training = True  # True means Train, False means simply Run
np.random.seed(1337)


def playGame():
    train_indicator = is_training
    env = FlyEnvHorizontalCircle()
    agent = DDPG_HORIZONTAL_CIRCLE(env)

    EPISODE_COUNT = 1000000
    TEST_START_SIZE = 100
    EPISODE_BATCH_SIZE = 20
    TEST_SIZE = 4
    step = 0
    best_reward = -1000
    reward_sum = 0
    SUMMARY_PATH = './summary'

    summary_writer = tf.summary.FileWriter(SUMMARY_PATH)
    summary = tf.Summary()

    # it take long to execute agent.noise_action first time, so execute a dummy run.
    s_t0 = np.zeros((5 * 4))
    a_t = agent.action(s_t0)

    logger.warn("Experiment Start...")
    # for episode in range(EPISODE_COUNT):
    episode = 0
    while episode <= EPISODE_COUNT:
        episode_reward = 0.
        step_eps = 0.
        done = False
        # x_t = env.reset()
        s_t = env.reset()
        if s_t is None:
            logger.warn("reset fail...")
            continue

        # logger.debug("s_t: %s" % s_t)

        while not done:
            # Take noisy actions during agent training
            if (train_indicator):  # to make the first step
                a_t = agent.noise_action(s_t)
            else:
                a_t = agent.action(s_t)
                # a_t = agent.expert_action(s_t)

            x_t1, r_t, done, _ = env.step(a_t)
            time.sleep(0.1)
            s_t1 = np.append(s_t[5:] ,x_t1) # stack continious 4 frames
            logger.debug("episode: %d, step_eps: %d, step_reward: %s" % (episode, step_eps, r_t))

            if (train_indicator):
                agent.perceive(s_t ,a_t ,r_t ,s_t1 ,done) # save the transition in the buffer

            s_t = s_t1
            step += 1
            step_eps += 1
            episode_reward += r_t

        logger.info("episode: %d, step_eps: %d, step: %d, episode_reward: %s, replay buffer: %d" %
        (episode, step_eps, step, episode_reward, agent.replay_buffer.count()))

        reward_sum += episode_reward
        if episode % EPISODE_BATCH_SIZE == (EPISODE_BATCH_SIZE - 1):
            average_reward = reward_sum / EPISODE_BATCH_SIZE
            logger.info('episode %d, average reward %f' % (episode, average_reward))
            summary.value.add(tag='reward', simple_value=float(average_reward))
            summary.value.add(tag='critic_loss', simple_value=float(agent.critic_cost))
            summary_writer.add_summary(summary, episode)
            summary_writer.flush()

            if train_indicator and episode >= TEST_START_SIZE:
                # test during train
                reward_test_sum = 0
                for episode_test in xrange(TEST_SIZE):
                    episode_reward_test = 0
                    done_test = False
                    state_test = env.reset()
                    if state_test is None:
                        logger.warn("test reset fail...")
                        continue
                    while not done_test:
                        # env.render()
                        action_test = agent.action(state_test)  # direct action for test
                        x_t1_test, reward_test, done_test, _ = env.step(action_test)
                        time.sleep(0.1)
                        state_test = np.append(state_test[5:], x_t1_test)  # stack continious 4 frames
                        episode_reward_test += reward_test
                    logger.debug("test episode %d, reward %s" % (episode_test, episode_reward_test))
                    reward_test_sum += episode_reward_test

                average_reward_test = reward_test_sum / TEST_SIZE
                logger.info("episode %d, test average reward %s" % (episode, average_reward_test))

                if average_reward_test >= best_reward:
                    logger.warn("save model with train reward %s, test reward %s, previous best reward %s" % (average_reward, average_reward_test, best_reward))
                    best_reward = average_reward_test
                    agent.saveNetwork()

            reward_sum = 0

        episode += 1

    logger.warn("Finish...")


if __name__ == "__main__":
    playGame()
