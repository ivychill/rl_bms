from ddpg import *

class DDPG_HORIZONTAL_CIRCLE(DDPG):
    def noise_action(self,state):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.action(state)
        expert = self.expert_action_unclipped()
        noise_action = (1 - self.epsilon_expert) * action + self.epsilon_expert * expert + self.epsilon_random * np.random.randn(3)
        logger.debug("action: %s, expert: %s" % (action, expert))
        clipped_noise_action = np.clip(noise_action, 0, 1)
        return clipped_noise_action

    def ai_and_expert_action(self,state):
        action = self.actor_network.action(state)
        expert = self.expert_action_unclipped()
        noise_action = (1 - self.epsilon_expert) * action + self.epsilon_expert * expert
        clipped_noise_action = np.clip(noise_action, 0, 1)
        return clipped_noise_action

    def ai_and_random_action(self,state):
        action = self.actor_network.action(state)
        noise_action = action + self.epsilon_random * np.random.randn(3)
        clipped_noise_action = np.clip(noise_action, 0, 1)
        return clipped_noise_action

    def expert_and_random_action(self):
        noise_action = self.expert_action_unclipped() + self.epsilon_random * np.random.randn(3)
        clipped_noise_action = np.clip(noise_action, 0, 1)
        return clipped_noise_action

    def expert_action(self):
        expert = self.expert_action_unclipped()
        clipped_expert_action = np.clip(expert, 0, 1)
        return clipped_expert_action

    def expert_action_unclipped(self):
        expert = np.zeros(self.action_dim)
        cur_altitude = self.environment.altitude
        cur_speed = self.environment.speed
        cur_roll = self.environment.roll
        cur_speed_vector = self.environment.speed_vector

        expert[0] = 0.5 + (self.environment.roll_start - cur_roll) * 0.4
        expert[0] = expert[0] + cur_speed_vector * 1.0
        proposed_speed_vector = (self.environment.altitude_start - cur_altitude)/100 * math.pi/180
        expert[1] = 0.7 + (proposed_speed_vector - cur_speed_vector) * 1.0
        expert[2] = 0.25 - (self.environment.speed_start - cur_speed) * 0.01
        expert[2] = expert[2] + (self.environment.roll_start - cur_roll) * 0.5
        logger.debug("unclipped expert: %s" % (expert))
        return expert