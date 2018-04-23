from ddpg import *

class DDPG_HORIZONTAL_CIRCLE(DDPG):
    def noise_action(self,state):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.action(state)
        noise = np.zeros(self.action_dim)

        # added 2018-03-29
        cur_altitude = self.environment.altitude
        cur_speed = self.environment.speed
        cur_roll = self.environment.roll
        cur_pitch = self.environment.pitch
        cur_speed_vector = self.environment.speed_vector

        x_mu = 0.5 + (self.environment.roll_start - cur_roll) * 0.4
        x_mu = x_mu + cur_speed_vector * 1.0
        proposed_speed_vector = (self.environment.altitude_start - cur_altitude)/100 * math.pi/180
        # logger.debug("proposed_speed_vector: %s" % (proposed_speed_vector))
        y_mu = 0.7 + (proposed_speed_vector - cur_speed_vector) * 1.0
        z_mu = 0.25 - (self.environment.speed_start - cur_speed) * 0.01
        z_mu = z_mu + (self.environment.roll_start - cur_roll) * 0.5
        # z_mu = z_mu - cur_speed_vector * 2
        logger.debug("x_mu: %s, y_mu: %s, z_mu: %s" % (x_mu, y_mu, z_mu))
        # noise[0] = self.epsilon * self.OU.function(action[0], x_mu, 1.00, 0.10)
        # noise[1] = self.epsilon * self.OU.function(action[1], y_mu, 1.00, 0.10)
        # noise[2] = self.epsilon * self.OU.function(action[2], z_mu, 1.00, 0.10)
        noise[0] = self.epsilon_expert * (x_mu - action[0]) + self.epsilon_random * np.random.randn(1)
        noise[1] = self.epsilon_expert * (y_mu - action[0]) + self.epsilon_random * np.random.randn(1)
        noise[2] = self.epsilon_expert * (z_mu - action[0]) + self.epsilon_random * np.random.randn(1)
        noise_action = action + noise
        logger.debug("action: %s, noise: %s" % (action, noise))
        clipped_noise_action = np.clip(noise_action, 0, 1)
        return clipped_noise_action

    def expert_action(self,state):
        cur_altitude = self.environment.altitude
        cur_speed = self.environment.speed
        cur_roll = self.environment.roll
        cur_pitch = self.environment.pitch
        cur_speed_vector = self.environment.speed_vector

        x_mu = 0.5 + (self.environment.roll_start - cur_roll) * 0.4
        x_mu = x_mu + cur_speed_vector * 1.0
        proposed_speed_vector = (self.environment.altitude_start - cur_altitude)/100 * math.pi/180
        # logger.debug("proposed_speed_vector: %s" % (proposed_speed_vector))
        y_mu = 0.7 + (proposed_speed_vector - cur_speed_vector) * 1.0
        z_mu = 0.25 - (self.environment.speed_start - cur_speed) * 0.01
        z_mu = z_mu + (self.environment.roll_start - cur_roll) * 0.5

        logger.debug("x_mu: %s, y_mu: %s, z_mu: %s" % (x_mu, y_mu, z_mu))
        expert_action = [x_mu, y_mu, z_mu]
        clipped_expert_action = np.clip(expert_action, 0, 1)
        return clipped_expert_action