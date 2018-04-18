from fly_env import *

class FlyEnvHorizontalCircle(FlyEnv):
    def set_fixed_start_param(self):
        self.altitude_start = 6000
        self.speed_start = 500
        self.roll_start = 30 * math.pi / 180


    def get_reward(self):
        reward = 0
        reward = reward\
                 - self.deviation_altitude / self.TOLERANCE_ALTITUDE\
                 - self.deviation_speed / self.TOLERANCE_SPEED\
                 - self.deviation_roll / self.TOLERANCE_ROLL\
                 - abs(self.speed_vector) / self.TOLERANCE_SPEED_VECTOR

        # punish if gs > 8 and gs < -1.5
        # reward = reward - max((self.gs - 8), 0) * 10 - abs(min((self.gs + 1.5), 0)) * 10
        if self.gs > 8:
            reward = reward - (self.gs - 8) * 10
        elif self.gs < -1.5:
            reward = reward - (-1.5 - self.gs) * 10


        return reward


    def get_done(self):
        if self.step_eps >= 300:
            logger.warn("reach 300 steps, done!")
            return True

        elif self.altitude < 5000:
            logger.warn("latitude less than the least altitude, done!")
            return True

        elif self.more_than_half_circle == True:
            if (self.deviation_altitude < self.TOLERANCE_ALTITUDE
                    and self.deviation_speed < self.TOLERANCE_SPEED
                    and self.deviation_roll < self.TOLERANCE_ROLL
                    and abs(self.speed_vector) < self.TOLERANCE_SPEED_VECTOR):
                logger.warn("spiral for half circle, done!")
                return True

        return False