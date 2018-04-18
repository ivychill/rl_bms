from fly_env import *

class FlyEnvHorizontalCircle(FlyEnv):
    def set_fixed_start_param(self):
        self.altitude_start = 12000
        self.speed_start = 400
        self.roll_start = 75 * math.pi / 180

    def set_continous_start_param(self):
        self.altitude_start = self.RANGE_ALTITUDE[0] + (self.RANGE_ALTITUDE[1] - self.RANGE_ALTITUDE[0]) * random.uniform(0, 1)
        self.speed_start = self.RANGE_SPEED[0] + (self.RANGE_SPEED[1] - self.RANGE_SPEED[0]) * random.uniform(0, 1)
        self.roll_start = self.RANGE_ROLL[0] + (self.RANGE_ROLL[1] - self.RANGE_ROLL[0]) * random.uniform(0, 1)
        logger.debug("altitude_start: %s, speed_start: %s, roll_start: %s"
                 % (self.altitude_start, self.speed_start, self.roll_start * 180 / math.pi))

    def set_narrow_discrete_start_param(self):
        params = ((6000, 420, 80 * math.pi / 180),
                  (9000, 410, 79 * math.pi / 180),
                  (12000, 400, 78 * math.pi / 180),
                  (15000, 390, 77 * math.pi / 180),
                  (18000, 380, 76 * math.pi / 180),
                  )

        self.altitude_start, self.speed_start, self.roll_start = params[np.random.randint(5)]
        logger.debug("altitude_start: %s, speed_start: %s, roll_start: %s"
                 % (self.altitude_start, self.speed_start, self.roll_start * 180 / math.pi))

    def set_wide_discrete_start_param(self):
        params = ((6000, 400, 78 * math.pi / 180),
                  (6000, 400, 80 * math.pi / 180),
                  (6000, 400, 82 * math.pi / 180),
                  (6000, 420, 78 * math.pi / 180),
                  (6000, 420, 80 * math.pi / 180),
                  (6000, 420, 82 * math.pi / 180),
                  (6000, 440, 78 * math.pi / 180),
                  (6000, 440, 80 * math.pi / 180),
                  (6000, 440, 82 * math.pi / 180),
                  (9000, 390, 77 * math.pi / 180),
                  (9000, 390, 79 * math.pi / 180),
                  (9000, 390, 81 * math.pi / 180),
                  (9000, 410, 77 * math.pi / 180),
                  (9000, 410, 79 * math.pi / 180),
                  (9000, 410, 81 * math.pi / 180),
                  (9000, 430, 77 * math.pi / 180),
                  (9000, 430, 79 * math.pi / 180),
                  (9000, 430, 81 * math.pi / 180),
                  (12000, 380, 76 * math.pi / 180),
                  (12000, 380, 78 * math.pi / 180),
                  (12000, 380, 80 * math.pi / 180),
                  (12000, 400, 76 * math.pi / 180),
                  (12000, 400, 78 * math.pi / 180),
                  (12000, 400, 80 * math.pi / 180),
                  (12000, 420, 76 * math.pi / 180),
                  (12000, 420, 78 * math.pi / 180),
                  (12000, 420, 80 * math.pi / 180),
                  (15000, 370, 75 * math.pi / 180),
                  (15000, 370, 77 * math.pi / 180),
                  (15000, 370, 79 * math.pi / 180),
                  (15000, 390, 75 * math.pi / 180),
                  (15000, 390, 77 * math.pi / 180),
                  (15000, 390, 79 * math.pi / 180),
                  (15000, 410, 75 * math.pi / 180),
                  (15000, 410, 77 * math.pi / 180),
                  (15000, 410, 79 * math.pi / 180),
                  (18000, 360, 74 * math.pi / 180),
                  (18000, 360, 76 * math.pi / 180),
                  (18000, 360, 78 * math.pi / 180),
                  (18000, 380, 74 * math.pi / 180),
                  (18000, 380, 76 * math.pi / 180),
                  (18000, 380, 78 * math.pi / 180),
                  (18000, 400, 74 * math.pi / 180),
                  (18000, 400, 76 * math.pi / 180),
                  (18000, 400, 78 * math.pi / 180),
                  )

        self.altitude_start, self.speed_start, self.roll_start = params[np.random.randint(5*3*3)]
        logger.debug("altitude_start: %s, speed_start: %s, roll_start: %s"
                 % (self.altitude_start, self.speed_start, self.roll_start * 180 / math.pi))


    def get_reward(self):
        reward = 0
        # proposed_speed_vector = (self.altitude_start - self.altitude)/100 * math.pi/180
        # reward = (self.TOLERANCE_SPEED_VECTOR - abs(self.speed_vector - proposed_speed_vector)) * 180/math.pi
        # logger.debug("proposed_speed_vector: %s, reward: %s" % (proposed_speed_vector * 180/math.pi, reward))

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

        # if self.more_than_half_circle == True:
        #     if (self.deviation_altitude < self.TOLERANCE_ALTITUDE
        #             and self.deviation_speed < self.TOLERANCE_SPEED
        #             and self.deviation_roll < self.TOLERANCE_ROLL
        #             and abs(self.speed_vector) < self.TOLERANCE_SPEED_VECTOR):
        #         reward += 1000

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