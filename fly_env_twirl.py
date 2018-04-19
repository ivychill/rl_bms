# -*- coding: utf-8 -*-

from fly_env import *


# This is spiral task
# 启停的顺序极其重要，不能变
class FlyEnvTwirl(FlyEnv):
    # state: altitude[3000,21000], speed[160,640], roll[-pi, pi], pitch[-pi, pi]
    # action: x(roll), y(pitch), z(speed)
    def __init__(self):
        self.bms_control_proxy = xmlrpclib.ServerProxy("http://192.168.20.72:8000/")
        self.fly_proxy = xmlrpclib.ServerProxy("http://192.168.20.118:4022/")
        # self.image_proxy = xmlrpclib.ServerProxy("http://192.168.20.129:5001/")
        self.bms_socket = socket(AF_INET, SOCK_DGRAM)
        self.bms_action_addr = ("192.168.20.72", 4001)
        self.episode = 0
        self.step_eps = 0
        self.RANGE_ALTITUDE = (4000, 18000)
        # self.RANGE_ALTITUDE = (10000, 14000)
        # self.RANGE_SPEED = (160, 640)
        # self.RANGE_SPEED = (360, 600)
        self.RANGE_SPEED = (430, 485)
        self.RANGE_YAW = (-5*math.pi/180, 5*math.pi/180)
        # self.RANGE_ROLL = (-math.pi, math.pi)
        # self.RANGE_PITCH = (-math.pi, math.pi)
        # self.RANGE_SPEED_VECTOR = (-math.pi, math.pi)
        # self.RANGE_ROLL = (72*math.pi/180, 84*math.pi/180)
        self.RANGE_PITCH = (-10*math.pi/180, 10*math.pi/180)
        self.RANGE_SPEED_VECTOR = (0*math.pi/180, 90*math.pi/180)
        self.RANGE_ACTION = (1, 32766)

        self.TOLERANCE_ALTITUDE = 500
        self.TOLERANCE_SPEED = 20
        self.TOLERANCE_ROLL = 2 * math.pi / 180
        self.TOLERANCE_YAW = 5 * math.pi / 180
        self.TOLERANCE_PITCH = 5 * math.pi / 180
        self.TOLERANCE_SPEED_VECTOR = 10 * math.pi / 180

        self.Turn_up_done = False
        self.altitude, self.speed, self.roll, self.pitch, self.speed_vector, self.yaw, self.gs = self.fly_proxy.get_fly_state()


    # "1": start
    # "2": pause
    # "3": restart
    # TODO:
    def reset(self):
        logger.warn("reboot...")
        self.send_ctrl_cmd('3')
        logger.info('reset joystick...')
        self.fly_proxy.prepare()
        logger.info('start bms...')
        self.send_ctrl_cmd('1')
        logger.info('start fly control...')
        prestate = self.start_fly()
        logger.info('reset over...')
        self.step_eps = 0
        self.Turn_up_done = False
        # state = self.get_state()
        # used for DDPG exploration input
        self.altitude, self.speed, self.roll, self.pitch, self.speed_vector, self.yaw, self.gs = self.fly_proxy.get_fly_state()
        logger.debug("altitude: %s, speed: %s, roll: %s, pitch: %s, speed_vector: %s, yaw: %s, gs: %s"
                     % (self.altitude, self.speed, self.roll*180/math.pi, self.pitch*180/math.pi, self.speed_vector*180/math.pi, self.yaw*180/math.pi, self.gs))
        return prestate


    def start_fly(self):
        self.set_discrete_param()
        fly_state = self.fly_proxy.fly_till_twirl(self.altitude_start, self.speed_start)
        if fly_state is None:
            logger.warn("fly_till_twirl fail")
            return None

        self.yaw_start = self.fly_proxy.get_yaw()
        logger.debug("yaw_start: %s" % (self.yaw_start*180/math.pi))
        # logger.debug("altitude_start: %s, speed_start: %s, roll_start: %s, yaw_start: %s"
        #              % (self.altitude_start, self.speed_start, self.roll_start*180/math.pi, self.yaw_start*180/math.pi))

        for index_i in range(4):
            fly_state[index_i * 4] = (fly_state[index_i * 4] - self.RANGE_ALTITUDE[0])/(self.RANGE_ALTITUDE[1]- self.RANGE_ALTITUDE[0])
            fly_state[index_i * 4 + 1] = (fly_state[index_i * 4 + 1] - self.RANGE_SPEED[0])/(self.RANGE_SPEED[1]- self.RANGE_SPEED[0])
            fly_state[index_i * 4 + 2] = (fly_state[index_i * 4 + 2] - self.RANGE_PITCH[0]) / (self.RANGE_PITCH[1] - self.RANGE_PITCH[0])
            fly_state[index_i * 4 + 3] = (fly_state[index_i * 4 + 3] - self.RANGE_YAW[0]) / (self.RANGE_YAW[1] - self.RANGE_YAW[0])
        logger.debug("fly_state: %s" % (fly_state))
        logger.debug("normalized fly_state: %s" % (fly_state))

        pre_state = np.array(fly_state)
        return pre_state


    def set_fixed_start_param(self):
        self.altitude_start = 4000
        self.speed_start = 430
        # self.roll_start = 75 * math.pi / 180

    def set_continous_start_param(self):
        self.altitude_start = self.RANGE_ALTITUDE[0] + (self.RANGE_ALTITUDE[1] - self.RANGE_ALTITUDE[0]) * random.uniform(0, 1)
        self.speed_start = self.RANGE_SPEED[0] + (self.RANGE_SPEED[1] - self.RANGE_SPEED[0]) * random.uniform(0, 1)
        self.roll_start = self.RANGE_ROLL[0] + (self.RANGE_ROLL[1] - self.RANGE_ROLL[0]) * random.uniform(0, 1)
        logger.debug("altitude_start: %s, speed_start: %s, roll_start: %s"
                 % (self.altitude_start, self.speed_start, self.roll_start * 180 / math.pi))

    def set_discrete_start_param(self):
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


    def set_discrete_param(self):
        params = ((5000, 430),
                  (5000, 450),
                  (5000, 470),
                  (5000, 485),
                  (7000, 430),
                  (7000, 450),
                  (7000, 470),
                  (7000, 485),
                  (9000, 430),
                  (9000, 450),
                  (9000, 470),
                  (9000, 485),
                  (11000, 430),
                  (11000, 450),
                  (11000, 470),
                  (11000, 485),
                  (13000, 430),
                  (13000, 450),
                  (13000, 470),
                  (13000, 485),
                  (15000, 430),
                  (15000, 450),
                  (15000, 470),
                  (15000, 485),
                  (17000, 430),
                  (17000, 450),
                  (17000, 470),
                  (17000, 485),
                  )

        self.altitude_start, self.speed_start = params[np.random.randint(7*4)]
        logger.debug("altitude_start: %s, speed_start: %s,"
                 % (self.altitude_start, self.speed_start))


    # done: (altitude, speed, roll) = inception (altitude, speed, roll)
    # Reward：-1/step,
    def step(self, action):
        self.take_action(action)
        self.step_eps += 1
        state = self.get_state()
        if self.Turn_up_done == False:
            if (abs(self.roll) < self.TOLERANCE_ROLL):
                self.Turn_up_done = True
                logger.debug('....roll_done....')
        reward = self.get_reward()
        done = self.get_done()
        return state, reward, done, {}


    def take_action(self, action):
        action_x = round(self.RANGE_ACTION[0] + (self.RANGE_ACTION[1] - self.RANGE_ACTION[0]) * action[0])
        action_y = round(self.RANGE_ACTION[0] + (self.RANGE_ACTION[1] - self.RANGE_ACTION[0]) * action[1])
        action_z = round(self.RANGE_ACTION[0] + (self.RANGE_ACTION[1] - self.RANGE_ACTION[0]) * action[2])
        logger.debug("action_x: %s ,action_y: %s, action_z: %s " % (action_x, action_y, action_z))
        self.bms_socket.sendto("x:" + str(action_x), self.bms_action_addr)
        self.bms_socket.sendto("y:" + str(action_y), self.bms_action_addr)
        self.bms_socket.sendto("z:" + str(action_z), self.bms_action_addr)


    def get_state(self):
        self.altitude, self.speed, self.roll, self.pitch, self.speed_vector, self.yaw, self.gs = self.fly_proxy.get_fly_state()
        # self.altitude = self.fly_proxy.get_altitude()
        normalized_altitude = (self.altitude - self.RANGE_ALTITUDE[0])/(self.RANGE_ALTITUDE[1]- self.RANGE_ALTITUDE[0])
        # self.speed = self.fly_proxy.get_speed()
        normalized_speed = (self.speed - self.RANGE_SPEED[0])/(self.RANGE_SPEED[1]- self.RANGE_SPEED[0])
        # self.roll = self.fly_proxy.get_roll()
        normalized_yaw = (self.yaw - self.RANGE_YAW[0])/(self.RANGE_YAW[1]- self.RANGE_YAW[0])
        normalized_pitch = (self.pitch - self.RANGE_PITCH[0]) / (self.RANGE_PITCH[1] - self.RANGE_PITCH[0])
        # normalized_speed_vector = (self.speed_vector - self.RANGE_SPEED_VECTOR[0]) / (self.RANGE_SPEED_VECTOR[1] - self.RANGE_SPEED_VECTOR[0])
        # self.yaw = self.fly_proxy.get_yaw()
        # self.gs = self.fly_proxy.get_gs()

        logger.debug("altitude: %s, speed: %s, roll: %s, pitch: %s, speed_vector: %s, yaw: %s, gs: %s"
                     % (self.altitude, self.speed, self.roll*180/math.pi, self.pitch*180/math.pi, self.speed_vector*180/math.pi, self.yaw*180/math.pi, self.gs))

        # self.deviation_altitude = self.altitude/self.altitude_start - 1
        # self.deviation_speed = self.speed/self.speed_start - 1
        # self.deviation_roll = self.roll/self.roll_start - 1
        self.deviation_altitude = abs(self.altitude- self.altitude_start)
        self.deviation_speed = abs(self.speed - self.speed_start)
        self.deviation_yaw = abs(self.yaw- self.yaw_start)

        # return np.asarray(
        #     [normalized_altitude, normalized_speed, normalized_roll, normalized_pitch, normalized_speed_vector])

        return np.asarray([normalized_altitude, normalized_speed, normalized_pitch, normalized_yaw])


    def get_reward(self):
        reward = 0
        # proposed_speed_vector = (self.altitude_start - self.altitude)/100 * math.pi/180
        # reward = (self.TOLERANCE_SPEED_VECTOR - abs(self.speed_vector - proposed_speed_vector)) * 180/math.pi
        # logger.debug("proposed_speed_vector: %s, reward: %s" % (proposed_speed_vector * 180/math.pi, reward))
        # reward = abs(self.TOLERANCE_SPEED_VECTOR - abs(self.speed_vector - proposed_speed_vector))
        reward = reward\
                 - self.deviation_speed / self.TOLERANCE_SPEED \
                 - self.deviation_yaw / self.TOLERANCE_YAW \
                - self.deviation_altitude / self.TOLERANCE_ALTITUDE

        # if self.deviation_altitude < 0:
        #     reward = reward - abs(self.deviation_altitude) / self.TOLERANCE_ALTITUDE

        # punish if gs > 8 and gs < -1.5
        # reward = reward - max((self.gs - 8), 0) * 10 - abs(min((self.gs + 1.5), 0)) * 10
        if self.gs > 7:
            reward = reward - (self.gs - 7) * 10
        elif self.gs < -1.5:
            reward = reward - (-1.5 - self.gs) * 10

        # if self.Turn_up_done == True:
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

        elif(abs(self.roll) < self.TOLERANCE_ROLL):
            if ((self.pitch) < self.TOLERANCE_PITCH)\
                    and self.pitch > 0:
                logger.warn("Turn_up_done!")
                return True

        return False


    def finalize(self):
        self.episode += 1
        self.Turn_up_done = False
        if self.episode < 1:
            logger.warn("stop...")
            self.send_ctrl_cmd('2')
            logger.info('stop return...')
        else:
            logger.warn("reboot...")
            self.send_ctrl_cmd('3')
            logger.info('reboot return...')
            self.episode = 0


    # 1: start; 2: stop; 3: reboot
    def send_ctrl_cmd(self, cmd):
        self.bms_control_proxy.RPCserverForGameserver(cmd)