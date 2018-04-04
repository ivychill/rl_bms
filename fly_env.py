# -*- coding: utf-8 -*-

import numpy as np
import xmlrpclib
import time
from log_config import *
from socket import *
import random
import math


# This is spiral task
# 启停的顺序极其重要，不能变
class FlyEnv:
    # state: altitude[3000,21000], speed[160,640], roll[-pi, pi], pitch[-pi, pi]
    # action: x(roll), y(pitch), z(speed)
    def __init__(self):
        self.bms_control_proxy = xmlrpclib.ServerProxy("http://192.168.24.72:8000/")
        self.fly_proxy = xmlrpclib.ServerProxy("http://192.168.20.129:4022/")
        # self.image_proxy = xmlrpclib.ServerProxy("http://192.168.20.129:5001/")
        self.bms_socket = socket(AF_INET, SOCK_DGRAM)
        self.bms_action_addr = ("192.168.24.72", 4001)
        self.episode = 0
        self.step_eps = 0
        # self.RANGE_ALTITUDE = (3000, 21000)
        # self.RANGE_ALTITUDE = (6000, 18000)
        self.RANGE_ALTITUDE = (10000, 14000)
        # self.RANGE_SPEED = (160, 640)
        self.RANGE_SPEED = (320, 480)
        # self.RANGE_ROLL = (-math.pi, math.pi)
        # self.RANGE_PITCH = (-math.pi, math.pi)
        # self.RANGE_SPEED_VECTOR = (-math.pi, math.pi)
        self.RANGE_ROLL = (math.pi*11/36, math.pi*19/36)
        self.RANGE_PITCH = (-math.pi/9, math.pi/9)
        self.RANGE_SPEED_VECTOR = (-math.pi/9, math.pi/9)
        self.RANGE_ACTION = (1, 32766)

        # self.altitude_start = self.RANGE_ALTITUDE[0] + (self.RANGE_ALTITUDE[1] - self.RANGE_ALTITUDE[0]) * random.uniform(0, 1)
        # self.speed_start = self.RANGE_SPEED[0] + (self.RANGE_SPEED[1] - self.RANGE_SPEED[0]) * random.uniform(0, 1)
        # self.roll_start = self.RANGE_ROLL[0] + (self.RANGE_ROLL[1] - self.RANGE_ROLL[0]) * random.uniform(0, 1)
        self.altitude_start = 12000
        self.speed_start = 400
        self.roll_start = 75 * math.pi / 180

        self.TOLERANCE_ALTITUDE = 500
        self.TOLERANCE_SPEED = 20
        self.TOLERANCE_ROLL = 5 * math.pi / 180
        self.TOLERANCE_YAW = 5 * math.pi / 180
        self.TOLERANCE_SPEED_VECTOR = 5 * math.pi / 180

        self.more_than_half_circle = False

        self.altitude, self.speed, self.roll, self.pitch, self.speed_vector, self.yaw, self.gs = self.fly_proxy.get_fly_state()

        self.send_ctrl_cmd('3')


    # "1": start
    # "2": pause
    # "3": restart
    # TODO:
    def reset(self):
        logger.info('reset joystick...')
        self.fly_proxy.prepare()
        logger.info('start bms...')
        self.send_ctrl_cmd('1')
        logger.info('start fly control...')
        prestate = self.start_fly()
        logger.info('reset over...')
        self.step_eps = 0
        # state = self.get_state()
        # used for DDPG exploration input
        self.altitude, self.speed, self.roll, self.pitch, self.speed_vector, self.yaw, self.gs = self.fly_proxy.get_fly_state()
        logger.debug("altitude: %s, speed: %s, roll: %s, pitch: %s, speed_vector: %s, yaw: %s, gs: %s"
                     % (self.altitude, self.speed, self.roll*180/math.pi, self.pitch*180/math.pi, self.speed_vector*180/math.pi, self.yaw*180/math.pi, self.gs))
        return prestate


    def start_fly(self):
        fly_state = self.fly_proxy.fly_till(self.altitude_start, self.speed_start, self.roll_start)
        self.yaw_start = self.fly_proxy.get_yaw()
        logger.debug("yaw_start: %s" % (self.yaw_start*180/math.pi))
        # logger.debug("altitude_start: %s, speed_start: %s, roll_start: %s, yaw_start: %s"
        #              % (self.altitude_start, self.speed_start, self.roll_start*180/math.pi, self.yaw_start*180/math.pi))

        for index_i in range(4):
            fly_state[index_i * 5] = (fly_state[index_i * 5] - self.RANGE_ALTITUDE[0])/(self.RANGE_ALTITUDE[1]- self.RANGE_ALTITUDE[0])
            fly_state[index_i * 5 + 1] = (fly_state[index_i * 5 + 1] - self.RANGE_SPEED[0])/(self.RANGE_SPEED[1]- self.RANGE_SPEED[0])
            fly_state[index_i * 5 + 2] = (fly_state[index_i * 5 + 2] - self.RANGE_ROLL[0])/(self.RANGE_ROLL[1]- self.RANGE_ROLL[0])
            fly_state[index_i * 5 + 3] = (fly_state[index_i * 5 + 3] - self.RANGE_PITCH[0]) / (self.RANGE_PITCH[1] - self.RANGE_PITCH[0])
            fly_state[index_i * 5 + 4] = (fly_state[index_i * 5 + 4] - self.RANGE_SPEED_VECTOR[0]) / (self.RANGE_SPEED_VECTOR[1] - self.RANGE_SPEED_VECTOR[0])

        logger.debug("fly_state: %s" % (fly_state))
        logger.debug("normalized fly_state: %s" % (fly_state))

        pre_state = np.array(fly_state)
        return pre_state


    # done: (altitude, speed, roll) = inception (altitude, speed, roll)
    # Reward：-1/step,
    def step(self, action):
        self.take_action(action)
        self.step_eps += 1
        state = self.get_state()
        if self.more_than_half_circle == False:
            if (abs(self.yaw - self.yaw_start + math.pi) < self.TOLERANCE_YAW or abs(self.yaw - self.yaw_start - math.pi) < self.TOLERANCE_YAW):
                self.more_than_half_circle = True
        reward = self.get_reward()
        done = self.get_done()
        return state, reward, done, {}


    def take_action(self, action):
        action_x = round(self.RANGE_ACTION[0] + (self.RANGE_ACTION[1] - self.RANGE_ACTION[0]) * action[0])
        action_y = round(self.RANGE_ACTION[0] + (self.RANGE_ACTION[1] - self.RANGE_ACTION[0]) * action[1])
        action_z = round(self.RANGE_ACTION[0] + (self.RANGE_ACTION[1] - self.RANGE_ACTION[0]) * action[2])
        logger.debug("action_x: %s, action_y: %s, action_z: %s" % (action_x, action_y, action_z))
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
        normalized_roll = (self.roll - self.RANGE_ROLL[0])/(self.RANGE_ROLL[1]- self.RANGE_ROLL[0])
        normalized_pitch = (self.pitch - self.RANGE_PITCH[0]) / (self.RANGE_PITCH[1] - self.RANGE_PITCH[0])
        normalized_speed_vector = (self.speed_vector - self.RANGE_SPEED_VECTOR[0]) / (self.RANGE_SPEED_VECTOR[1] - self.RANGE_SPEED_VECTOR[0])
        # self.yaw = self.fly_proxy.get_yaw()
        # self.gs = self.fly_proxy.get_gs()

        logger.debug("altitude: %s, speed: %s, roll: %s, pitch: %s, speed_vector: %s, yaw: %s, gs: %s"
                     % (self.altitude, self.speed, self.roll*180/math.pi, self.pitch*180/math.pi, self.speed_vector*180/math.pi, self.yaw*180/math.pi, self.gs))

        # self.deviation_altitude = self.altitude/self.altitude_start - 1
        # self.deviation_speed = self.speed/self.speed_start - 1
        # self.deviation_roll = self.roll/self.roll_start - 1
        self.deviation_altitude = abs(self.altitude-self.altitude_start)
        self.deviation_speed = abs(self.speed-self.speed_start)
        self.deviation_roll = abs(self.roll-self.roll_start)

        return np.asarray([normalized_altitude, normalized_speed, normalized_roll, normalized_pitch, normalized_speed_vector])


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


    def finalize(self):
        self.episode += 1
        self.more_than_half_circle = False
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