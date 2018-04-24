# -*- coding: utf-8 -*-

import numpy as np
import xmlrpclib
import time
from log_config import *
from socket import *
import random
import math
from ConfigReader import Config


# This is spiral task
# 启停的顺序极其重要，不能变
class FlyEnv(object):
    # state: altitude[3000,21000], speed[160,640], roll[-pi, pi], pitch[-pi, pi]
    # action: x(roll), y(pitch), z(speed)
    def __init__(self):
        config_obj = Config.singleton()
        self.bms_control_proxy = xmlrpclib.ServerProxy(config_obj.config.get("BMS_CONTROL", "HTTP"))
        self.fly_proxy = xmlrpclib.ServerProxy(config_obj.config.get("FLY_CONTROL", "HTTP"))
        self.bms_socket = socket(AF_INET, SOCK_DGRAM)
        self.bms_action_addr = (config_obj.config.get("BMS_ACTION", "IP"), int(config_obj.config.get("BMS_ACTION", "PORT")))
        self.step_eps = 0
        self.MAX_STEP = 300
        self.MIN_ALTITUDE = 5000
        self.RANGE_ALTITUDE = (6000, 18000)
        # self.RANGE_ALTITUDE = (10000, 14000)
        # self.RANGE_SPEED = (160, 640)
        # self.RANGE_SPEED = (360, 600)
        self.RANGE_SPEED = (320, 480)
        # self.RANGE_ROLL = (-math.pi, math.pi)
        # self.RANGE_PITCH = (-math.pi, math.pi)
        # self.RANGE_SPEED_VECTOR = (-math.pi, math.pi)
        self.RANGE_ROLL = (72*math.pi/180, 84*math.pi/180)
        # self.RANGE_ROLL = (-72*math.pi/180, -84*math.pi/180)          # opposite
        self.RANGE_PITCH = (-10*math.pi/180, 10*math.pi/180)
        self.RANGE_SPEED_VECTOR = (-10*math.pi/180, 10*math.pi/180)
        self.RANGE_ACTION = (1, 32766)

        self.TOLERANCE_ALTITUDE = 500
        self.TOLERANCE_SPEED = 20
        self.TOLERANCE_ROLL = 5 * math.pi / 180
        self.TOLERANCE_YAW = 5 * math.pi / 180
        self.TOLERANCE_SPEED_VECTOR = 5 * math.pi / 180

        self.more_than_half_circle = False
        self.altitude, self.speed, self.roll, self.pitch, self.speed_vector, self.yaw, self.gs = self.fly_proxy.get_fly_state()


    # "1": start
    # "2": pause
    # "3": restart
    def reset(self):
        logger.warn("reboot...")
        self.send_ctrl_cmd('3')
        logger.info('reset joystick...')
        self.fly_proxy.prepare()
        logger.info('start bms...')
        self.send_ctrl_cmd('1')
        logger.info('start fly control...')
        prestate = self.start_fly()
        logger.info('ai take over fly control...')
        self.step_eps = 0
        self.more_than_half_circle = False
        # state = self.get_state()
        self.altitude, self.speed, self.roll, self.pitch, self.speed_vector, self.yaw, self.gs = self.fly_proxy.get_fly_state()
        logger.debug("altitude: %s, speed: %s, roll: %s, pitch: %s, speed_vector: %s, yaw: %s, gs: %s"
                     % (self.altitude, self.speed, self.roll*180/math.pi, self.pitch*180/math.pi, self.speed_vector*180/math.pi, self.yaw*180/math.pi, self.gs))
        return prestate


    def start_fly(self):
        # self.set_opposite_start_param()     # opposite
        self.set_wide_discrete_start_param()
        fly_state = self.fly_proxy.fly_till(self.altitude_start, self.speed_start, self.roll_start)
        if fly_state is None:
            logger.warn("fly_till fail")
            return None

        self.yaw_start = self.fly_proxy.get_yaw()
        logger.debug("yaw_start: %s" % (self.yaw_start*180/math.pi))

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
        logger.debug("action_x: %d ,action_y: %d, action_z: %d " % (action_x, action_y, action_z))
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


    # 1: start; 2: stop; 3: reboot
    def send_ctrl_cmd(self, cmd):
        self.bms_control_proxy.RPCserverForGameserver(cmd)