# -*- coding: utf-8 -*-

import numpy as np
import xmlrpclib
# import numpy as np
from log_config import *
import time
from socket import *

class BmsEnv:
    def __init__(self):
        self.fly_proxy = xmlrpclib.ServerProxy("http://192.168.20.127:8180/")
        self.image_proxy = xmlrpclib.ServerProxy("http://192.168.24.108:5001/")
        self.bms_socket = socket(AF_INET, SOCK_DGRAM)
        self.bms_addr = ("92.168.24.92", 4001)
        self.step_eps = 0

    # "1": start
    # "2": pause
    # "3": restart
    def reset(self):
        logger.info('reset...')
        # start_time = time.time()
        # self.fly_proxy.gamectrlFunc('1')
        # now = time.time()
        # elapse = now - start_time
        # logger.info('elapse: %s' % (elapse))
        # logger.info('unlocking pause button...')
        self.step_eps = 0

    # 结束条件：敌机出现在屏幕，1000
    # step
    # Reward：-1 / step，出现在屏幕 + 1000
    def step(self, action):
        # self.proxy.step(action)
        # 10个动作依次为：无,仰角上/中/下,扫描角度增/减,扫描线数增/减,TD框左/右
        action_command = ['0', '108', '109', '110', '', '', '', '', '103', '104']
        if action != 0:
            self.bms_socket.sendto(action_command[action], self.bms_addr)

        z, speed, pitch, yaw = self.fly_proxy.get_fly_state()
        td_topleft, upper, lower = self.image_proxy.get_td()
        state = np.asarray([z, speed, pitch, yaw, td_topleft, upper, lower])

        # done = False
        self.step_eps += 1
        reward = -1

        enemy_coord = self.image_proxy.get_enemy_coord()

        if self.step_eps >= 1000:
            done = True
        elif enemy_coord is not None:
            done = True
            reward += 1000
        else:
            done = False

        return state, reward, done, {}