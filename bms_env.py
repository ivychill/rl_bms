# -*- coding: utf-8 -*-
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

    # "1": start
    # "2": pause
    # "3": restart
    def reset(self):
        start_time = time.time()
        logger.info('starting...')
        self.fly_proxy.gamectrlFunc('1')
        now = time.time()
        elapse = now - start_time
        logger.info('elapse: %s' % (elapse))
        logger.info('unlocking pause button...')
        self.fly_proxy.gamectrlFunc('2')  # simulate to press the pause key to start combat

    def get_feedback(self):
        state, reward, done = self.fly_proxy.get_feedback()
        logger.debug('state: %s, reward: %s, done: %s' % (state, reward, done))
        return state, reward, done

    def get_flag_entrance(self):
        return self.fly_proxy.get_flag_entrance()

    def step(self, action):
        # self.proxy.step(action)
        # 10个动作依次为：无,仰角上/中/下,扫描角度增/减,扫描线数增/减,TD框左/右
        action_command = ['0', '108', '109', '110', '', '', '', '', '103', '104']
        if action != 0:
            self.bms_socket.sendto(action_command[action], self.bms_addr)