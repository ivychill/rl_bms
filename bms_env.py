# -*- coding: utf-8 -*-

import numpy as np
import xmlrpclib
# import numpy as np
from log_config import *
import time
from socket import *

class BmsEnv:
    def __init__(self):
        # self.fly_proxy = xmlrpclib.ServerProxy("http://192.168.24.116:4022/")
        self.bms_control_proxy = xmlrpclib.ServerProxy("http://192.168.20.122:8000/")
        self.fly_proxy = xmlrpclib.ServerProxy("http://192.168.20.129:4022/")
        self.image_proxy = xmlrpclib.ServerProxy("http://192.168.20.129:5001/")
        # self.image_proxy = xmlrpclib.ServerProxy("http://192.168.24.108:5001/")
        self.bms_socket = socket(AF_INET, SOCK_DGRAM)
        self.bms_addr = ("92.168.24.92", 4001)
        self.episode = 0
        self.step_eps = 0

    # "1": start
    # "2": pause
    # "3": restart
    def reset(self):
        logger.info('reset...')
        self.send_ctrl_cmd('1')
        self.image_proxy.start()
        self.fly_proxy.start()
        self.step_eps = 0
        state = self.get_state()
        return state

    # 结束条件：敌机出现在屏幕，1000
    # step
    # Reward：-1 / step，出现在屏幕 + 1000
    def step(self, action):
        # self.proxy.step(action)
        # 8个动作依次为：无,仰角上/中/下,扫描角度,扫描线数,TD框左/右
        action_command = ['0', 'K:108', 'K:109', 'K:110', 'K:171', 'K:170', 'K:103', 'K:104']
        if action != 0:
            self.bms_socket.sendto(action_command[action], self.bms_addr)

        state = self.get_state()

        # done = False
        self.step_eps += 1
        reward = -1

        enemy_coord = self.image_proxy.get_enemy_coord()

        if self.step_eps >= 1000:
            self.finalize()
            done = True
        elif enemy_coord is not None:
            self.finalize()
            done = True
            reward += 1000
        else:
            done = False

        return state, reward, done, {}

    def get_state(self):
        z, speed, pitch, yaw = self.fly_proxy.get_fly_state()
        td_topleft, upper, lower = self.image_proxy.get_td()
        state = np.asarray([z, speed, pitch, yaw, td_topleft[0], td_topleft[1], upper, lower])
        return state

    def finalize(self):
        self.fly_proxy.stop()
        self.image_proxy.stop()
        self.send_ctrl_cmd('2')
        self.episode += 1
        if self.episode >= 20:
            self.send_ctrl_cmd('3')
            self.episode = 0

    # # 1: start; 2: stop; 3: reboot
    # def send_ctrl_cmd(self, cmd):
    #     self.bms_control_proxy.RPCserverForGameserver(cmd)