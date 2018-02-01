# -*- coding: utf-8 -*-

import numpy as np
import xmlrpclib
import time
from log_config import *
from socket import *

class BmsEnv:
    def __init__(self):
        # self.fly_proxy = xmlrpclib.ServerProxy("http://192.168.24.116:4022/")
        self.bms_control_proxy = xmlrpclib.ServerProxy("http://192.168.20.122:8000/")
        self.fly_proxy = xmlrpclib.ServerProxy("http://192.168.20.129:4022/")
        self.image_proxy = xmlrpclib.ServerProxy("http://192.168.20.129:5001/")
        # self.image_proxy = xmlrpclib.ServerProxy("http://192.168.24.108:5001/")
        self.bms_socket = socket(AF_INET, SOCK_DGRAM)
        self.bms_addr = ("192.168.20.122", 4001)
        self.episode = 0
        self.step_eps = 0
        self.enemy_appear_ever = False
        self.enemy_appear_last = time.time()

    # "1": start
    # "2": pause
    # "3": restart
    def reset(self):
        logger.info('reset...')
        # self.fly_proxy.prepare()
        self.fly_proxy.start()
        self.image_proxy.start()
        self.send_ctrl_cmd('1')
        logger.info('reset return...')
        self.enemy_appear_ever = False
        self.enemy_appear_last = time.time()
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

        # done的条件为超过500步,飞机出现在MFD中至少2秒,飞机被击中,z<=4000
        if self.step_eps >= 500:
            logger.warn("reach 500 steps, done!")
            self.finalize()
            done = True
        elif state[0] <= 4000:  # z = state[0]
            logger.warn("latitude less than 1000, done!")
            self.finalize()
            done = True
        elif self.fly_proxy.is_dead():
            logger.warn("shot by enemy, done!")
            self.finalize()
            done = True
        elif enemy_coord is not None:
            logger.debug("enemy appear...")
            if time.time() - self.enemy_appear_last >= 2 and self.enemy_appear_ever:
                logger.warn("enemy appear for more than 2 seconds, done!")
                self.finalize()
                done = True
                reward += 500
            else:
                done = False
            self.enemy_appear_ever = True
        else:
            logger.debug("enemy disappear...")
            self.enemy_appear_last = time.time()
            self.enemy_appear_ever = False
            done = False

        return state, reward, done, {}

    def get_state(self):
        z, speed, pitch, yaw = self.fly_proxy.get_fly_state()
        # td_topleft = self.image_proxy.get_td()
        # upper, lower = self.image_proxy.get_td_high_low()
        # while td_topleft is None\
        #         or upper is None\
        #         or lower is None:
        #     logger.warn("get td_topleft or upper or lower none")
        #     time.sleep(0.05)
        #     td_topleft = self.image_proxy.get_td()
        #     upper, lower = self.image_proxy.get_td_high_low()
        # state = np.asarray([z, speed, pitch, yaw, td_topleft[0], td_topleft[1], upper, lower])
        state = np.asarray([z, speed, pitch, yaw])
        logger.debug("state: %s" % (state))
        return state

    def finalize(self):
        self.episode += 1
        if self.episode < 5:
            logger.warn("stop...")
            self.fly_proxy.stop()
            self.image_proxy.stop()
            self.send_ctrl_cmd('2')
            logger.info('stop return...')
        else:
            logger.warn("reboot...")
            self.fly_proxy.reboot()
            self.image_proxy.reboot()
            self.send_ctrl_cmd('3')
            logger.info('reboot return...')
            self.episode = 0

    # 1: start; 2: stop; 3: reboot
    def send_ctrl_cmd(self, cmd):
        self.bms_control_proxy.RPCserverForGameserver(cmd)