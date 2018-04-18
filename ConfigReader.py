#!/usr/bin/env python
# encoding: utf-8
'''
@author: qingyao.wang
@license: (C) Copyright 2017-2020, kuang-chi Corporation Limited.
@contact: qingyao.wang@kuang-chi.com
@file: ConfigRead.py
@time: 2017-09-14 15:48
@desc:
'''
import ConfigParser

class Config(object):
    __instance = None
    def __init__(self):
        self.config = ConfigParser.ConfigParser()
        self.config.readfp(open("config.ini", "rb"))
        #self.config.read("config.ini")

    @staticmethod
    def singleton():
        if Config.__instance:
            return Config.__instance
        else:
            Config.__instance = Config()
            return Config.__instance