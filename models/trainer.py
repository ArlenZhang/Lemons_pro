# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description: Deal with training and validation
"""


class Trainer:
    def __init__(self, train_dt=None, dev_dt=None):
        self.train_dt = train_dt
        self.dev_dt = dev_dt

    def do_train(self):
        ...

    def do_eval(self):
        ...
