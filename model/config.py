#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple

import tensorflow as tf

params_dict = {'batch_size': 1024,
               'epoch': 200,
               'comp_hidden': [60, 30, 10, 1, 10, 30, 60],
               'comp_activation': tf.nn.tanh,
               'est_hidden': [10, ],
               'est_activation': tf.nn.tanh,
               'keep_prob': 0.5,
               'mix_components': 4,
               'lambda1': 0.1,
               'lambda2': 0.005,
               # 'lambda2': 0.0001,
               'lr': 0.0001, }

Config = namedtuple('Config', params_dict.keys())

# model variable
hyperparams = Config(**params_dict)

# print(params)
# print(params.batch_size)
# print(params.epoch)
