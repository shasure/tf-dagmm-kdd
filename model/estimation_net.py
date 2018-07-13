#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf


class EstimationNet(object):
    def __init__(self, hidden, activation, keep_prob, components):
        self.hidden = hidden
        self.activation = activation
        self.keep_prob = keep_prob
        self.components = components

    def inference(self, z, training):
        X = z
        with tf.name_scope('est_net'):
            for nums in self.hidden:
                X = tf.layers.dense(X, nums, activation=self.activation)
            X = tf.layers.dropout(X, rate=1 - self.keep_prob, training=training)
            X = tf.layers.dense(X, self.components, activation=None)
            gamma = tf.nn.softmax(X)
        return gamma
