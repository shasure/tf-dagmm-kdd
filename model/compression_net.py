#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class CompressionNet(object):
    def __init__(self, hidden, activation):
        self.hidden = hidden
        self.activation = activation
        self.z_c = None

    def inference(self, X):
        self.X = X
        input_units = X.shape[1]

        with tf.name_scope('comp_net'):
            for nums in self.hidden:
                X = tf.layers.dense(X, nums, activation=self.activation if nums != 1 else None)
                if nums == 1:
                    self.z_c = X
            self.X_rec = tf.layers.dense(X, input_units, activation=None)

            z_r_seq = self.extract_rec_features()
            z = tf.concat([*z_r_seq, self.z_c], axis=1)
        return z

    def loss(self):
        loss_euclid = tf.reduce_mean(tf.reduce_sum(tf.square(self.X - self.X_rec), axis=1))
        return loss_euclid

    def extract_rec_features(self):
        def l2(x):
            return tf.sqrt(tf.reduce_sum(tf.square(x), axis=1, keepdims=True))

        X_l2 = l2(self.X)
        X_rec_l2 = l2(self.X_rec)
        res_l2 = l2(self.X - self.X_rec)
        relative_euclid = res_l2 / X_l2
        cos_similarity = tf.reduce_sum(self.X * self.X_rec, axis=1, keepdims=True) / (X_l2 * X_rec_l2)
        return relative_euclid, cos_similarity
