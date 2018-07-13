#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from model.compression_net import CompressionNet
from model.dataset import gen_train_valid_data
from model.estimation_net import EstimationNet
from model.gmm import GMM


class DAGMM(object):
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.sess = None
        # get train/valid data for train/valid function
        self.X_train, self.X_test, self.y_train, self.y_test = gen_train_valid_data()
        # standard scale
        self.scaler = scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        # init compression net, estimation net and gmm
        self.cn = CompressionNet(hyperparams.comp_hidden, hyperparams.comp_activation)
        self.en = EstimationNet(hyperparams.est_hidden, hyperparams.est_activation, hyperparams.keep_prob,
                                hyperparams.mix_components)
        self.gmm = GMM()

    def build_graph(self):
        with tf.Graph().as_default() as graph:
            params_assign_op, energies, energies_eval = self.inference()
            global_step = tf.train.get_or_create_global_step()
            loss = self.loss(self.hyperparams.lambda1, self.hyperparams.lambda2, energies)

            optimizer = tf.train.AdamOptimizer(self.hyperparams.lr)
            train_op = optimizer.minimize(loss, global_step=global_step)
            # Create tensorflow session and initilize
            init_op = tf.global_variables_initializer()
            # local_init = tf.local_variables_initializer()
        return graph, init_op, train_op, params_assign_op, loss, energies_eval, global_step

    def inference(self):
        self.input = input = tf.placeholder(tf.float32, shape=(None, self.X_train.shape[1]))
        self.training = training = tf.placeholder(tf.bool, shape=())
        z = self.cn.inference(input)
        gamma = self.en.inference(z, training)
        phi, mu, sigma = self.gmm.inference(gamma, z)
        energies = self.gmm.energy(z, phi, mu, sigma)
        params_assign_op = self.gmm.assign_gmm_params(mu.shape[0], mu.shape[1])
        # energies for eval
        energies_eval = self.gmm.energy(z, self.gmm.phi_var, self.gmm.mu_var, self.gmm.sigma_var)

        return params_assign_op, energies, energies_eval

    def loss(self, lambda1, lambda2, energies):
        loss_euclid = self.cn.loss()
        energy_mean, diag_loss = self.gmm.loss(energies)
        total_loss = loss_euclid + lambda1 * energy_mean + lambda2 * diag_loss
        return total_loss

    def train(self, valid=False):
        self.graph, init_op, train_op, self.params_assign_op, self.loss_op, self.energies_eval, global_step = self.build_graph()

        epoch = self.hyperparams.epoch
        batch_size = self.hyperparams.batch_size
        n_samples = self.X_train.shape[0]
        steps = (n_samples // batch_size)
        self.sess = sess = tf.Session(graph=self.graph)

        sess.run(init_op)

        for e in range(epoch):
            for step in range(steps):
                start = step * batch_size
                end = (step + 1) * batch_size
                X_batch = self.X_train[start:end]
                _, loss, global_step_val = sess.run([train_op, self.loss_op, global_step],
                                                    feed_dict={self.input: X_batch, self.training: True})

                if global_step_val % 300 == 0:
                    print('global step: {0}\t\t\tloss: {1:.3f}'.format(global_step_val, loss, ))
            if valid and e % 30 == 0:
                self.valid()
        if valid:
            self.valid()

    def valid(self):
        # update phi, mu, sigma with whole training data
        self.sess.run(self.loss_op, feed_dict={self.input: self.X_train, self.training: False})
        # assign phi, mu, sigma vars
        self.sess.run(self.params_assign_op, feed_dict={self.input: self.X_train, self.training: False})
        energies = self.sess.run(self.energies_eval, feed_dict={self.input: self.X_test, self.training: False})

        anomaly_energy_threshold = np.percentile(energies, 80)
        print("Energy threshold to detect anomaly : {0:.3f}".format(anomaly_energy_threshold))
        y_pred = np.where(energies >= anomaly_energy_threshold, 1, 0)
        precision, recall, fscore, _ = precision_recall_fscore_support(self.y_test, y_pred, average='binary')
        print(" Precision = {0:.3f}".format(precision))
        print(" Recall    = {0:.3f}".format(recall))
        print(" F1-Score  = {0:.3f}".format(fscore))
