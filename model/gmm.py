#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class GMM(object):
    def __init__(self):
        pass

    def inference(self, gamma, z):
        # gamma: N*K; z: N*D
        gamma_sum = tf.reduce_sum(gamma, axis=0)  # K
        self.phi = phi = tf.reduce_mean(gamma, axis=0)  # K
        self.mu = mu = tf.reduce_sum(tf.expand_dims(gamma, axis=-1) * tf.expand_dims(z, axis=1),
                                     axis=0) / tf.expand_dims(gamma_sum, axis=-1)  # K*D / K*1 -> K*D
        z_mu = tf.expand_dims(z, axis=1) - tf.expand_dims(mu, axis=0)  # N*1*D - 1*K*D -> N*K*D
        z_mu_mul = tf.expand_dims(z_mu, axis=-1) * tf.expand_dims(z_mu, axis=-2)  # N*K*D*1 * N*K*1*D -> N*K*D*D
        sigma = tf.reduce_sum(tf.expand_dims(tf.expand_dims(gamma, -1), -1) * z_mu_mul, axis=0) / tf.expand_dims(
            tf.expand_dims(gamma_sum, -1), -1)  # K*D*D
        # sigma + epsilon
        eps = 1e-12
        diag_eps = tf.diag(tf.ones(z.shape[-1])) * eps  # D*D
        diag_eps = tf.expand_dims(diag_eps, axis=0)  # 1*D*D
        self.sigma = sigma_eps = sigma + diag_eps  # K*D*D
        return phi, mu, sigma_eps

    def energy(self, z, phi, mu, sigma_eps):
        # z: N*D
        # phi: K
        # mu: K*D
        # sigma: K*D*D
        sigma_inverse = tf.matrix_inverse(sigma_eps)  # K*D*D
        z_mu = tf.expand_dims(z, axis=1) - tf.expand_dims(mu, axis=0)  # N*1*D - 1*K*D -> N*K*D
        exp_val_tmp = -0.5 * tf.reduce_sum(
            tf.reduce_sum(tf.expand_dims(z_mu, -1) * tf.expand_dims(sigma_inverse, 0), -2) * z_mu, -1)  # N*K
        det_sigma = tf.matrix_determinant(sigma_eps)  # K
        log_det_simga = tf.expand_dims(tf.log(tf.sqrt(2 * np.pi * det_sigma)), 0)  # K
        log_phi = tf.expand_dims(tf.log(phi), 0)  # 1*K
        exp_val = log_phi + exp_val_tmp - log_det_simga  # 1*K + N*k - 1*K -> N*K
        energies = -tf.reduce_logsumexp(exp_val, axis=1)  # N
        return energies

    def loss(self, energies):
        energy_mean = tf.reduce_mean(energies)
        diag_loss = tf.reduce_sum(1 / tf.matrix_diag_part(self.sigma))  # reduce_sum(K*D)
        return energy_mean, diag_loss

    def assign_gmm_params(self, K, D):
        self.create_variables(K, D)
        params_assign_op = tf.group(tf.assign(self.phi_var, self.phi), tf.assign(self.mu_var, self.mu),
                                    tf.assign(self.sigma_var, self.sigma))
        return params_assign_op

    def create_variables(self, K, D):
        self.phi_var = tf.Variable(tf.zeros(shape=(K,)), trainable=False)
        self.mu_var = tf.Variable(tf.zeros(shape=(K, D)), trainable=False)
        self.sigma_var = tf.Variable(tf.zeros(shape=(K, D, D)), trainable=False)
