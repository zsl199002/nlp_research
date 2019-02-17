#!/usr/bin/python
# -*-encoding:utf8-*-
# @Time     :22/1/19
# @Author   :CopyBug
# @Mail     :zsl199002@yeah.net

"""
Description: a implementation of paper "Gradient Harmonized Single-stage Detector" published on AAAI2019
"""

import numpy as np
import tensorflow as tf

class GHMC(object):
    """
    classific loss class for GHM
    """
    def __init__(self, bins=10, momentum=0):
        self.bins = bins
        self.momentum = momentum
        self.edges = np.linspace(0, 1, self.bins+1)
        self.edges[-1] += 1e-6
        self.edges =  tf.constant(self.edges)
        if self.momentum > 0:
            self.acc_sum = [0]*self.bins

    def __call__(self, predict, target, *args, **kwargs):
        with tf.variable_scope('ghmc'):
            g = tf.norm(target-predict, axis=-1)
            for i in range(self.bins):
                pass



        pass

