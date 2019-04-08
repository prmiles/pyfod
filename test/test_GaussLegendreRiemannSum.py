# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:14:07 2018

@author: prmiles
"""

import numpy as np
import unittest
from pyfod.GaussLegendreRiemannSum import GaussLegendreRiemannSum


def f(t):
    return np.exp(2*t)


# --------------------------
class GaussRiemannSumTesting(unittest.TestCase):

    def test_init(self):
        Q = GaussLegendreRiemannSum(start=1.0, finish=12.0)
        attributes = ['GQ', 'RS', 'f', 'alpha', 'description']
        for att in attributes:
            self.assertTrue(hasattr(Q, att),
                            msg=str('Missing {} attribute'.format(att)))

    def test_update_weights(self):
        Q = GaussLegendreRiemannSum(start=1.0, finish=12.0)
        Q.GQ.weights = []
        Q.RS.weights = []
        Q.update_weights(alpha=0.0)
        self.assertTrue(isinstance(Q.GQ.weights, np.ndarray),
                        msg='Weights should be updated.')
        self.assertTrue(isinstance(Q.RS.weights, np.ndarray),
                        msg='Weights should be updated.')

    def test_integrate(self):
        Q = GaussLegendreRiemannSum(start=0.0, finish=1.0)
        a = Q.integrate(f=f)
        self.assertTrue(isinstance(a, float), msg='Expect float')
