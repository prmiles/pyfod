#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:01:37 2019

@author: prmiles
"""


import numpy as np
import sympy as sp
import unittest
from pyfod.GaussLegendreGaussLaguerre import GaussLegendreGaussLaguerre


def f(t):
    try:
        tmp = np.exp(2*t)
    except:
        tmp = sp.exp(2*t)
    return tmp


# --------------------------
class GaussLegendreLaguerreTesting(unittest.TestCase):

    def test_init(self):
        Q = GaussLegendreGaussLaguerre(start=1.0, finish=12.0)
        attributes = ['GLeg', 'GLag', 'f', 'alpha', 'description']
        for att in attributes:
            self.assertTrue(hasattr(Q, att),
                            msg=str('Missing {} attribute'.format(att)))

    def test_update_weights(self):
        Q = GaussLegendreGaussLaguerre(start=1.0, finish=12.0)
        Q.GLeg.weights = []
        Q.GLag.weights = []
        Q.update_weights(alpha=0.0)
        self.assertTrue(isinstance(Q.GLeg.weights, np.ndarray),
                        msg='Weights should be updated.')
        self.assertFalse(isinstance(Q.GLag.weights,
                                   np.ndarray),
                        msg='Weights should be updated.')

    def test_integrate(self):
        Q = GaussLegendreGaussLaguerre(start=0.0, finish=1.0)
        a = Q.integrate(f=f)
        self.assertTrue(isinstance(a, object),
                        msg='Expect object if extended')
        Q = GaussLegendreGaussLaguerre(start=0.0, finish=1.0,
                                       extend_precision=False)
        a = Q.integrate(f=f)
        self.assertTrue(isinstance(a, float),
                        msg='Expect float if not extended')
