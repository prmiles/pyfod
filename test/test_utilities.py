#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:14:07 2018

@author: prmiles
"""

import numpy as np
import unittest
from pyfod import utilities

# --------------------------
def exponential_function(x):
    return np.exp(2*x)


class RiemannSumRLFD(unittest.TestCase):
    def test_rs_rl_fd(self):
        alpha = 0.9
        gridpoints = np.linspace(0.0, 1.0, 64)
        df = utilities.riemann_sum_RL_FD(f=exponential_function,
                                         gridpoints=gridpoints, alpha=alpha)
        self.assertTrue(float(df), msg='Expect output to be a float')
