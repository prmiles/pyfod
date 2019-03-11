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
class BaseGaussPoints(unittest.TestCase):

    def test_base_gauss_points(self):
        gpts = utilities.base_gauss_points()
        mlgpts = np.array([0.069431844202974, 0.330009478207572,
                           0.669990521792428, 0.930568155797026])
        self.assertEqual(gpts.size, 4, msg='Expect 4 elements')
        self.assertEqual(gpts.shape, (4,), msg='Expect shape = (4,)')
        self.assertTrue(np.allclose(gpts, mlgpts),
                        msg=str('Expect arrays to match: {} neq {}'.format(
                                gpts,
                                mlgpts)))


# --------------------------
class BaseGaussWeights(unittest.TestCase):

    def test_base_gauss_weights(self):
        gwts = utilities.base_gauss_weights(h=np.array([0.1]))
        mlgwts = np.array([[0.017392742256873, 0.032607257743127,
                            0.032607257743127, 0.017392742256873]]).T
        self.assertEqual(gwts.size, 4, msg='Expect 4 elements')
        self.assertEqual(gwts.shape, (4, 1), msg='Expect shape = (4,1)')
        self.assertTrue(np.allclose(gwts, mlgwts),
                        msg=str('Expect arrays to match: {} neq {}'.format(
                                gwts, mlgwts)))


# --------------------------
class IntervalGaussPoints(unittest.TestCase):

    def test_interval_gauss_points(self):
        igp = utilities.interval_gauss_points(
                base_gpts=utilities.base_gauss_points(),
                N=4, h=np.array([0.1, 0.2]), low_lim=0.)
        self.assertEqual(igp.shape, (4*4, 2), msg='Expect shape = (16,2)')


# --------------------------
class GaussPoints(unittest.TestCase):

    def test_gauss_points(self):
        gpoints = utilities.gauss_points(N=4,
                                         h=np.array([0.1, 0.2]), low_lim=0.)
        self.assertEqual(gpoints.shape, (4*4, 2), msg='Expect shape = (16,2)')


# --------------------------
class GaussWeights(unittest.TestCase):

    def test_gauss_weights(self):
        gweights = utilities.gauss_weights(N=4, h=np.array([0.1, 0.2]))
        self.assertEqual(gweights.shape, (4*4, 2), msg='Expect shape = (16,2)')


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
