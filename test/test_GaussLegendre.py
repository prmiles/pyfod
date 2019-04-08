# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:14:07 2018

@author: prmiles
"""

import numpy as np
import unittest
from pyfod.GaussLegendre import GaussLegendre


def f(t):
    return np.exp(2*t)


# --------------------------
class GLegTesting(unittest.TestCase):

    def test_init(self):
        GL = GaussLegendre(N=10, start=1.0, finish=12.0)
        attributes = ['points', 'weights', 'f', 'alpha', 'singularity',
                      'initial_weights', 'description']
        for att in attributes:
            self.assertTrue(hasattr(GL, att),
                            msg=str('Missing {} attribute'.format(att)))

    def test_update_weights(self):
        GLeg = GaussLegendre(N=10, start=1.0, finish=12.0)
        GLeg.weights = []
        GLeg.update_weights(alpha=0.0)
        self.assertTrue(isinstance(GLeg.weights, np.ndarray),
                        msg='Weights should be updated.')

    def test_integrate(self):
        GLeg = GaussLegendre(N=10, start=0.0, finish=1.0)
        a = GLeg.integrate(f=f)
        self.assertTrue(isinstance(a, float), msg='Expect float')


# --------------------------
class BaseGaussPoints(unittest.TestCase):

    def test_base_gauss_points(self):
        GQ = GaussLegendre(N=10, start=1.0, finish=12.0)
        gpts = GQ._base_gauss_points()
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
        GQ = GaussLegendre(N=10, start=1.0, finish=12.0)
        gwts = GQ._base_gauss_weights(h=0.1)
        mlgwts = np.array([0.017392742256873, 0.032607257743127,
                           0.032607257743127, 0.017392742256873]).T
        self.assertEqual(gwts.size, 4, msg='Expect 4 elements')
        self.assertEqual(gwts.shape, (4,), msg='Expect shape = (4,)')
        self.assertTrue(np.allclose(gwts, mlgwts),
                        msg=str('Expect arrays to match: {} neq {}'.format(
                                gwts, mlgwts)))


# --------------------------
class IntervalGaussPoints(unittest.TestCase):

    def test_interval_gauss_points(self):
        GQ = GaussLegendre(N=10, start=1.0, finish=12.0)
        igp = GQ._interval_gauss_points(
                base_gpts=GQ._base_gauss_points(),
                N=4, h=0.1, start=0.)
        self.assertEqual(igp.shape, (4*4,), msg='Expect shape = (16,)')


# --------------------------
class GaussPoints(unittest.TestCase):

    def test_gauss_points(self):
        GQ = GaussLegendre(N=10, start=1.0, finish=12.0)
        gpoints = GQ._gauss_points(N=4, h=0.1, start=0.)
        self.assertEqual(gpoints.shape, (4*4,), msg='Expect shape = (16,)')


# --------------------------
class GaussWeights(unittest.TestCase):

    def test_gauss_weights(self):
        GQ = GaussLegendre(N=10, start=1.0, finish=12.0)
        gweights = GQ._gauss_weights(N=4, h=0.1)
        self.assertEqual(gweights.shape, (4*4,), msg='Expect shape = (16,)')
