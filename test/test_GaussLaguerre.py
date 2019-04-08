# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:14:07 2018

@author: prmiles
"""

import sympy as sp
import numpy as np
import unittest
from pyfod.GaussLaguerre import GaussLaguerre


def f(t):
    return np.cos(t)


def fsp(t):
    return sp.cos(t)


# --------------------------
class GaussLaguerreTesting(unittest.TestCase):

    def test_init(self):
        Q = GaussLaguerre(N=10, start=1.0, finish=12.0)
        attributes = ['points', 'weights', 'f', 'alpha',
                      'initial_weights', 'description']
        for att in attributes:
            self.assertTrue(hasattr(Q, att),
                            msg=str('Missing {} attribute'.format(att)))

    def test_update_weights(self):
        Q = GaussLaguerre(N=10, start=1.0, finish=12.0, extend_precision=False)
        Q.weights = []
        Q.update_weights(alpha=0.0)
        self.assertTrue(isinstance(Q.weights, np.ndarray),
                        msg='Weights should be updated.')

    def test_integrate(self):
        Q = GaussLaguerre(N=10, start=0.0, finish=1.0, extend_precision=False)
        a = Q.integrate(f=f)
        self.assertTrue(isinstance(a, float), msg='Expect float')


# --------------------------
class Initialization_non_extended(unittest.TestCase):

    def test_init(self):
        GQ = GaussLaguerre(N=10, start=1.0, finish=12.0,
                           extend_precision=False)
        self.assertTrue(hasattr(GQ, 'f'), msg='Expect attribute f to exist')
        self.assertEqual(GQ.f, None, msg='Expect value of None')
        self.assertEqual(GQ.points.size, 10, msg='Expect 10 nodes')
        self.assertEqual(GQ.weights.size, 10, msg='Expect 10 weights')
        self.assertEqual(GQ.alpha, 0, msg='Expect alpha eq 0')

    def test_init_with_f(self):
        GQ = GaussLaguerre(N=10, start=1.0, finish=12.0, alpha=0.5,
                           f=f, extend_precision=False)
        self.assertTrue(hasattr(GQ, 'f'), msg='Expect attribute f to exist')
        self.assertEqual(GQ.f, f, msg='Expect function f')
        self.assertEqual(GQ.points.size, 10, msg='Expect 10 nodes')
        self.assertEqual(GQ.weights.size, 10, msg='Expect 10 weights')
        self.assertEqual(GQ.alpha, 0.5, msg='Expect alpha eq 0.5')


# --------------------------
class Initialization_with_extended(unittest.TestCase):

    def test_init(self):
        GQ = GaussLaguerre(N=10, start=1.0, finish=12.0)
        self.assertTrue(hasattr(GQ, 'f'), msg='Expect attribute f to exist')
        self.assertEqual(GQ.f, None, msg='Expect value of None')
        self.assertEqual(len(GQ.points), 10, msg='Expect 10 nodes')
        self.assertEqual(len(GQ.weights), 10, msg='Expect 10 weights')
        self.assertEqual(GQ.alpha, 0, msg='Expect alpha eq 0')

    def test_init_with_f(self):
        GQ = GaussLaguerre(N=10, start=1.0, finish=12.0, alpha=0.5, f=fsp)
        self.assertTrue(hasattr(GQ, 'f'), msg='Expect attribute f to exist')
        self.assertEqual(GQ.f, fsp, msg='Expect function fsp')
        self.assertEqual(len(GQ.points), 10, msg='Expect 10 nodes')
        self.assertEqual(len(GQ.weights), 10, msg='Expect 10 weights')
        self.assertEqual(GQ.alpha, 0.5, msg='Expect alpha eq 0.5')


# --------------------------
class Integrate_non_extended(unittest.TestCase):

    def test_no_f(self):
        GQ = GaussLaguerre(extend_precision=False)
        with self.assertRaises(SystemExit):
            GQ.integrate()

    def test_with_f(self):
        GQ = GaussLaguerre(extend_precision=False)
        a = GQ.integrate(f=f)
        self.assertEqual(a.size, 1, msg='Expect float return')

    def test_with_alpha(self):
        GQ = GaussLaguerre(extend_precision=False)
        a = GQ.integrate(f=f)
        self.assertEqual(a.size, 1, msg='Expect float return')


# --------------------------
class Integrate_with_extended(unittest.TestCase):

    def test_no_f(self):
        GQ = GaussLaguerre()
        with self.assertRaises(SystemExit):
            GQ.integrate()

    def test_with_f(self):
        GQ = GaussLaguerre()
        a = GQ.integrate(f=fsp)
        self.assertTrue(a.is_Float, msg='Expect float return')

    def test_with_alpha(self):
        GQ = GaussLaguerre()
        a = GQ.integrate(f=fsp)
        self.assertTrue(a.is_Float, msg='Expect float return')
