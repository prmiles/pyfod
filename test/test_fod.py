# -*- coding: utf-8 -*-
import unittest
from pyfod import fod
from pyfod.fod import riemannliouville as rlou
from pyfod.fod import caputo as cap
from pyfod.fod import grunwaldletnikov as glet
import numpy as np
import sympy as sp


def fexp(t):
    return np.exp(2*t)


def fsp(t):
    return sp.exp(2*t)


# --------------------------
class SelectQuadratureMethodTesting(unittest.TestCase):

    def test_selection(self):
        with self.assertRaises(SystemExit):
            fod.select_quadrature_method(quadrature='hello')


# --------------------------
class RiemannLiouville(unittest.TestCase):

    def check_contents(self, out):
        keys = ['fd', 'I1', 'I2', 'Q1', 'Q2']
        for key in keys:
            self.assertTrue(key in out,
                            msg=str('{} not in output'.format(key)))
            self.assertTrue(isinstance(out['fd'], float),
                            msg='Expect float')

    def test_fod(self):
        start = 0.0
        finish = 1.0
        NRS = 10
        # Test alpha = 0.0
        alpha = 0.0
        out = rlou(f=fexp, alpha=alpha, start=start,
                      finish=finish, NRS=NRS)
        self.check_contents(out)
        self.assertTrue(isinstance(out['fd'], float),
                        msg='Expect float return')
        out = rlou(f=fsp, alpha=alpha, start=start,
                      finish=finish, quadrature='glag')
        self.check_contents(out)
        self.assertTrue(isinstance(out['fd'], float),
                        msg='Expect float return')


# --------------------------
class Caputo(unittest.TestCase):

    def check_contents(self, out):
        keys = ['fd', 'I1', 'Q1']
        for key in keys:
            self.assertTrue(key in out,
                            msg=str('{} not in output'.format(key)))
            self.assertTrue(isinstance(out['fd'], float),
                            msg='Expect float')

    def test_fod(self):
        start = 0.0
        finish = 1.0
        NRS = 10
        # Test alpha = 0.0
        alpha = 0.0
        out = cap(f=fexp, alpha=alpha, start=start,
                      finish=finish, NRS=NRS)
        self.check_contents(out)
        self.assertTrue(isinstance(out['fd'], float),
                        msg='Expect float return')
        out = cap(f=fsp, alpha=alpha, start=start,
                      finish=finish, quadrature='glag')
        self.check_contents(out)
        self.assertTrue(isinstance(out['fd'], float),
                        msg='Expect float return')


# --------------------------
class GrunwaldLetnikov(unittest.TestCase):

    def check_contents(self, out):
        keys = ['fd']
        for key in keys:
            self.assertTrue(key in out,
                            msg=str('{} not in output'.format(key)))
            self.assertTrue(isinstance(out['fd'], float),
                            msg='Expect float')

    def test_fod(self):
        start = 0.0
        finish = 1.0
        NRS = 10
        # Test alpha = 0.0
        alpha = 0.0
        out = glet(f=fexp, alpha=alpha, start=start,
                      finish=finish, NRS=NRS)
        self.check_contents(out)
        self.assertTrue(isinstance(out['fd'], float),
                        msg='Expect float return')
        out = glet(f=fsp, alpha=alpha, start=start,
                      finish=finish, quadrature='glag')
        self.check_contents(out)
        self.assertTrue(isinstance(out['fd'], float),
                        msg='Expect float return')
