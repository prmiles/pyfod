# -*- coding: utf-8 -*-
import unittest
from pyfod import fdc
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
            fdc.select_quadrature_method(quadrature='hello')


# --------------------------
class FDC(unittest.TestCase):

    def check_contents(self, out):
        keys = ['fd', 'I1', 'I2', 'Q1', 'Q2']
        for key in keys:
            self.assertTrue(key in out,
                            msg=str('{} not in output'.format(key)))
            self.assertTrue(isinstance(out['fd'], float),
                            msg='Expect float')

    def test_fdc(self):
        start = 0.0
        finish = 1.0
        NRS = 10
        # Test alpha = 0.0
        alpha = 0.0
        out = fdc.fdc(f=fexp, alpha=alpha, start=start,
                      finish=finish, NRS=NRS)
        self.check_contents(out)
        out = fdc.fdc(f=fsp, alpha=alpha, start=start,
                      finish=finish, quadrature='glag')
        self.check_contents(out)
