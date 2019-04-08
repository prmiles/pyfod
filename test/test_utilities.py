#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:14:07 2018

@author: prmiles
"""

import unittest
from pyfod import utilities as ut


def f(t):
    return 'function'


class Utilities(unittest.TestCase):

    def test_check_alpha(self):
        with self.assertRaises(SystemExit):
            ut.check_alpha(alpha=1.0)
        self.assertEqual(ut.check_alpha(alpha=0.5), None, msg='Expect None')

    def test_check_f(self):
        with self.assertRaises(SystemExit):
            ut.check_function(None, None)
        a = ut.check_function(f=f, default_f=None)
        self.assertEqual(a, f, msg='Expect user defined return')
        a = ut.check_function(f=None, default_f=f)
        self.assertEqual(a, f, msg='Expect default return')

    def test_check_singularity(self):
        a = ut.check_singularity(singularity=None, finish=1.0)
        self.assertEqual(a, 1.0, msg='Expect finish')
        a = ut.check_singularity(singularity=0.5, finish=1.0)
        self.assertEqual(a, 0.5, msg='Expect user defined')
