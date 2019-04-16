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

    def test_check_singularity(self):
        a = ut.check_singularity(singularity=None, finish=1.0)
        self.assertEqual(a, 1.0, msg='Expect finish')
        a = ut.check_singularity(singularity=0.5, finish=1.0)
        self.assertEqual(a, 0.5, msg='Expect user defined')

    def test_check_node_type(self):
        a = ut.check_node_type(N=1)
        self.assertEqual(a, 1, msg='Expect 1')
        self.assertTrue(isinstance(a, int), msg='Expect int')
        a = ut.check_node_type(N=1.0)
        self.assertTrue(isinstance(a, int), msg='Expect int')

    def test_check_values(self):
        with self.assertRaises(SystemExit):
            ut.check_value(None, None)
        a = ut.check_value(value=f, default_value=None)
        self.assertEqual(a, f, msg='Expect user defined return')
        a = ut.check_value(value=None, default_value=f)
        self.assertEqual(a, f, msg='Expect default return')
