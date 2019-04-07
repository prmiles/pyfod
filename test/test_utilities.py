#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:14:07 2018

@author: prmiles
"""

import unittest
from pyfod import utilities

class Utilities(unittest.TestCase):
    
    def test_check_alpha(self):
            with self.assertRaises(SystemExit):
                utilities.check_alpha(alpha=1.0)
