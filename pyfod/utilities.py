#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:14:07 2018

@author: prmiles
"""


def check_alpha(alpha):
    try:
        1/(1-alpha)
    except ZeroDivisionError as err:
        print('Invalid value! The value of alpha cannot \
              be 1.0. alpha = {}:\n\t{}'.format(alpha, err))
        raise SystemExit
