#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:14:07 2018

@author: prmiles
"""

import sys


def check_alpha(alpha):
    try:
        1/(1-alpha)
    except ZeroDivisionError as err:
        print('Invalid value! The value of alpha cannot \
              be 1.0. alpha = {}:\n\t{}'.format(alpha, err))
        raise SystemExit


def check_function(f, default_f):
    if f is None:
        f = default_f
    if f is None:
        sys.exit('No function defined... provide function f')
    return f


def check_singularity(singularity, finish):
    if singularity is None:
        return finish
    else:
        return singularity


def check_node_type(N):
    return int(N)
