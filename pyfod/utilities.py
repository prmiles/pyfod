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
        sys.exit('No function defined. Provide function f')
    return f


def check_value(value, default_value, varname=None):
    if value is None:
        value = default_value
    if value is None:
        sys.exit(str('No value defined. Provide \
                     value for {}'.format(varname)))
    return value


def check_input(value, varname=None):
    if value is None:
        sys.exit(str('No value defined. Provide value for {}.'.format(varname)))


def check_singularity(singularity, finish):
    if singularity is None:
        return finish
    else:
        return singularity


def check_node_type(N):
    return int(N)
