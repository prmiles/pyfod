#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:14:07 2018

@author: prmiles
"""


# ---------------
def riemann_sum_RL_FD(f, gridpoints, alpha):
    n = len(gridpoints)
    jj = n - 1
    g1 = 0.0
    for kk in range(jj):
        term1 = f((gridpoints[kk + 1] + gridpoints[kk])/2)
        term2 = (gridpoints[jj] - gridpoints[kk + 1])**(1-alpha)
        term3 = (gridpoints[jj] - gridpoints[kk])**(1-alpha)
        g1 = g1 + term1*(term2 - term3)
    return -1/(1-alpha)*g1
