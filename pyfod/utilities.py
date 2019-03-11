#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:14:07 2018

@author: prmiles
"""

import numpy as np


# *********************************************
# ---------------
def base_gauss_points():

    # base points
    gpts = np.zeros([4])
    gpts[0] = (1/2) - np.sqrt(15+2*np.sqrt(30))/(2*np.sqrt(35))
    gpts[1] = (1/2) - np.sqrt(15-2*np.sqrt(30))/(2*np.sqrt(35))
    gpts[2] = (1/2) + np.sqrt(15-2*np.sqrt(30))/(2*np.sqrt(35))
    gpts[3] = (1/2) + np.sqrt(15+2*np.sqrt(30))/(2*np.sqrt(35))
    return gpts


# ---------------
def base_gauss_weights(h):

    # define the Gauss weights for a four point quadrature rule
    w = np.zeros([4, len(h)])
    w[0, :] = 49*h[:]/(12*(18 + np.sqrt(30)))
    w[1, :] = 49*h[:]/(12*(18 - np.sqrt(30)))
    w[2, :] = w[1, :]
    w[3, :] = w[0, :]
    return w


# ---------------
def interval_gauss_points(base_gpts, N, h, low_lim):
    # determines the Gauss points for all N intervals.
    Gpts = np.zeros([4*N, len(h)])
    for gct in range(N):
        for ell in range(4):
            Gpts[(gct)*4 + ell, :] = (gct)*h[:] + base_gpts[ell]*h[:] + low_lim
    return Gpts


# ---------------
def gauss_points(N, h, low_lim):
    # base points
    gpts = base_gauss_points()
    # determines the Gauss points for all N intervals.
    Gpts = interval_gauss_points(gpts, N, h, low_lim)
    return Gpts


# ---------------
def gauss_weights(N, h):
    # determine the Gauss weights for a four point quadrature rule
    w = base_gauss_weights(h)
    # copy the weights to form a vector for all N intervals
    weights = w.copy()
    for gct in range(N-1):
        weights = np.concatenate((weights, w))
    return weights


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
