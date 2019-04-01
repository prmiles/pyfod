#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:12:41 2019

@author: prmiles
"""

import numpy as np


class GaussLegendre:

    def __init__(self, N=5, start=0.0, finish=1.0):
        self.description = 'Gaussian-Legendre Quadrature'
        h = (finish - start)/N
        self.points = self.gauss_points(N=N, h=h, start=start)
        self.weights = self.gauss_weights(N=N, h=h)

    def integrate(self, f):
        return (self.weights*f(self.points)).sum()

    @classmethod
    def base_gauss_points(cls):

        # base points
        gpts = np.zeros([4])
        gpts[0] = (1/2) - np.sqrt(15+2*np.sqrt(30))/(2*np.sqrt(35))
        gpts[1] = (1/2) - np.sqrt(15-2*np.sqrt(30))/(2*np.sqrt(35))
        gpts[2] = (1/2) + np.sqrt(15-2*np.sqrt(30))/(2*np.sqrt(35))
        gpts[3] = (1/2) + np.sqrt(15+2*np.sqrt(30))/(2*np.sqrt(35))
        return gpts

    @classmethod
    def base_gauss_weights(cls, h):
        # define the Gauss weights for a four point quadrature rule
        w = np.zeros([4])
        w[0] = 49*h/(12*(18 + np.sqrt(30)))
        w[1] = 49*h/(12*(18 - np.sqrt(30)))
        w[2] = w[1]
        w[3] = w[0]
        return w

    @classmethod
    def interval_gauss_points(cls, base_gpts, N, h, start):
        # determines the Gauss points for all N intervals.
        Gpts = np.zeros([4*N])
        for gct in range(N):
            for ell in range(4):
                Gpts[(gct)*4 + ell] = ((gct)*h
                                       + base_gpts[ell]*h + start)
        return Gpts

    def gauss_points(self, N, h, start):
        # base points
        gpts = self.base_gauss_points()
        # determines the Gauss points for all N intervals.
        Gpts = self.interval_gauss_points(gpts, N, h, start)
        return Gpts

    def gauss_weights(self, N, h):
        # determine the Gauss weights for a four point quadrature rule
        w = self.base_gauss_weights(h)
        # copy the weights to form a vector for all N intervals
        weights = w.copy()
        for gct in range(N-1):
            weights = np.concatenate((weights, w))
        return weights


if __name__ == '__main__':  # pragma: no cover

    GQ = GaussLegendre(N=10, start=1.0, finish=12.0)

    def f(t):
        return np.cos(2*t) + 3

    a = GQ.integrate(f)
    print('a = {}'.format(a))
