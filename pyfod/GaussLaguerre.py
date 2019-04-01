#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:12:41 2019

@author: prmiles
"""

import numpy as np
import sys


class GaussLaguerre:

    def __init__(self, N=5, start=0.0, finish=1.0, alpha=0.0, f=None):
        self.description = 'Gaussian-Laguerre Quadrature'
        points, weights = np.polynomial.laguerre.laggauss(deg=N)
        self.points = 1 - np.exp(-points)
        self.weights = weights
        self.start = start
        self.finish = finish
        self.alpha = alpha
        self.f = f

    def integrate(self, f=None, alpha=None):
        if alpha is None:
            alpha = self.alpha
        if f is None:
            f = self.f
        if f is None:
            sys.exit('No function defined... provide function f')
        # transform kernel
        span = self.finish - self.start
        return (self.weights*(span**(1-alpha)*f(span*self.points
                              + self.start)*(1-self.points)**(-alpha))).sum()


if __name__ == '__main__':  # pragma: no cover

    def f(t):
        return np.cos(2*t)

    GLag = GaussLaguerre(N=100, start=1.0, finish=12.0)

    a = GLag.integrate(f)
    print('a = {}'.format(a))

    GLag = GaussLaguerre(N=8, start=0.0, finish=1.0, f=f, alpha=0.9)
    F1 = GLag.integrate()
    dt = 1e-4
    GLag = GaussLaguerre(N=8, start=0.0, finish=1.0-dt, f=f, alpha=0.9)
    F2 = GLag.integrate()
    print('D[f(t)] = {}'.format((F1-F2)/dt))
