#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:12:41 2019

@author: prmiles
"""

import numpy as np
import sys
from sympy.integrals.quadrature import gauss_gen_laguerre as sp_gauss_laguerre
import sympy as sp


class GaussLaguerre:

    def __init__(self, N=5, start=0.0, finish=1.0, alpha=0.0,
                 f=None, extend_precision=True, n_digits=30):
        self.description = 'Gaussian-Laguerre Quadrature'
        if extend_precision is False:
            points, weights = np.polynomial.laguerre.laggauss(deg=N)
            self.points = 1 - np.exp(-points)
        else:
            points, weights = sp_gauss_laguerre(
                    n=N, n_digits=n_digits, alpha=0)
            points = [-p for p in points]
            points = sp.Array(points)
            self.points = sp.Array(
                    np.ones(shape=len(points))) - points.applyfunc(sp.exp)
            weights = sp.Array(weights)
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
        # check if sympy
        if isinstance(self.points, sp.Array):
            evalpoints = self.points.applyfunc(
                    lambda x: span*x + self.start)
            feval = evalpoints.applyfunc(f)
            coef = self.points.applyfunc(
                    lambda x: span**(1-alpha)*(1-x)**(-alpha))
            s = 0
            for ii, (w, f, t) in enumerate(zip(self.weights, feval, coef)):
                s += w*f*t
            return s
        else:
            coef = span**(1-alpha)*(1-self.points)**(-alpha)
            s = (self.weights*(coef*f(span*self.points + self.start))).sum()
            return s


if __name__ == '__main__':  # pragma: no cover

    n_digits = 50
    alpha = 0.9
    start = 0.0
    finish = 1.0
    dt = 1e-6
    N = 24

    def fcosnp(x):
        return np.cos(2*x) + 3

    GLag = GaussLaguerre(N=N, start=start, finish=finish,
                         f=fcosnp, alpha=0, extend_precision=False)
    F1 = GLag.integrate()
    print('Int(cos(2t)+3) = {} ({})'.format(F1, 3.4546))
    
    def fexp(x):
        return sp.exp(2*x)
    GLag = GaussLaguerre(N=N, start=start, finish=finish,
                         f=fexp, alpha=alpha, n_digits=n_digits)
    F1 = GLag.integrate()
    GLag = GaussLaguerre(N=N, start=start, finish=finish-dt,
                         f=fexp, alpha=alpha, n_digits=n_digits)
    F2 = GLag.integrate()
    print('D[f(t)=exp(2t)] = {} ({})'.format(
            (F1-F2)/(dt*sp.gamma(1-0.9)), 13.815))

    def fcos(x):
        return sp.cos(2*x)

    GLag = GaussLaguerre(N=N, start=start, finish=finish,
                         f=fcos, alpha=alpha, n_digits=n_digits)
    F1 = GLag.integrate()
    GLag = GaussLaguerre(N=N, start=start, finish=finish-dt,
                         f=fcos, alpha=alpha, n_digits=n_digits)
    F2 = GLag.integrate()
    print('D[f(t)=cos(2t)] = {} ({})'.format(
            (F1-F2)/(dt*sp.gamma(1-0.9)), -1.779))
