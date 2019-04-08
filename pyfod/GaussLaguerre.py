#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:12:41 2019

@author: prmiles
"""

import numpy as np
from sympy.integrals.quadrature import gauss_gen_laguerre as sp_gauss_laguerre
import sympy as sp
from pyfod.utilities import check_function


class GaussLaguerre:

    def __init__(self, N=5, start=0.0, finish=1.0, alpha=0.0,
                 f=None, extend_precision=True, n_digits=30):
        self.description = 'Gaussian-Laguerre Quadrature'
        self.start = start
        self.finish = finish
        self.alpha = alpha
        self.f = f
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
        self.initial_weights = weights.copy()
        self.update_weights(alpha=alpha)

    def integrate(self, f=None):
        f = check_function(f, self.f)
        self.f = f
        # transform kernel
        span = self.finish - self.start
        # check if sympy
        if isinstance(self.points, sp.Array):
            evalpoints = self.points.applyfunc(
                    lambda x: span*x + self.start)
            feval = evalpoints.applyfunc(f)
            s = 0
            for ii, (w, f) in enumerate(zip(self.weights, feval)):
                s += w*f
            return s
        else:
            s = (self.weights*(f(span*self.points + self.start))).sum()
            return s

    def update_weights(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        self.alpha = alpha
        span = self.finish - self.start
        # check if sympy
        if isinstance(self.points, sp.Array):
            coef = self.points.applyfunc(
                    lambda x: span**(1-alpha)*(1-x)**(-alpha))
            wtmp = []
            for ii, (c, w) in enumerate(zip(coef, self.initial_weights)):
                wtmp.append(c*w)
            self.weights = sp.Array(wtmp)
        else:
            coef = span**(1-alpha)*(1-self.points)**(-alpha)
            self.weights = self.initial_weights*coef


if __name__ == '__main__':  # pragma: no cover

    n_digits = 50
    start = 0.0
    finish = 1.0
    N = 24
    '''
    Test normal precision
    '''
    # f(t) = exp(2t)

    def fexpnp(x):
        return np.exp(2*x)

    def fcosnp(x):
        return np.cos(2*x)

    # Test alpha = 0.0
    alpha = 0.0
    GLag = GaussLaguerre(N=N, start=start, finish=finish,
                         f=fexpnp, alpha=alpha, extend_precision=False)
    F1 = GLag.integrate()
    F2 = GLag.integrate(f=fcosnp)
    print('Int(exp(2t)/(1-t)^{}) = {} ({})'.format(alpha, F1, 3.19453))
    print('Int(cos(2t)/(1-t)^{}) = {} ({})'.format(alpha, F2, 0.454649))

    '''
    Test extended precision
    '''

    def fexp(x):
        return sp.exp(2*x)

    def fcos(x):
        return sp.cos(2*x)

    # Test alpha = 0.0
    alpha = 0.0
    GLag = GaussLaguerre(N=N, start=start, finish=finish,
                         alpha=alpha, n_digits=n_digits)
    F1 = GLag.integrate(f=fexp)
    F2 = GLag.integrate(f=fcos)
    print('Int(exp(2t)/(1-t)^{}) = {} ({})'.format(alpha, F1, 3.19453))
    print('Int(cos(2t)/(1-t)^{}) = {} ({})'.format(alpha, F2, 0.454649))
    # Test alpha = 0.1
    alpha = 0.1
    GLag.update_weights(alpha=alpha)
    F1 = GLag.integrate(f=fexp)
    F2 = GLag.integrate(f=fcos)
    print('Int(exp(2t)/(1-t)^{}) = {} ({})'.format(alpha, F1, 3.749))
    print('Int(cos(2t)/(1-t)^{}) = {} ({})'.format(alpha, F2, 0.457653))
    # Test alpha = 0.9
    alpha = 0.9
    GLag.update_weights(alpha=alpha)
    F1 = GLag.integrate(f=fexp)
    F2 = GLag.integrate(f=fcos)
    print('Int(exp(2t)/(1-t)^{}) = {} ({})'.format(alpha, F1, 65.2162))
    print('Int(cos(2t)/(1-t)^{}) = {} ({})'.format(alpha, F2, -2.52045))
