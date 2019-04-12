#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:12:41 2019

@author: prmiles
"""

import numpy as np
from pyfod.utilities import check_alpha
from pyfod.utilities import check_function, check_singularity
from pyfod.utilities import check_node_type


class GaussLegendre:

    def __init__(self, N=5, start=0.0, finish=1.0,
                 alpha=0.0, f=None, singularity=None):
        self.description = 'Gaussian-Legendre Quadrature'
        check_alpha(alpha)
        N = check_node_type(N)
        h = (finish - start)/N
        self.alpha = alpha
        self.f = f
        self.N = N
        self.finish = finish
        self.singularity = check_singularity(singularity, self.finish)
        self.points = self._gauss_points(N=N, h=h, start=start)
        self.weights = self._gauss_weights(N=N, h=h)
        self.initial_weights = self.weights.copy()
        self.update_weights(alpha=alpha)

    def update_weights(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        self.alpha = alpha
        # update weights based on alpha
        self.weights = self.initial_weights*(
                self.singularity - self.points)**(-alpha)

    def integrate(self, f=None):
        f = check_function(f, self.f)
        self.f = f
        return (self.weights*f(self.points)).sum()

    @classmethod
    def _base_gauss_points(cls):

        # base points
        gpts = np.zeros([4])
        gpts[0] = (1/2) - np.sqrt(15+2*np.sqrt(30))/(2*np.sqrt(35))
        gpts[1] = (1/2) - np.sqrt(15-2*np.sqrt(30))/(2*np.sqrt(35))
        gpts[2] = (1/2) + np.sqrt(15-2*np.sqrt(30))/(2*np.sqrt(35))
        gpts[3] = (1/2) + np.sqrt(15+2*np.sqrt(30))/(2*np.sqrt(35))
        return gpts

    @classmethod
    def _base_gauss_weights(cls, h):
        # define the Gauss weights for a four point quadrature rule
        w = np.zeros([4])
        w[0] = 49*h/(12*(18 + np.sqrt(30)))
        w[1] = 49*h/(12*(18 - np.sqrt(30)))
        w[2] = w[1]
        w[3] = w[0]
        return w

    @classmethod
    def _interval_gauss_points(cls, base_gpts, N, h, start):
        # determines the Gauss points for all N intervals.
        Gpts = np.zeros([4*N])
        for gct in range(N):
            for ell in range(4):
                Gpts[(gct)*4 + ell] = ((gct)*h
                                       + base_gpts[ell]*h + start)
        return Gpts

    def _gauss_points(self, N, h, start):
        # base points
        gpts = self._base_gauss_points()
        # determines the Gauss points for all N intervals.
        Gpts = self._interval_gauss_points(gpts, N, h, start)
        return Gpts

    def _gauss_weights(self, N, h):
        # determine the Gauss weights for a four point quadrature rule
        w = self._base_gauss_weights(h)
        # copy the weights to form a vector for all N intervals
        weights = w.copy()
        for gct in range(N-1):
            weights = np.concatenate((weights, w))
        return weights


if __name__ == '__main__':  # pragma: no cover

    start = 0.0
    finish = 1.0
    N = 2400
    GQ = GaussLegendre(N=N, start=start, finish=finish)

    def fexp(x):
        return np.exp(2*x)

    def fcos(x):
        return np.cos(2*x)

    # Test alpha = 0.0
    alpha = 0.0
    GLeg = GaussLegendre(N=N, start=start, finish=finish, alpha=alpha)
    F1 = GLeg.integrate(f=fexp)
    F2 = GLeg.integrate(f=fcos)
    print('Int(exp(2t)/(1-t)^{}) = {} ({})'.format(alpha, F1, 3.19453))
    print('Int(cos(2t)/(1-t)^{}) = {} ({})'.format(alpha, F2, 0.454649))
    # Test alpha = 0.1
    alpha = 0.1
    GLeg.update_weights(alpha=alpha)
    F1 = GLeg.integrate(f=fexp)
    F2 = GLeg.integrate(f=fcos)
    print('Int(exp(2t)/(1-t)^{}) = {} ({})'.format(alpha, F1, 3.749))
    print('Int(cos(2t)/(1-t)^{}) = {} ({})'.format(alpha, F2, 0.457653))
    # Test alpha = 0.9
    alpha = 0.9
    GLeg.update_weights(alpha=alpha)
    F1 = GLeg.integrate(f=fexp)
    F2 = GLeg.integrate(f=fcos)
    print('Int(exp(2t)/(1-t)^{}) = {} ({})'.format(alpha, F1, 65.2162))
    print('Int(cos(2t)/(1-t)^{}) = {} ({})'.format(alpha, F2, -2.52045))
