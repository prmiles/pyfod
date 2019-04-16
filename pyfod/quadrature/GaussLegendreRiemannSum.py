#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:51:32 2019

@author: prmiles
"""

from pyfod.GaussLegendre import GaussLegendre
from pyfod.RiemannSum import RiemannSum
from pyfod.utilities import check_function
import numpy as np


class GaussLegendreRiemannSum(object):

    def __init__(self, NGLeg=5, NRS=20, pGLeg=0.9,
                 start=0.0, finish=1.0, alpha=0.0, f=None):
        self.description = 'Gaussian Quadrature, Riemann-Sum'
        # setup GQ points/weights
        switch_time = (finish - start)*pGLeg
        self.GLeg = GaussLegendre(N=NGLeg, start=start, finish=switch_time,
                                  alpha=alpha, singularity=finish, f=f)
        # setup RS points/weights
        self.RS = RiemannSum(N=NRS, start=switch_time,
                             finish=finish, alpha=alpha, f=f)
        self.alpha = alpha
        self.pGLeg = pGLeg
        self.f = f

    def integrate(self, f=None):
        f = check_function(f, self.f)
        self.f = f
        return self.GLeg.integrate(f=f) + self.RS.integrate(f=f)

    def update_weights(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        self.alpha = alpha
        self.GLeg.update_weights(alpha=alpha)
        self.RS.update_weights(alpha=alpha)


if __name__ == '__main__':  # pragma: no cover

    start = 0.0
    finish = 1.0
    NGLeg = 100
    NRS = 2400
    pGLeg = 0.95

    def fexp(x):
        return np.exp(2*x)

    def fcos(x):
        return np.cos(2*x)

    # Test alpha = 0.0
    alpha = 0.0
    GRS = GaussLegendreRiemannSum(pGLeg=pGLeg, NGLeg=NGLeg, NRS=NRS,
                                  start=start, finish=finish)
    F1 = GRS.integrate(f=fexp)
    F2 = GRS.integrate(f=fcos)
    print('Int(exp(2t)/(1-t)^{}) = {} ({})'.format(alpha, F1, 3.19453))
    print('Int(cos(2t)/(1-t)^{}) = {} ({})'.format(alpha, F2, 0.454649))
    # Test alpha = 0.1
    alpha = 0.1
    GRS.update_weights(alpha=alpha)
    F1 = GRS.integrate(f=fexp)
    F2 = GRS.integrate(f=fcos)
    print('Int(exp(2t)/(1-t)^{}) = {} ({})'.format(alpha, F1, 3.749))
    print('Int(cos(2t)/(1-t)^{}) = {} ({})'.format(alpha, F2, 0.457653))
    # Test alpha = 0.9
    alpha = 0.9
    GRS.update_weights(alpha=alpha)
    F1 = GRS.integrate(f=fexp)
    F2 = GRS.integrate(f=fcos)
    print('Int(exp(2t)/(1-t)^{}) = {} ({})'.format(alpha, F1, 65.2162))
    print('Int(cos(2t)/(1-t)^{}) = {} ({})'.format(alpha, F2, -2.52045))
