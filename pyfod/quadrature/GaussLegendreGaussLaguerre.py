#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 08:45:04 2019

@author: prmiles
"""


from pyfod.GaussLegendre import GaussLegendre
from pyfod.GaussLaguerre import GaussLaguerre
from pyfod.utilities import check_function


class GaussLegendreGaussLaguerre(object):

    def __init__(self, NGLeg=5, NGLag=20, pGLeg=0.9,
                 start=0.0, finish=1.0, alpha=0.0, f=None,
                 extend_precision=True, n_digits=30):
        self.description = 'Hybrid: Gauss-Legendre, Gauss-Laguerre'
        # setup GLeg points/weights
        switch_time = (finish - start)*pGLeg
        self.GLeg = GaussLegendre(N=NGLeg, start=start, finish=switch_time,
                                  alpha=alpha, singularity=finish, f=f)
        # setup GLag points/weights
        self.GLag = GaussLaguerre(N=NGLag, start=switch_time,
                                  finish=finish, alpha=alpha, f=f,
                                  extend_precision=extend_precision,
                                  n_digits=n_digits)
        self.alpha = alpha
        self.pGLeg = pGLeg
        self.f = f

    def integrate(self, f=None):
        f = check_function(f, self.f)
        self.f = f
        return self.GLeg.integrate(f=f) + self.GLag.integrate(f=f)

    def update_weights(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        self.alpha = alpha
        self.GLeg.update_weights(alpha=alpha)
        self.GLag.update_weights(alpha=alpha)
