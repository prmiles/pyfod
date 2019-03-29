#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:51:32 2019

@author: prmiles
"""

from GaussianQuadrature import GaussianQuadrature
from RiemannSum import RiemannSum
import numpy as np


class GaussianRiemannSum(GaussianQuadrature, RiemannSum):

    def __init__(self, NGQ=5, NRS=20, pGQ=0.9,
                 start=0.0, finish=1.0, startRS=None):
        self.description = 'Gaussian Quadrature, Riemann-Sum'
        # setup GQ points/weights
        switch_time = (finish - start)*pGQ
        hGQ = (switch_time - start)/NGQ
        self.gpts = self.gauss_points(N=NGQ, h=hGQ, start=start)
        self.gwts = self.gauss_weights(N=NGQ, h=hGQ)
        # setup RS points/weights
        grid = self.rs_grid(start=switch_time, finish=finish, N=NRS)
        self.rpts = self.rs_points(grid)
        self.rwts = self.rs_weights(grid, alpha=0.0)


if __name__ == '__main__':

    GRS = GaussianRiemannSum(NGQ=10, NRS=100,
                             pGQ=0.999, start=1.0, finish=12.0)

    def f(t):
        return np.cos(2*t) + 3

    a = (GRS.gwts*f(GRS.gpts)).sum() + (GRS.rwts*f(GRS.rpts)).sum()
    print('a = {}'.format(a))
