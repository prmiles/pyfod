#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:51:32 2019

@author: prmiles
"""

from pyfod.GaussLegendre import GaussLegendre
from pyfod.RiemannSum import RiemannSum
import numpy as np


class GaussRiemannSum(object):

    def __init__(self, NGQ=5, NRS=20, pGQ=0.9, start=0.0, finish=1.0):
        self.description = 'Gaussian Quadrature, Riemann-Sum'
        # setup GQ points/weights
        switch_time = (finish - start)*pGQ
        self.GQ = GaussLegendre(N=NGQ, start=start, finish=switch_time)
        # setup RS points/weights
        self.RS = RiemannSum(N=NRS, start=switch_time, finish=finish)


if __name__ == '__main__':  # pragma: no cover

    GRS = GaussRiemannSum(NGQ=10, start=1.0, finish=12.0)

    def f(t):
        return np.cos(2*t) + 3

    a = ((GRS.GQ.weights*f(GRS.GQ.points)).sum()
         + (GRS.RS.weights*f(GRS.RS.points)).sum())
    print('a = {}'.format(a))
