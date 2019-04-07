#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:00:01 2019

@author: prmiles
"""
import sys
import numpy as np
from .utilities import check_alpha


class RiemannSum(object):

    def __init__(self, N=5, start=0.0, finish=1.0, alpha=0.0, f=None):
        self.description = 'Riemann-Sum'
        check_alpha(alpha=alpha)
        self.grid = self.rs_grid(start, finish, N)
        self.points = self.rs_points(grid=self.grid)
        self.weights = self.rs_weights(grid=self.grid, alpha=alpha)

    def integrate(self, f=None):
        if f is None:
            f = self.f
        if f is None:
            sys.exit('No function defined... provide function f')
        return (self.weights*f(self.points)).sum()

    @classmethod
    def rs_grid(cls, start, finish, N):
        return np.linspace(start=start, stop=finish, num=N)

    @classmethod
    def rs_points(cls, grid):
        jj = grid.size - 1
        return (grid[1:jj+1] + grid[0:jj])/2

    @classmethod
    def rs_weights(cls, grid, alpha=0.0):
        jj = grid.size - 1
        term2 = (grid[jj] - grid[1:jj+1])**(1-alpha)
        term3 = (grid[jj] - grid[0:jj])**(1-alpha)
        return -1/(1-alpha)*(term2 - term3)

    def update_weights(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        self.alpha = alpha
        self.weights = self.rs_weights(grid=self.grid, alpha=alpha)


if __name__ == '__main__':  # pragma: no cover

    start = 0.0
    finish = 1.0
    N = 2400

    def fexp(x):
        return np.exp(2*x)

    def fcos(x):
        return np.cos(2*x)

    # Test alpha = 0.0
    alpha = 0.0
    RS = RiemannSum(N=N, start=start, finish=finish)
    F1 = RS.integrate(f=fexp)
    F2 = RS.integrate(f=fcos)
    print('Int(exp(2t)/(1-t)^{}) = {} ({})'.format(alpha, F1, 3.19453))
    print('Int(cos(2t)/(1-t)^{}) = {} ({})'.format(alpha, F2, 0.454649))
    # Test alpha = 0.1
    alpha = 0.1
    RS.update_weights(alpha=alpha)
    F1 = RS.integrate(f=fexp)
    F2 = RS.integrate(f=fcos)
    print('Int(exp(2t)/(1-t)^{}) = {} ({})'.format(alpha, F1, 3.749))
    print('Int(cos(2t)/(1-t)^{}) = {} ({})'.format(alpha, F2, 0.457653))
    # Test alpha = 0.9
    alpha = 0.9
    RS.update_weights(alpha=alpha)
    F1 = RS.integrate(f=fexp)
    F2 = RS.integrate(f=fcos)
    print('Int(exp(2t)/(1-t)^{}) = {} ({})'.format(alpha, F1, 65.2162))
    print('Int(cos(2t)/(1-t)^{}) = {} ({})'.format(alpha, F2, -2.52045))
