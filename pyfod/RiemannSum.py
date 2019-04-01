#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:00:01 2019

@author: prmiles
"""

import numpy as np


class RiemannSum(object):

    def __init__(self, alpha=0.0, N=5, start=0.0, finish=1.0):
        self.description = 'Riemann-Sum'
        self.check_alpha(alpha=alpha)
        grid = self.rs_grid(start, finish, N)
        self.points = self.rs_points(grid)
        self.weights = self.rs_weights(grid, alpha=alpha)

    def rs_grid(cls, start, finish, N):
        return np.linspace(start=start, stop=finish, num=N)

    def rs_points(cls, grid):
        jj = grid.size - 1
        return (grid[1:jj+1] + grid[0:jj])/2

    def rs_weights(cls, grid, alpha=0.0):
        jj = grid.size - 1
        term2 = (grid[jj] - grid[1:jj+1])**(1-alpha)
        term3 = (grid[jj] - grid[0:jj])**(1-alpha)
        return -1/(1-alpha)*(term2 - term3)

    def check_alpha(cls, alpha):
        try:
            1/(1-alpha)
        except ZeroDivisionError as err:
            print('Invalid value! The value of alpha cannot \
                  be 1.0. alpha = {}:\n\t{}'.format(alpha, err))
            raise SystemExit


if __name__ == '__main__':  # pragma: no cover

    RS = RiemannSum(alpha=0.0, N=100, start=1.0, finish=12.0)

    def f(t):
        return np.cos(2*t) + 3

    a = (RS.weights*f(RS.points)).sum()
    print('a = {}'.format(a))
