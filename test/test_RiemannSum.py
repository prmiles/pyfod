# -*- coding: utf-8 -*-

import numpy as np
import unittest
from pyfod.RiemannSum import RiemannSum


def f(t):
    return np.exp(2*t)


# --------------------------
class RiemannSumTesting(unittest.TestCase):

    def test_init(self):
        RS = RiemannSum(N=10, start=1.0, finish=12.0)
        attributes = ['points', 'weights', 'f', 'alpha', 'grid', 'description']
        for att in attributes:
            self.assertTrue(hasattr(RS, att),
                            msg=str('Missing {} attribute'.format(att)))

    def test_update_weights(self):
        RS = RiemannSum(N=10, start=1.0, finish=12.0)
        RS.weights = []
        RS.update_weights(alpha=0.0)
        self.assertTrue(isinstance(RS.weights, np.ndarray),
                        msg='Weights should be updated.')

    def test_integrate(self):
        RS = RiemannSum(N=10, start=0.0, finish=1.0)
        a = RS.integrate(f=f)
        self.assertTrue(isinstance(a, float), msg='Expect float')

    def test_grid(self):
        RS = RiemannSum()
        grid = RS._rs_grid(start=1.0, finish=12.0, N=50)
        self.assertTrue(np.allclose(grid, np.linspace(1.0, 12.0, num=50)),
                        msg=str('Expect arrays equal: {} neq {}'.format(grid,
                                 np.linspace(1.0, 12.0, num=50))))
        self.assertFalse(np.allclose(grid, np.linspace(2.0, 12.0, num=50)),
                        msg=str('Expect arrays not equal: {} eq {}'.format(grid,
                                 np.linspace(2.0, 12.0, num=50))))

    def test_rs_points(self):
        RS = RiemannSum(N=10, start=1.0, finish=12.0)
        self.assertTrue(isinstance(RS.points, np.ndarray), msg='Output numpy array')
        grid = RS._rs_grid(N=10, start=1.0, finish=12.0)
        jj = grid.size - 1
        values = (grid[1:jj+1] + grid[0:jj])/2
        self.assertTrue(np.allclose(RS.points, values), msg=str('Expect arrays equal: {} neq {}'.format(RS.points, values)))

    def different_alphas(self, alpha):
        RS = RiemannSum(alpha=alpha, N=10, start=1.0, finish=12.0)
        self.assertTrue(isinstance(RS.weights, np.ndarray), msg='Output numpy array')
        grid = RS._rs_grid(N=10, start=1.0, finish=12.0)
        jj = grid.size - 1
        term2 = (grid[jj] - grid[1:jj+1])**(1-alpha)
        term3 = (grid[jj] - grid[0:jj])**(1-alpha)
        values = -1/(1-alpha)*(term2 - term3)
        self.assertTrue(np.allclose(RS.weights, values), msg=str('Expect arrays equal: {} neq {}'.format(RS.weights, values)))
    
    def test_rs_weights(self):
        self.different_alphas(alpha=0.0)
        self.different_alphas(alpha=0.5)
        self.different_alphas(alpha=0.99)
