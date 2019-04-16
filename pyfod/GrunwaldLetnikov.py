# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp
from pyfod.utilities import check_node_type
from pyfod.utilities import check_value


class GrunwaldLetnikov:

    def __init__(self, N=5, start=0.0, finish=1.0, alpha=0.0, f=None):
        self.description = 'Grunwald-Letnikov Fractional Derivative'
        N = check_node_type(N)
        self.start = start
        self.finish = finish
        self.alpha = alpha
        self.N = N
        self.f = f

    def evaluate(self, x, N=None, alpha=None, f=None):
        # Check user input
        self._check_user_settings(N=N, alpha=alpha, f=f)
        # Evaluate fractional derivative
        fd = 0.0
        self.h = (self.finish - self.start)/(self.N)
        for m in range(0, self.N):
            tmp = (-1.0)**m * sp.binomial(self.alpha, m) * (
                    self.f(x + (self.alpha - m)*self.h))
            fd += tmp
#            print('{} of {}: {}'.format(m + 1, self.N,
#                  fd/(self.h**self.alpha)))
        return fd/(self.h**self.alpha)

    def _check_user_settings(self, N, alpha, f):
        self.f = check_value(f, self.f, varname='f')
        self.alpha = check_value(alpha, self.alpha, varname='alpha')
        self.N = check_value(N, self.N, varname='N')
        self.N = check_node_type(self.N)


if __name__ == '__main__':  # pragma: no cover

    def fspexp(t):
        return np.exp(2*t)

    m = 20
    start = 0.0
    finish = 1.0

    # Test Grunwald-Letnikov
    print('Testing quadrature method: GLet')
#    # Test alpha = 0.0
#    alpha = 0.0
#    GL = GrunwaldLetnikov(N=m, f=fspexp, alpha=alpha,
#                          start=start, finish=finish)
#    out = GL.evaluate(x=finish)
#    print('D^{}[exp(2t)] = {} ({})'.format(alpha, out, 7.38906))
#    # Test alpha = 0.1
#    alpha = 0.1
#    GL = GrunwaldLetnikov(N=m, f=fspexp, alpha=alpha,
#                          start=start, finish=finish)
#    out = GL.evaluate(x=finish)
#    print('D^{}[exp(2t)] = {} ({})'.format(alpha, out, 7.95224))
    # Test alpha = 0.9
    alpha = 0.9
    GL = GrunwaldLetnikov(N=m, f=fspexp, alpha=alpha,
                          start=start, finish=finish)
    out = GL.evaluate(x=finish)
    print('D^{}[exp(2t)] = {} ({})'.format(alpha, out, 13.8153))
