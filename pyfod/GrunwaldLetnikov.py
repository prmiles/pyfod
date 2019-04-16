# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp
from pyfod.utilities import check_node_type
from pyfod.utilities import check_input


def grunwaldletnikov(x, f=None, N=5, start=0.0, finish=1.0, alpha=0.0):
        # Check user input
        _check_user_settings(N=N, alpha=alpha, f=f)
        N = check_node_type(N)
        # Evaluate fractional derivative
        fd = 0.0
        h = (finish - start)/(N)
        for m in range(0, N):
            tmp = (-1.0)**m * sp.binomial(alpha, m) * (
                    f(x + (alpha - m)*h))
            fd += tmp
        return fd/(h**alpha)

def _check_user_settings(N, alpha, f):
    check_input(f, varname='f')
    check_input(alpha, varname='alpha')
    check_input(N, varname='N')


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
    out = grunwaldletnikov(x=finish, N=m, alpha=alpha,
                          start=start, finish=finish)
    print('D^{}[exp(2t)] = {} ({})'.format(alpha, out, 13.8153))
