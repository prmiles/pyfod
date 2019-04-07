# -*- coding: utf-8 -*-

import sys
import numpy as np
from scipy.special import gamma as sc_gamma
import sympy as sp
from GaussRiemannSum import GaussRiemannSum
from pyfod.GaussLegendre import GaussLegendre
from pyfod.GaussLaguerre import GaussLaguerre
from pyfod.RiemannSum import RiemannSum


def fdc(f, start, finish, dt=1e-4, alpha=0.0, quadrature='GLegRS', **kwargs):

    quad = select_quadrature_method(quadrature)
    Q1 = quad(start=start, finish=finish, alpha=alpha, **kwargs)
    I1 = Q1.integrate(f=f)
    Q2 = quad(start=start, finish=finish-dt, alpha=alpha, **kwargs)
    I2 = Q2.integrate(f=f)

    if quadrature == 'GLag':
        extend_precision = True
    else:
        extend_precision = False

    if extend_precision is True:
        fd = (I1-I2)/(dt*sp.gamma(1-alpha))
    else:
        fd = (I1-I2)/(dt*sc_gamma(1 - alpha))
    # assemble output
    return dict(fd=fd, I1=I1, I2=I2, Q1=Q1, Q2=Q2)


def select_quadrature_method(quadrature):
    methods = dict(
            GLegRS=GaussRiemannSum,
            GLag=GaussLaguerre,
            GLeg=GaussLegendre,
            RS=RiemannSum
            )
    try:
        quad = methods[quadrature]
        return quad
    except KeyError:
        print('Invalid quadrature method specified: {}'.format(quadrature))
        print('Please specify one of the following:')
        for method in methods:
            print('\t{}'.format(method))
        sys.exit('Invalid quadrature method')


if __name__ == '__main__':  # pragma: no cover

    def fcos(t):
        return np.cos(2*t)

    def fexp(t):
        return np.exp(2*t)

    f = fexp
    # Check Gauss-Legendre, Riemann-Sum
    dft = fdc(quadrature='GLegRS', f=f, start=0.0, finish=1.0,
              alpha=0.9, dt=1e-6, NGQ=10, NRS=20)
    print(dft['fd'])
    # Check Riemann-Sum
    dft = fdc(quadrature='RS', f=f, start=0.0, finish=1.0,
              alpha=0.9, dt=1e-4, N=32)
    print(dft['fd'])
    # Check Gauss-Legendre
    dft = fdc(quadrature='GLeg', f=f, start=0.0, finish=1.0,
              alpha=0.9, dt=1e-4, N=32)
    print(dft['fd'])
    # Check Gauss-Laguerre.  Redefine functions with extended precision.

    def gcos(t):
        return sp.cos(2*t)

    def gexp(t):
        return sp.exp(2*t)

    g = gexp
    dft = fdc(quadrature='GLag', f=g, start=0.0, finish=1.0,
              alpha=0.9, dt=1e-4, N=24, n_digits=50, extend_precision=True)
    print(dft['fd'])
