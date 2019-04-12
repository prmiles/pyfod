# -*- coding: utf-8 -*-

import sys
import numpy as np
import sympy as sp
from scipy.special import gamma as sc_gamma
from pyfod.GaussLegendreRiemannSum import GaussLegendreRiemannSum
from pyfod.GaussLegendre import GaussLegendre
from pyfod.GaussLaguerre import GaussLaguerre
from pyfod.RiemannSum import RiemannSum


def fdc(f, start, finish, dt=1e-4, alpha=0.0, quadrature='GLegRS', **kwargs):

    quad = select_quadrature_method(quadrature)
    Q1 = quad(start=start, finish=finish, alpha=alpha, **kwargs)
    I1 = Q1.integrate(f=f)
    Q2 = quad(start=start, finish=finish-dt, alpha=alpha, **kwargs)
    I2 = Q2.integrate(f=f)

    if quadrature.lower() == 'glag':
        extend_precision = True
    else:
        extend_precision = False

    if extend_precision is True:
        fd = float((I1-I2)/(dt*sp.gamma(1-alpha)))
    else:
        fd = (I1-I2)/(dt*sc_gamma(1 - alpha))
    # assemble output
    return dict(fd=fd, I1=I1, I2=I2, Q1=Q1, Q2=Q2)


def select_quadrature_method(quadrature):
    methods = dict(
            glegrs=GaussLegendreRiemannSum,
            glag=GaussLaguerre,
            gleg=GaussLegendre,
            rs=RiemannSum
            )
    try:
        quad = methods[quadrature.lower()]
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

    def fspexp(t):
        return sp.exp(2*t)

    start = 0.0
    finish = 1.0
    dt = 1e-6
    NRS = 1000
    NGLeg = 5
    print('Testing quadrature method: GLegRS')
    # Test alpha = 0.0
    alpha = 0.0
    out = fdc(f=fexp, alpha=alpha, start=start, finish=finish, dt=dt,
              NGLeg=NGLeg, NRS=NRS)
    print('D^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.38906))
    # Test alpha = 0.1
    alpha = 0.1
    out = fdc(f=fexp, alpha=alpha, start=start, finish=finish, dt=dt,
              NGLeg=NGLeg, NRS=NRS)
    print('D^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.95224))
    # Test alpha = 0.9
    alpha = 0.9
    out = fdc(f=fexp, alpha=alpha, start=start, finish=finish, dt=dt,
              NGLeg=NGLeg, NRS=NRS)
    print('D^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 13.8153))

    # Test - Riemann Quadrature
    print('Testing quadrature method: RS')
    # Test alpha = 0.0
    alpha = 0.0
    out = fdc(f=fexp, alpha=alpha, start=start, finish=finish, dt=dt,
              quadrature='rs', N=NRS)
    print('D^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.38906))
    # Test alpha = 0.1
    alpha = 0.1
    out = fdc(f=fexp, alpha=alpha, start=start, finish=finish, dt=dt,
              quadrature='rs', N=NRS)
    print('D^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.95224))
    # Test alpha = 0.9
    alpha = 0.9
    out = fdc(f=fexp, alpha=alpha, start=start, finish=finish, dt=dt,
              quadrature='rs', N=NRS)
    print('D^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 13.8153))

    # Test Extended Precision - Gauss Laguerre Quadrature
    print('Testing quadrature method: GLag')
    # Test alpha = 0.0
    alpha = 0.0
    out = fdc(f=fspexp, alpha=alpha, start=start, finish=finish, dt=dt,
              quadrature='glag')
    print('D^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.38906))
    # Test alpha = 0.1
    alpha = 0.1
    out = fdc(f=fspexp, alpha=alpha, start=start, finish=finish, dt=dt,
              quadrature='glag')
    print('D^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.95224))
    # Test alpha = 0.9
    alpha = 0.9
    out = fdc(f=fspexp, alpha=alpha, start=start, finish=finish, dt=dt,
              quadrature='glag', N=32, n_digits=60)
    print('D^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 13.8153))
