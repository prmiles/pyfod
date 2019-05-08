# -*- coding: utf-8 -*-

import sys
import numpy as np
import sympy as sp
from scipy.special import gamma as sc_gamma
from pyfod import quadrature as qm
from pyfod.utilities import check_input as _check_input
from pyfod.utilities import check_value as _check_value


def riemannliouville(f, lower, upper, dt=1e-4,
                     alpha=0.0, quadrature='GLegRS', **kwargs):
    '''
    Riemann-Liouville Fractional Derivative Calculator
    '''
    quad = _select_quadrature_method(quadrature)
    q1 = quad(lower=lower, upper=upper, alpha=alpha, **kwargs)
    i1 = q1.integrate(f=f)
    q2 = quad(lower=lower, upper=upper-dt, alpha=alpha, **kwargs)
    i2 = q2.integrate(f=f)

    if quadrature.lower() == 'glag':
        extend_precision = True
    else:
        extend_precision = False

    if extend_precision is True:
        fd = float((i1-i2)/(dt*sp.gamma(1-alpha)))
    else:
        fd = (i1-i2)/(dt*sc_gamma(1 - alpha))
    # assemble output
    return dict(fd=fd, i1=i1, i2=i2, q1=q1, q2=q2)


def caputo(f, lower, upper, dt=1e-4, alpha=0.0,
           df=None, quadrature='GLegRS', **kwargs):
    '''
    Caputo Fractional Derivative Calculator
    '''
    # Check finite difference function
    df = _setup_finite_difference(df, f, dt)

    quad = _select_quadrature_method(quadrature)
    quadobj = quad(lower=lower, upper=upper, alpha=alpha, **kwargs)
    integral = quadobj.integrate(f=df)

    if quadrature.lower() == 'glag':
        extend_precision = True
    else:
        extend_precision = False

    if extend_precision is True:
        fd = float((integral)/(sp.gamma(1-alpha)))
    else:
        fd = (integral)/(sc_gamma(1 - alpha))
    # assemble output
    return dict(fd=fd, i1=integral, q1=quadobj)


def grunwaldletnikov(f, lower, upper, n=100, dt=None, alpha=0.0):
    '''
    Grunwald-Letnikov Fractional Derivative Calculator
    '''
    # Check user input
    _check_input(f, 'f')
    _check_input(lower, 'lower')
    _check_input(upper, 'upper')
    # Evaluate fractional derivative
    fd = 0.0
    if dt is not None:
        n = np.floor((upper - lower)/dt).astype(int)
    else:
        dt = (upper - lower)/n
    for m in range(0, n):
        tmp = (-1.0)**m * sp.binomial(alpha, m) * (
                f(upper - m*dt))
        fd += tmp
    fd = fd/(dt**alpha)
    # assemble output
    return dict(fd=fd)


def _setup_finite_difference(df, f, dt):
    '''
    Check if finite difference function is defined
    '''
    def default_df(t):
        return (f(t) - f(t-dt))/dt
    if df is None:
        return default_df
    else:
        return df


def _select_quadrature_method(quadrature):
    methods = dict(
            glegrs=qm.GaussLegendreRiemannSum,
            glegglag=qm.GaussLegendreGaussLaguerre,
            glag=qm.GaussLaguerre,
            gleg=qm.GaussLegendre,
            rs=qm.RiemannSum
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

    lower = 0.0
    upper = 1.0
    dt = 1e-4
    NRS = 1000
    NGLeg = 5
    print('Testing Riemann-Liouville Fractional Derivative:')
    print('\tQuadrature method: GLegRS')
    # Test alpha = 0.0
    alpha = 0.0
    rlou = riemannliouville
    out = rlou(f=fexp, alpha=alpha, lower=lower, upper=upper, dt=dt,
               ndom=NGLeg, nrs=NRS)
    print('\tQD^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.38906))
    # Test alpha = 0.1
    alpha = 0.1
    out = rlou(f=fexp, alpha=alpha, lower=lower, upper=upper, dt=dt,
               ndom=NGLeg, nrs=NRS)
    print('\tQD^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.95224))
    # Test alpha = 0.9
    alpha = 0.9
    out = rlou(f=fexp, alpha=alpha, lower=lower, upper=upper, dt=dt,
               ndom=NGLeg, nrs=NRS)
    print('\tQD^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 13.8153))

    # Test - Riemann Quadrature
    print('\tQuadrature method: RS')
    # Test alpha = 0.0
    alpha = 0.0
    out = rlou(f=fexp, alpha=alpha, lower=lower, upper=upper, dt=dt,
               quadrature='rs', n=NRS)
    print('\tD^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.38906))
    # Test alpha = 0.1
    alpha = 0.1
    out = rlou(f=fexp, alpha=alpha, lower=lower, upper=upper, dt=dt,
               quadrature='rs', n=NRS)
    print('\tD^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.95224))
    # Test alpha = 0.9
    alpha = 0.9
    out = rlou(f=fexp, alpha=alpha, lower=lower, upper=upper, dt=dt,
               quadrature='rs', n=NRS)
    print('\tD^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 13.8153))

    # Test Extended Precision - Gauss Laguerre Quadrature
    print('\tQuadrature method: GLag')
    # Test alpha = 0.0
    alpha = 0.0
    out = rlou(f=fspexp, alpha=alpha, lower=lower, upper=upper, dt=dt,
               quadrature='glag')
    print('\tD^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.38906))
    # Test alpha = 0.1
    alpha = 0.1
    out = rlou(f=fspexp, alpha=alpha, lower=lower, upper=upper, dt=dt,
               quadrature='glag')
    print('\tD^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.95224))
    # Test alpha = 0.9
    alpha = 0.9
    out = rlou(f=fspexp, alpha=alpha, lower=lower, upper=upper, dt=dt,
               quadrature='glag', deg=32, n_digits=60)
    print('\tD^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 13.8153))

    # Test Caputo Fractional Derivative
    print('Testing Caputo Fractional Derivative:')
    # Test alpha = 0.0
    alpha = 0.0
    out = caputo(f=fexp, alpha=alpha, lower=lower, upper=upper, dt=dt)
    print('\tD^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.38906))
    # Test alpha = 0.1
    alpha = 0.1
    out = caputo(f=fexp, alpha=alpha, lower=lower, upper=upper, dt=dt)
    print('\tD^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.95224))
    # Test alpha = 0.9
    alpha = 0.9
    out = caputo(f=fexp, alpha=alpha, lower=lower, upper=upper, dt=dt)
    print('\tD^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 13.8153))

    # Test Grunwald-Letnikov Fractional Derivative
    dt = 1e-2
    print('Testing Grunwald-Letnikov Fractional Derivative:')
    # Test alpha = 0.0
    alpha = 0.0
    out = grunwaldletnikov(f=fexp, alpha=alpha,
                           lower=lower, upper=upper, dt=dt)
    print('\tD^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.38906))
    # Test alpha = 0.1
    alpha = 0.1
    out = grunwaldletnikov(f=fexp, alpha=alpha,
                           lower=lower, upper=upper, dt=dt)
    print('\tD^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 7.95224))
    # Test alpha = 0.9
    alpha = 0.9
    out = grunwaldletnikov(f=fexp, alpha=alpha,
                           lower=lower, upper=upper, dt=dt)
    print('\tD^{}[exp(2t)] = {} ({})'.format(alpha, out['fd'], 13.8153))
