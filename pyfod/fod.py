# -*- coding: utf-8 -*-
'''
This module provides support for three common definitions of fractional
derivative in the limiting case of :math:`\\alpha \\in [0,1)`.  The
definitions available include:

    * Riemann-Liouville - :func:`riemannliouville`
    * Caputo - :func:`caputo`
    * Grünwald-Letnikov - :func:`grunwaldletnikov`

For more details regarding this definitions of fractional derivatives
please see :cite:`podlubny1998fractional`.

.. note::
    In each method you are required to provide a function handle.  Depending
    on the method being used, you may need to define your function using
    sympy_ to allow for extended numerical precision.

.. _sympy: https://www.sympy.org/en/index.html

'''
import sys
import numpy as np
import sympy as sp
from scipy.special import gamma as sc_gamma
from pyfod import quadrature as qm
from pyfod.utilities import check_input as _check_input


def riemannliouville(f, lower, upper, dt=1e-4,
                     alpha=0.0, quadrature='GLegRS', **kwargs):
    '''
    Riemann-Liouville fractional derivative calculator for
    :math:`\\alpha \\in [0,1)`.

    The general definition for Riemann-Liouville fractional derivative
    is

    .. math::

        D_{RL}^\\alpha[f(t)] = \\frac{1}{\\Gamma(n-\\alpha)}
        \\frac{d^n}{dt^n}\\int_0^t\\frac{f(s)}{(t-s)^{\\alpha+1-n}}ds,

    where :math:`n = \\lceil\\alpha\\rceil`. In the limiting case where
    :math:`\\alpha \\in [0, 1)` this further simplifies to

    .. math::

        D_{RL}^\\alpha[f(t)] = \\frac{1}{\\Gamma(1-\\alpha)}
        \\frac{d}{dt}\\int_0^t\\frac{f(s)}{(t-s)^{\\alpha}}ds.

    By defining

    .. math::
        F[t] = \\int_{t_0}^t(t-s)^{-\\alpha}f(s)ds,

    we can numerically approximate this definition of fractional
    derivative as

    .. math::

        D_{RL}^\\alpha[f(t)] = \\chi\\frac{d}{dt}F[t] \\approx
        \\chi\\frac{F(t_{j+1}) - F(t_{j})}{t_{j+1}-t_{j}},

    where :math:`\\chi = \\Gamma(1-\\alpha)^{-1}`.  For more details regarding
    this approach please see :cite:`atangana2017numerical`
    and :cite:`miles2018numerical`.

    Args:
        * **f** (def): Function handle.
        * **lower** (:py:class:`float`): Lower limit - should be zero.
        * **upper** (:py:class:`float`): Upper limit, i.e., point at which
          fractional derivative is being evaluated.

    Kwargs: name (type) - default
        * **dt** (:py:class:`float`) - `1e-4`: Time step, :math:`t_{j+1}-t_j`.
        * **alpha** (:py:class:`float`) - `0`: Order of fractional derivative.
        * **quadrature** (:py:class:`str`) - `'glegrs'`: Quadrature method
        * **kwargs**: Quadrature specific settings.

    Returns: :py:class:`dict`
        * `fd`: Fractional derivative
        * `i1`: Value of integral :math:`F(t_{j+1})`.
        * `i2`: Value of integral :math:`F(t_j)`.
        * `q1`: Quadrature object for :math:`F(t_{j+1})`.
        * `q2`: Quadrature object for :math:`F(t_{j})`.
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
    Caputo fractional derivative calculator for
    :math:`\\alpha \\in [0,1)`.

    The general definition for Caputo fractional derivative
    is

    .. math::

        D_{C}^\\alpha[f(t)] = \\frac{1}{\\Gamma(n-\\alpha)}
        \\int_0^t\\frac{f(s)^{(n)}}{(t-s)^{\\alpha+1-n}}ds,

    where :math:`n = \\lceil\\alpha\\rceil`. In the limiting case where
    :math:`\\alpha \\in [0, 1)` this further simplifies to

    .. math::

        D_{C}^\\alpha[f(t)] = \\frac{1}{\\Gamma(1-\\alpha)}
        \\int_0^t\\frac{f(s)^{(1)}}{(t-s)^{\\alpha}}ds.

    To evaluate this we simply need to define a finite-difference scheme
    for approximating :math:`f(s)^{(1)}`.

    Args:
        * **f** (def): Function handle.
        * **lower** (:py:class:`float`): Lower limit - should be zero.
        * **upper** (:py:class:`float`): Upper limit, i.e., point at which
          fractional derivative is being evaluated.

    Kwargs: name (type) - default
        * **dt** (:py:class:`float`) - `1e-4`: Time step, :math:`t_{j+1}-t_j`.
        * **alpha** (:py:class:`float`) - `0`: Order of fractional derivative.
        * **df** (def) - `None`: Finite difference function.  See tutorials
          for examples of how to utilize this feature.
        * **quadrature** (:py:class:`str`) - `'glegrs'`: Quadrature method
        * **kwargs**: Quadrature specific settings.

    Returns: :py:class:`dict`
        * `fd`: Fractional derivative.
        * `i1`: Value of integral.
        * `q1`: Quadrature object.
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
    Grünwald-Letnikov fractional derivative calculator.

    We have implemented the reverse Grünwald-Letnikov definition.

    .. math::

        D_G^\\alpha [f(t)]=\\lim_{h\\rightarrow 0}\\frac{1}{h^\\alpha}
        \\sum_{0\\leq m< \\infty}(-1)^m\\binom{\\alpha}{m}f(t-mh).

    .. note::
        The package as a whole was built for problems where
        :math:`\\alpha \\in [0, 1)`; however, this definition for
        Grünwald-Letnikov does not necessarily have the same constraints.
        It has not been tested for values of :math:`\\alpha \\ge 1`,
        but in principle it will work.

    Args:
        * **f** (def): Function handle.
        * **lower** (:py:class:`float`): Lower limit - should be zero.
        * **upper** (:py:class:`float`): Upper limit, i.e., point at which
          fractional derivative is being evaluated.

    Kwargs: name (type) - default
        * **n** (:py:class:`int`) - `100`: Number of terms to be used in
          approximation.
        * **dt** (:py:class:`float`) - `1e-4`: Time step. If `dt is None`,
          then the value of `dt` will be `(upper - lower)/n`.
        * **alpha** (:py:class:`float`) - `0`: Order of fractional derivative.

    Returns: :py:class:`dict`
        * `fd`: Fractional derivative.
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
