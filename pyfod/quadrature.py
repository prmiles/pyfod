import numpy as np
import sympy as sp
from sympy.integrals.quadrature import gauss_gen_laguerre as sp_gauss_laguerre
from pyfod.utilities import check_alpha
from pyfod.utilities import check_value
from pyfod.utilities import check_singularity
from pyfod.utilities import check_node_type
from pyfod.utilities import check_range


# ---------------------
class GaussLegendre:

    def __init__(self, ndom=5, deg=5, lower=0.0, upper=1.0,
                 alpha=0.0, f=None, singularity=None):
        self.description = 'Gaussian-Legendre Quadrature'
        check_alpha(alpha)
        ndom = check_node_type(ndom)
        deg = check_node_type(deg)
        h = (upper - lower)/ndom
        self.alpha = alpha
        self.lower = lower
        self.upper = upper
        self.f = f
        self.ndom = ndom
        self.deg = deg
        self.singularity = check_singularity(singularity, self.upper)
        self.points = self._gauss_points(ndom=ndom, deg=deg, h=h, lower=lower)
        self.weights = self._gauss_weights(ndom=ndom, deg=deg, h=h)
        self.initial_weights = self.weights.copy()
        self.update_weights(alpha=alpha)

    def update_weights(self, alpha=None):
        alpha = check_value(alpha, self.alpha, 'fractional order - alpha')
        self.alpha = alpha
        # update weights based on alpha
        self.weights = self.initial_weights*(
            self.singularity - self.points)**(-alpha)

    def integrate(self, f=None):
        f = check_value(f, self.f, 'function - f')
        self.f = f
        return (self.weights*f(self.points)).sum()

    @classmethod
    def _base_gauss_points(cls, deg):
        # base points
        gpts = .5 + .5*np.polynomial.legendre.leggauss(deg)[0]
        return gpts

    @classmethod
    def _base_gauss_weights(cls, deg, h):
        # define the Gauss weights for a deg-point quadrature rule
        w = .5*np.polynomial.legendre.leggauss(deg)[1]*h
        return w

    @classmethod
    def _interval_gauss_points(cls, base_gpts, ndom, deg, h, lower):
        # determines the Gauss points for all ndom intervals.
        gpts = np.zeros([deg*ndom])
        for gct in range(ndom):
            for ell in range(deg):
                gpts[(gct)*deg + ell] = ((gct)*h
                                         + base_gpts[ell]*h + lower)
        return gpts

    def _gauss_points(self, ndom, deg, h, lower):
        # base points
        gpts = self._base_gauss_points(deg)
        # determines the Gauss points for all Ndom intervals.
        gpoints = self._interval_gauss_points(gpts, ndom, deg, h, lower)
        return gpoints

    def _gauss_weights(self, ndom, deg, h):
        # determine the Gauss weights for a deg-point quadrature rule
        w = self._base_gauss_weights(deg, h)
        # copy the weights to form a vector for all Ndom intervals
        weights = w.copy()
        for _ in range(ndom-1):
            weights = np.concatenate((weights, w))
        return weights


# ---------------------
class GaussLaguerre:

    def __init__(self, deg=5, lower=0.0, upper=1.0, alpha=0.0,
                 f=None, extend_precision=True, n_digits=30,
                 singularity=None):
        self.description = 'Gaussian-Laguerre Quadrature'
        deg = check_node_type(deg)
        self.lower = lower
        self.upper = upper
        self.alpha = alpha
        self.deg = deg
        self.singularity = check_singularity(singularity, self.upper)
        self.f = f
        if extend_precision is False:
            points, weights = np.polynomial.laguerre.laggauss(deg=deg)
            self.points = 1 - np.exp(-points)
        else:
            points, weights = sp_gauss_laguerre(
                n=deg, n_digits=n_digits, alpha=0)
            points = [-p for p in points]
            points = sp.Array(points)
            self.points = sp.Array(
                np.ones(shape=len(points))) - points.applyfunc(sp.exp)
            weights = sp.Array(weights)
        self.weights = weights
        self.initial_weights = weights.copy()
        self.update_weights(alpha=alpha)

    def integrate(self, f=None):
        f = check_value(f, self.f, 'function - f')
        self.f = f
        # transform kernel
        span = self.upper - self.lower
        # check if sympy
        if isinstance(self.points, sp.Array):
            evalpoints = self.points.applyfunc(
                lambda x: span*x + self.lower)
            feval = evalpoints.applyfunc(f)
            s = 0
            for _, (w, f) in enumerate(zip(self.weights, feval)):
                s += w*f
            return np.float(s)
        else:
            s = (self.weights*(f(span*self.points + self.lower))).sum()
            return s

    def update_weights(self, alpha=None):
        alpha = check_value(alpha, self.alpha, 'fractional order - alpha')
        self.alpha = alpha
        span = self.singularity - self.lower
        # check if sympy
        if isinstance(self.points, sp.Array):
            coef = self.points.applyfunc(
                lambda x: span**(1-alpha)*(1-x)**(-alpha))
            wtmp = []
            for _, (c, w) in enumerate(zip(coef, self.initial_weights)):
                wtmp.append(c*w)
            self.weights = sp.Array(wtmp)
        else:
            coef = span**(1-alpha)*(1-self.points)**(-alpha)
            self.weights = self.initial_weights*coef


# ---------------------
class RiemannSum(object):

    def __init__(self, n=5, lower=0.0, upper=1.0, alpha=0.0, f=None):
        self.description = 'Riemann-Sum'
        check_alpha(alpha=alpha)
        n = check_node_type(n)
        self.alpha = alpha
        self.f = f
        self.n = n
        self.grid = self._rs_grid(lower, upper, n)
        self.points = self._rs_points(grid=self.grid)
        self.weights = self._rs_weights(grid=self.grid, alpha=alpha)
        self.lower = lower
        self.upper = upper

    def update_weights(self, alpha=None):
        alpha = check_value(alpha, self.alpha, 'fractional order - alpha')
        check_alpha(alpha=alpha)
        self.alpha = alpha
        self.weights = self._rs_weights(grid=self.grid, alpha=alpha)

    def integrate(self, f=None):
        f = check_value(f, self.f, 'function - f')
        self.f = f
        return (self.weights*f(self.points)).sum()

    @classmethod
    def _rs_grid(cls, lower, upper, n):
        return np.linspace(start=lower, stop=upper, num=n)

    @classmethod
    def _rs_points(cls, grid):
        jj = grid.size - 1
        return (grid[1:jj+1] + grid[0:jj])/2

    @classmethod
    def _rs_weights(cls, grid, alpha=0.0):
        jj = grid.size - 1
        term2 = (grid[jj] - grid[1:jj+1])**(1-alpha)
        term3 = (grid[jj] - grid[0:jj])**(1-alpha)
        return -1/(1-alpha)*(term2 - term3)


# ---------------------
class GaussLegendreRiemannSum(object):

    def __init__(self, ndom=5, deg=4, nrs=20, percent=0.9, ts=None,
                 lower=0.0, upper=1.0, alpha=0.0, f=None):
        self.description = 'Gaussian Quadrature, Riemann-Sum'
        # setup GQ points/weights
        if ts is not None:
            check_range(lower, upper, ts)
            switch_time = ts
        else:
            switch_time = (upper - lower)*percent + lower
        self.gleg = GaussLegendre(ndom=ndom, deg=deg, lower=lower,
                                  upper=switch_time, alpha=alpha,
                                  singularity=upper, f=f)
        # setup RS points/weights
        self.rs = RiemannSum(n=nrs, lower=switch_time,
                             upper=upper, alpha=alpha, f=f)
        self.alpha = alpha
        self.percent = percent
        self.f = f
        self.switch_time = switch_time

    def integrate(self, f=None):
        f = check_value(f, self.f, 'function - f')
        self.f = f
        return self.gleg.integrate(f=f) + self.rs.integrate(f=f)

    def update_weights(self, alpha=None):
        alpha = check_value(alpha, self.alpha, 'fractional order - alpha')
        self.alpha = alpha
        self.gleg.update_weights(alpha=alpha)
        self.rs.update_weights(alpha=alpha)


# ---------------------
class GaussLegendreGaussLaguerre(object):

    def __init__(self, ndom=5, gleg_deg=4, glag_deg=20, percent=0.9, ts=None,
                 lower=0.0, upper=1.0, alpha=0.0, f=None,
                 extend_precision=True, n_digits=30):
        self.description = 'Hybrid: Gauss-Legendre, Gauss-Laguerre'
        # setup GLeg points/weights
        if ts is not None:
            check_range(lower, upper, ts)
            switch_time = ts
        else:
            switch_time = (upper - lower)*percent + lower
        self.gleg = GaussLegendre(ndom=ndom, deg=gleg_deg, lower=lower,
                                  upper=switch_time, alpha=alpha,
                                  singularity=upper, f=f)
        # setup GLag points/weights
        self.glag = GaussLaguerre(deg=glag_deg, lower=switch_time,
                                  upper=upper, alpha=alpha, f=f,
                                  extend_precision=extend_precision,
                                  n_digits=n_digits)
        self.alpha = alpha
        self.percent = percent
        self.f = f

    def integrate(self, f=None):
        f = check_value(f, self.f, 'function - f')
        self.f = f
        return self.gleg.integrate(f=f) + self.glag.integrate(f=f)

    def update_weights(self, alpha=None):
        alpha = check_value(alpha, self.alpha, 'fractional order - alpha')
        self.alpha = alpha
        self.gleg.update_weights(alpha=alpha)
        self.glag.update_weights(alpha=alpha)
