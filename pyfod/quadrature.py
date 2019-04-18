import numpy as np
import sympy as sp
from sympy.integrals.quadrature import gauss_gen_laguerre as sp_gauss_laguerre
from pyfod.utilities import check_alpha
from pyfod.utilities import check_value
from pyfod.utilities import check_singularity
from pyfod.utilities import check_node_type


# ---------------------
class GaussLegendre:

    def __init__(self, Ndom=5, deg=5, start=0.0, finish=1.0,
                 alpha=0.0, f=None, singularity=None):
        self.description = 'Gaussian-Legendre Quadrature'
        check_alpha(alpha)
        Ndom = check_node_type(Ndom)
        deg = check_node_type(deg)
        h = (finish - start)/Ndom
        self.alpha = alpha
        self.f = f
        self.Ndom = Ndom
        self.deg = deg
        self.finish = finish
        self.singularity = check_singularity(singularity, self.finish)
        self.points = self._gauss_points(Ndom=Ndom, deg=deg, h=h, start=start)
        self.weights = self._gauss_weights(Ndom=Ndom, deg=deg, h=h)
        self.initial_weights = self.weights.copy()
        self.update_weights(alpha=alpha)

    def update_weights(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
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
    def _interval_gauss_points(cls, base_gpts, Ndom, deg, h, start):
        # determines the Gauss points for all Ndom intervals.
        Gpts = np.zeros([deg*Ndom])
        for gct in range(Ndom):
            for ell in range(deg):
                Gpts[(gct)*deg + ell] = ((gct)*h
                                         + base_gpts[ell]*h + start)
        return Gpts

    def _gauss_points(self, Ndom, deg, h, start):
        # base points
        gpts = self._base_gauss_points(deg)
        # determines the Gauss points for all Ndom intervals.
        Gpts = self._interval_gauss_points(gpts, Ndom, deg, h, start)
        return Gpts

    def _gauss_weights(self, Ndom, deg, h):
        # determine the Gauss weights for a deg-point quadrature rule
        w = self._base_gauss_weights(deg, h)
        # copy the weights to form a vector for all Ndom intervals
        weights = w.copy()
        for gct in range(Ndom-1):
            weights = np.concatenate((weights, w))
        return weights


# ---------------------
class GaussLaguerre:

    def __init__(self, N=5, start=0.0, finish=1.0, alpha=0.0,
                 f=None, extend_precision=True, n_digits=30):
        self.description = 'Gaussian-Laguerre Quadrature'
        N = check_node_type(N)
        self.start = start
        self.finish = finish
        self.alpha = alpha
        self.N = N
        self.f = f
        if extend_precision is False:
            points, weights = np.polynomial.laguerre.laggauss(deg=N)
            self.points = 1 - np.exp(-points)
        else:
            points, weights = sp_gauss_laguerre(
                    n=N, n_digits=n_digits, alpha=0)
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
        span = self.finish - self.start
        # check if sympy
        if isinstance(self.points, sp.Array):
            evalpoints = self.points.applyfunc(
                    lambda x: span*x + self.start)
            feval = evalpoints.applyfunc(f)
            s = 0
            for ii, (w, f) in enumerate(zip(self.weights, feval)):
                s += w*f
            return np.float(s)
        else:
            s = (self.weights*(f(span*self.points + self.start))).sum()
            return s

    def update_weights(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        self.alpha = alpha
        span = self.finish - self.start
        # check if sympy
        if isinstance(self.points, sp.Array):
            coef = self.points.applyfunc(
                    lambda x: span**(1-alpha)*(1-x)**(-alpha))
            wtmp = []
            for ii, (c, w) in enumerate(zip(coef, self.initial_weights)):
                wtmp.append(c*w)
            self.weights = sp.Array(wtmp)
        else:
            coef = span**(1-alpha)*(1-self.points)**(-alpha)
            self.weights = self.initial_weights*coef


# ---------------------
class RiemannSum(object):

    def __init__(self, N=5, start=0.0, finish=1.0, alpha=0.0, f=None):
        self.description = 'Riemann-Sum'
        check_alpha(alpha=alpha)
        N = check_node_type(N)
        self.alpha = alpha
        self.f = f
        self.N = N
        self.grid = self._rs_grid(start, finish, N)
        self.points = self._rs_points(grid=self.grid)
        self.weights = self._rs_weights(grid=self.grid, alpha=alpha)

    def update_weights(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        check_alpha(alpha=alpha)
        self.alpha = alpha
        self.weights = self._rs_weights(grid=self.grid, alpha=alpha)

    def integrate(self, f=None):
        f = check_value(f, self.f, 'function - f')
        self.f = f
        return (self.weights*f(self.points)).sum()

    @classmethod
    def _rs_grid(cls, start, finish, N):
        return np.linspace(start=start, stop=finish, num=N)

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

    def __init__(self, NGLeg=5, NRS=20, pGLeg=0.9,
                 start=0.0, finish=1.0, alpha=0.0, f=None):
        self.description = 'Gaussian Quadrature, Riemann-Sum'
        # setup GQ points/weights
        switch_time = (finish - start)*pGLeg
        self.GLeg = GaussLegendre(N=NGLeg, start=start, finish=switch_time,
                                  alpha=alpha, singularity=finish, f=f)
        # setup RS points/weights
        self.RS = RiemannSum(N=NRS, start=switch_time,
                             finish=finish, alpha=alpha, f=f)
        self.alpha = alpha
        self.pGLeg = pGLeg
        self.f = f

    def integrate(self, f=None):
        f = check_value(f, self.f, 'function - f')
        self.f = f
        return self.GLeg.integrate(f=f) + self.RS.integrate(f=f)

    def update_weights(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        self.alpha = alpha
        self.GLeg.update_weights(alpha=alpha)
        self.RS.update_weights(alpha=alpha)


# ---------------------
class GaussLegendreGaussLaguerre(object):

    def __init__(self, NGLeg=5, NGLag=20, pGLeg=0.9,
                 start=0.0, finish=1.0, alpha=0.0, f=None,
                 extend_precision=True, n_digits=30):
        self.description = 'Hybrid: Gauss-Legendre, Gauss-Laguerre'
        # setup GLeg points/weights
        switch_time = (finish - start)*pGLeg
        self.GLeg = GaussLegendre(N=NGLeg, start=start, finish=switch_time,
                                  alpha=alpha, singularity=finish, f=f)
        # setup GLag points/weights
        self.GLag = GaussLaguerre(N=NGLag, start=switch_time,
                                  finish=finish, alpha=alpha, f=f,
                                  extend_precision=extend_precision,
                                  n_digits=n_digits)
        self.alpha = alpha
        self.pGLeg = pGLeg
        self.f = f

    def integrate(self, f=None):
        f = check_value(f, self.f, 'function - f')
        self.f = f
        return self.GLeg.integrate(f=f) + self.GLag.integrate(f=f)

    def update_weights(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        self.alpha = alpha
        self.GLeg.update_weights(alpha=alpha)
        self.GLag.update_weights(alpha=alpha)
