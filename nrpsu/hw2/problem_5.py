"""
This script solves the BVP

    - [k(x) u'(x)]' = eta(x) - alpha(x) u(x)
    u(0) = u(1) = 0

We use uniform grid and finite differencing to convert this to a linear
problem in the form

    A u = f

The problem is then solved using the conjugate gradient method
"""

import matplotlib.pyplot as plt
import numpy
from scipy.sparse.linalg import cg, LinearOperator


class FiniteDiffOperator(LinearOperator):
    """
    This class represents the finite differencing discretization of

        - [k(x) u'(x)]' + alpha(x) u(x)

    It inherits from LinearOperator, so it can be used with the iterative
    solvers implemented in scipy.linalg
    """

    def __init__(self, h, kappa, alpha):
        """
        Initialize the FiniteDiffOperator

        Parameters
        ----------
        h : real
            Grid spacing.
        kappa : numpy array
            An array storing kappa at the grid point locations.
        alpha : numpy array
            An array storing alpha at the grid point locations.

        Returns
        -------
        None.
        """
        assert kappa.shape == alpha.shape

        # Initializes the base class
        super().__init__(kappa.dtype, (kappa.shape[0], kappa.shape[0]))

        self.h = h
        self.kappa = kappa
        self.alpha = alpha

    def _matvec(self, u):
        """
        Parameters
        ----------
        u : numpy array of shape (N,)
            input array.

        Returns
        -------
        v : numpy array
            A*u
        """
        # Output array
        v = u.copy()

        v[0] = u[0]
        v[-1] = u[-1]

        u_prime = (u[1:] - u[:-1]) / self.h

        kappa_avg = (self.kappa[1:] + self.kappa[:-1]) / 2

        flux = u_prime * kappa_avg

        flux_prime = (flux[1:] - flux[:-1]) / self.h

        v[1:-1] = flux_prime

        return -v + self.alpha * u


class HeatEquationSolver:
    def __init__(self, kappa, eta, alpha, N):
        """
        Initialize the heat equation solver

        Parameters
        ----------
        N : integer
            Number of grid points.
        kappa : real function
            Heat diffusion coefficient as a function of position.
        eta : real function
            Heating source as a function of position.
        alpha : real function
            Absorption opacity as a function of position.

        Returns
        -------
        None.
        """
        self.initialized = False
        self.kappa = kappa
        self.eta = eta
        self.alpha = alpha
        self.set_npoints(N)

    def set_npoints(self, N):
        """
        Set the number of points and initialize the linear operator

        Parameters
        ----------
        N : integer
            Number of grid points.

        Returns
        -------
        None.

        """
        self.N = N
        self.xp = numpy.linspace(0, 1, self.N + 2)
        self.h = self.xp[1] - self.xp[0]
        self.A = FiniteDiffOperator(self.h, self.kappa(self.xp),
                                    self.alpha(self.xp))
        self.b = self.eta(self.xp)

        # Apply boundary conditions to b
        self.b[0] = 0
        self.b[-1] = 0
        self.initialized = True

    def solve(self, opt={"tol": 1e-8}):
        """
        Solve the differential equation

        Parameters
        ----------
        opt : dictionary
              options for the linear solver

        Returns
        -------
        None
        """
        assert self.initialized
        self.u, ierr = cg(self.A, self.b, **opt)
        if ierr > 0:
            print("Warning: CG did not converge to desired accuracy!")
        if ierr < 0:
            raise Exception("Error: invalid input for CG")


@numpy.vectorize
def alpha(x):
    return 1.0


@numpy.vectorize
def eta(x):
    return ((1 + numpy.pi ** 2 * kappa(x)) * numpy.sin(numpy.pi * x) +
            numpy.pi * (2 * x - 1) * numpy.cos(numpy.pi * x))


@numpy.vectorize
def kappa(x):
    return 1.0 + x * (1.0 - x)


def u_ex(x):
    return numpy.sin(numpy.pi * x)


def calc_n(n: int, verbose: bool = False):
    if verbose:
        print('Calculating N: {:.1e}'.format(n))
    xp = numpy.linspace(0, 1, n + 2)
    heat = HeatEquationSolver(kappa, eta, alpha, n)
    heat.solve()
    sol = heat.u
    return xp, sol


def l2_norm(u, u_ex):
    return numpy.sqrt(numpy.sum((u - u_ex) ** 2))


def plot_mpl(xp, sol):
    # Run and plot errors
    # L2 = (h*numpy.sum((heat-u_ex)**2))**(0.5)
    plt.plot(xp, u_ex(xp))
    plt.plot(xp, sol)
    plt.show()


def plot_convergence(ks, norms):
    plt.figure()
    plt.plot(1/ks, -1/numpy.log10(norms))
    plt.show()


def main():
    # xp, sol = calc_n(1000)
    # plot_mpl(xp, sol)

    # Test convergence
    ks = numpy.arange(1, 5, 1.0)
    results = [calc_n(int(10**k), verbose=True) for k in ks]
    norms = numpy.array([l2_norm(sol, u_ex(xp)) for xp, sol in results])

    plot_convergence(ks, norms)




if __name__ == '__main__':
    main()
