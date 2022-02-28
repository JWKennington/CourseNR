from math import pi, sqrt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.sparse.linalg import LinearOperator, cg


class LaplaceOperator(LinearOperator):
    """
    This class represents the finite differencing discretization of
        jac * \partial_xi [ jac \partial_xi u(xi) ]
    where jac is a given grid function.
    This is intended to discretize the 1D Laplace operator on mapped grids.
    It inherits from LinearOperator, so it can be used with the iterative
    solvers implemented in scipy.linalg
    """

    def __init__(self, xic, xif, jac):
        """
        Initialize the FiniteDiffOperator
        Parameters
        ----------
        xic : numpy array
            Numerical grid (cell centers).
        xif: numpy array
            Numerical grid (cell faces).
        jac : lambda function
            The Jacobian of the coordinate transformation.
        Returns
        -------
        None.
        """
        assert xif.shape[0] == xic.shape[0] + 1
        # Initializes the base class
        super().__init__(xic.dtype, (xic.shape[0], xic.shape[0]))
        self.xic = xic
        self.xif = xif
        self.hic = self.xic[1] - self.xic[0]
        self.hif = self.xif[1] - self.xif[0]
        self.jac = jac

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
        F = np.zeros(u.size + 1)

        u_x = (u[1:] - u[:-1]) / self.hic
        F[1:-1] = self.jac(self.xif[1:-1]) * u_x

        inner_x = (F[1:] - F[:-1]) / self.hif
        return self.jac(self.xic) * inner_x


class CrankNicholsonOperator(LinearOperator):
    """
    Linear operator to invert for the Crank-Nicholson scheme:
        [I + mu A]
    where mu is a given coefficient (typically -1/2 dt)
    """

    def __init__(self, A, mu):
        """
        Initializes the operator
        Parameters
        ----------
        A : LinearOperator
            Discrete Laplace operator.
        mu : float
            Coefficient
        Returns
        -------
        None.
        """
        super().__init__(A.dtype, A.shape)
        self.A = A
        self.mu = mu

    def _matvec(self, u):
        v = u.copy()
        return v + self.mu * self.A.matvec(u)


class ThermalWaveSolver:
    """
    This equation solves the 1+1 heat equation in a compactified domain
    using the Crank-Nicholson method.
    """

    def __init__(self, N):
        """
        Initializes the solver
        Parameters
        ----------
        N : int
            Number of grid points.
        Returns
        -------
        None.
        """
        self.N = N
        # Grid
        # Cell faces
        self.xif = np.linspace(-1, 1, N + 1)
        self.xf = np.tan(pi * self.xif / 2)
        # Cell centers
        self.xic = 0.5 * (self.xif[1:] + self.xif[:-1])
        self.xc = np.tan(pi * self.xic / 2)
        # Jacobian
        self.jac = lambda xi: 2 * np.cos(np.pi * xi / 2) ** 2 / np.pi
        # Discrete Laplace operator
        self.A = LaplaceOperator(self.xic, self.xif, self.jac)

    def animate(self, tmin=0.1, dt=0.05, tmax=1.1, outevery=1, theta=0.5, opt=None):
        """
        Solves the diffusion equation and makes an animation
        Parameters
        ----------
        tmin : float, optional
            Initial time. The default is 0.0
        dt   : float, optional
            Time step. The default is 0.1
        tmax : float, optional
            Final time. The default is 16.
        outevery : int, optional
            Output frequency. The default is 1.
        theta : float, optional
            Theta method to use. The default is 0.5 (Crank-Nicholson).
        opt : dictionary, optional
            Options for the CG solver. The default is {"tol": 1e-8}.
        Returns
        -------
        None.
        """
        if opt is None:
            opt = {"tol": 1e-8}
        times, U = self.solve(tmin, dt, tmax, outevery, theta, opt)
        fig, ax = plt.subplots()
        ln1, = plt.plot([], [], 'r.', label="Numerical solution")
        ln2, = plt.plot([], [], 'k-', label="Analytical solution")
        time_lab = ax.text(0.5, 0.95, "", transform=ax.transAxes, ha='center')
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u$")

        def init():
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)
            # ax.set_yscale("log")
            return ln1, ln2, time_lab

        def update(i):
            t = max(times[i], 1e-10)
            Uex = 1 / sqrt(4 * pi * t) * np.exp(-self.xc ** 2 / (4 * t))
            ln1.set_data(self.xic, U[i])
            ln2.set_data(self.xic, Uex)
            time_lab.set_text(r"$t = {:.2f}$".format(times[i]))
            # ax.set_title("r$t = {}$".format(time[i]))
            return ln1, ln2, time_lab

        ani = FuncAnimation(fig, update, frames=range(len(U)), init_func=init, blit=True)
        ani.save("09-thermal-wave.gif")

    def delta_function(self):
        """
        Creates a discrete delta function centered at the origin
        Returns
        -------
        U : numpy array
        """
        i0 = self.N // 2
        delta = np.zeros_like(self.xc)
        delta[i0] = 1.0 / (self.xf[i0 + 1] - self.xf[i0])
        return delta

    def gaussian(self, t):
        """
        Creates a Gaussian profile
        Parameters
        ----------
        t : float
            Time.
        Returns
        -------
        U : numpy array
        """
        assert t > 0
        return 1 / sqrt(4 * pi * t) * np.exp(-self.xc ** 2 / (4 * t))

    def integrate(self, U):
        """
        Computes the integral of a grid function
        Parameters
        ----------
        U : numpy array
            Grid function to integrate.
        Returns
        -------
        v : float
            Integral of U on the real line.
        """
        # Jacobian |dx/dxi|
        vol = np.pi / (2 * np.cos(np.pi / 2 * self.xic) ** 2)
        return np.sum(U * vol * np.diff(self.xif))

    def step(self, U, dt, theta=0.5, opt={"tol": 1e-8}):
        """
        Make a single step of the Crank-Nicholson scheme
        Parameters
        ----------
        U : numpy array
            Solution at the beginning of the time step.
        dt : float
            Time step.
        theta : float
            Theta parameter (0.0 for explicit Euler,
                             0.5 for Crank-Nicholson,
                             1.0 for implicit Euler)
        opt : dictionary, optional
            Options for the CG solver. The default is {"tol": 1e-8}.
        Returns
        -------
        Unew : numpy array
            Solution at the end of the time step.
        ierr : int
            Error code from lingalg.cg
        """
        rhs = dt * (1 - theta) * self.A.matvec(U) + U
        lhs = CrankNicholsonOperator(A=self.A, mu=-dt * theta)
        return cg(A=lhs, b=rhs, **opt)

    def solve(self, tmin=0.1, dt=0.05, tmax=1.1, outevery=0, theta=0.5, opt=None):
        """
        Solves the diffusion equation
        Parameters
        ----------
        tmin : float, optional
            Initial time. The default is 0.0
        dt   : float, optional
            Time step. The default is 0.1
        tmax : float, optional
            Final time. The default is 16.
        outevery : int, optional
            Output frequency. The default is 1.
        theta : float, optional
            Theta method to use. The default is 0.5 (Crank-Nicholson).
        opt : dictionary, optional
            Options for the CG solver. The default is {"tol": 1e-8}.
        Returns
        -------
        times : list
            Solution times
        U : list of numpy arrays
            Numerical solution
        """
        # Initial conditions
        if opt is None:
            opt = {"tol": 1e-8}
        if tmin == 0.0:
            U = [self.delta_function()]
        else:
            U = [self.gaussian(tmin)]
        Unew = U[0].copy()
        times = [tmin]
        # Compute the solution and store all temporary results
        time, idx = tmin, 0
        while time < tmax:
            Unew, ierr = self.step(Unew, dt, theta=0.5, opt=opt)
            if ierr > 0:
                print("Warning: CG did not converge to desired accuracy!")
            if ierr < 0:
                raise Exception("Error: invalid input for CG")
            time += dt
            idx += 1
            if outevery > 0 and idx % outevery == 0:
                U.append(Unew)
                times.append(time)
        U.append(Unew)
        times.append(time)
        return times, U


def plot(resu, err):
    plt.figure()
    plt.loglog(resu, err, 'ro')
    plt.loglog(resu, err[0] * (resu / resu[0]) ** (-2), 'k-', label='2nd order')
    plt.xlabel(r"$N$")
    plt.ylabel(r"$\| e \|_2$")
    plt.legend()
    plt.savefig("09-thermal-wave-conv.pdf")
    plt.show()


def main():
    # %% Solve once and make a movie
    solver = ThermalWaveSolver(100)
    solver.animate(tmin=0.1, dt=0.05, tmax=1.1, theta=0.5)
    # %% Resolution study
    resu = [50, 100, 200, 400, 600, 1200]
    err = []
    for N in resu:
        solver = ThermalWaveSolver(N)
        t, U = solver.solve(tmin=0.1, dt=5 / N, tmax=1.1, outevery=0, theta=1.0)
        Uex = solver.gaussian(t[-1])
        err.append(sqrt(solver.integrate((U[-1] - Uex) ** 2)))
    err = np.array(err)
    resu = np.array(resu)

    plot(resu, err)


if __name__ == '__main__':
    main()
