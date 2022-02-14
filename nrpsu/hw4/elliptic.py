import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import bicgstab, spsolve, LinearOperator
from sympy import Symbol, diff, lambdify


def generate_analytic_solution():
    """
    Computes the analytic solution
    Returns
    -------
    Psi(x)
        lambda function.
    """
    r = Symbol("r", positive=True)
    Psi = r ** 2 / (1 + r ** 3)
    return lambdify(r, Psi)


def generate_analytic_source():
    """
    Computes g = \nabla^2 \Psi / \Psi^4 assuming  \Psi = r^2/(1 + r^3)
    Returns
    -------
    g(x)
        lambda function.
    """
    r = Symbol("r", positive=True)
    Psi = r ** 2 / (1 + r ** 3)
    LHS = diff(r ** 2 * diff(Psi, r), r) / r ** 2
    g = LHS / Psi ** 4
    return lambdify(r, g.simplify())


class LinearizedProblem(LinearOperator):
    """
    This class implements the linearized problem
        [ r^2 Psi_r(r) ]_r + u(r) r^2 Psi(r) = v(r) r^2
    It inherits from LinearOperator, so it can be used with the iterative
    solvers implemented in scipy.linalg
    """

    def __init__(self, rp, up):
        """
        Initializes the linear operator.
        Parameters
        ----------
        rp : numpy array
            spatial grid.
        up : numpy array
            function u(x).
        Returns
        -------
        None.
        """
        assert rp.shape == up.shape
        # Initializes the base class
        super().__init__(rp.dtype, (rp.shape[0], rp.shape[0]))
        self.rp = rp
        self.up = up

    def full_matrix(self):
        """
        Return the linear operator as a full matrix
        Note. It would be more efficient to generate the matrix directly,
        but this is not a bottleneck here.
        Returns
        -------
        A : numpy array
            full matrix.
        """
        return self.matmat(np.eye(self.shape[0]))

    def _matvec(self, Phi_):
        """
        Parameters
        ----------
        Phi : numpy array of shape (N,)
            input array.
        Returns
        -------
        A_Phi : numpy array
            A Phi.
        """
        out = Phi_.copy()

        # First derivatives are on shifted 1/2 grid
        d_r = self.rp[1] - self.rp[0]
        d_Phi = (Phi_[1:] - Phi_[:-1]) / d_r
        r_sq = ((self.rp[1:] + self.rp[:-1]) / 2) ** 2
        H = d_Phi * r_sq

        # Now we take derivative of H on non-shifted grid
        d_H = (H[1:] - H[:-1]) / d_r
        out[1:-1] = d_H

        # Add exact term
        g = generate_analytic_source()(self.rp[1:-1])
        out[1:-1] -= 4 * g * self.rp[1:-1] ** 2 * Phi_[1:-1] * self.up[1:-1] ** 3

        # Boundary conditions for A * Phi
        out[0] = 0
        out[-1] = 0

        return out


class NonLinearSolver:
    """
    This class solves
        \nabla^2 \Psi(r) = \Psi^4(r) g(r)
    assuming spherical symmetry.
    The equation using a Newton method:
        \Psi_{k+1} = \Psi_k + \Phi,
    where Phi solves
        (\nabla^2 - 4 \Psi^3 g) \Phi = \Psi^4 g - \nabla^2 \Psi
    We solve this system using second order finite differencing.
    """

    def __init__(self, N, Rmax, fun):
        """
        Construct the nonlinear solver.
        Parameters
        ----------
        N : int
            number of grid points.
        Rmax : real
            maximum radius.
        fun : lambda function
            source function.
        Returns
        -------
        None.
        """
        self.rp = np.linspace(0.0, Rmax, N)
        self.h = self.rp[1]
        # Note that we are using gp in the interior of the grid
        self.gp = fun(self.rp[1:-1])

    def solve(self, Psi0, mu=0.1, nmax=30, reltol=1e-10, abstol=1e-15):
        """
        Solve the nonlinear problem given an initial guess
        Parameters
        ----------
        Psi0 : lambda function
            Initial guess.
        mu : float
            relaxation parameter. The default is 0.1.
        nmax : int, optional
            Maximum number of iterations. The default is 100.
        reltol : float, optional
            Relative tolerance. The default is 1e-5.
        abstol : float, optional
            Absolute tolerance. The default is 1e-7.
        Returns
        -------
        Psi : numpy array
            Numerical solution.
        """
        # We make aliases to avoid having to write self all the time
        rp = self.rp
        Psi = Psi0(rp)
        step = 0
        while True:
            step += 1
            # Inpose BCs
            Psi[0] = 4. / 3. * Psi[1] - 1. / 3. * Psi[2]
            Psi[-1] = 1. / rp[-1] * (4. / 3. * (rp[-2] * Psi[-2]) -
                                     1. / 3. * (rp[-3] * Psi[-3]))
            dPsi = self.step(Psi)
            Psi += mu * dPsi  # small mu is better for converegence, but slower
            delta = self.h * np.sum(np.abs(dPsi))
            nrm = self.h * np.sum(np.abs(Psi))
            print("step:", step, "delta:", delta)
            if step >= nmax or delta < max(abstol, reltol * nrm):
                break
        return Psi

    def step(self, Psi):
        """Makes a step of the nonlinear solver

        Solves the equation A Phi = - F_n

        Parameters
        ----------
        Psi : numpy array
            current guess for the solution.
        Returns
        -------
        dPsi : numpy array
            update vector.
        """
        rhs = Psi.copy()
        g = generate_analytic_source()(self.rp[1:-1])

        # Compute discretized F
        # First derivatives are on shifted 1/2 grid
        d_r = self.rp[1] - self.rp[0]
        d_Psi = (Psi[1:] - Psi[:-1]) / d_r
        r_sq = ((self.rp[1:] + self.rp[:-1]) / 2) ** 2
        H = d_Psi * r_sq
        # Now we take derivative of H on non-shifted grid
        d_H = (H[1:] - H[:-1]) / d_r

        rhs[1: -1] = d_H
        rhs[1: -1] -= self.rp[1:-1] ** 2 * g * Psi[1:-1] ** 4

        # Boundary conditions for RHS
        rhs[0] = 0
        rhs[-1] = 0

        A = LinearizedProblem(rp=self.rp, up=Psi)
        Phi, info = bicgstab(A, rhs, maxiter=10)
        return Phi


def solve(N, Rmax, Psi0):
    sol = NonLinearSolver(N, Rmax, generate_analytic_source())
    Psih = sol.solve(Psi0)
    Psi = generate_analytic_solution()
    err = Psi(sol.rp) - Psih
    return sol.rp, Psih, err


def solve_and_plot():
    # Maximum radius
    Rmax = 10

    # Initial guess
    def Psi0(r): return r ** 2 / (5 + r ** 3)

    # Solve
    rp, Psih, err = solve(400, Rmax, Psi0)
    h = rp[1] - rp[0]
    # Print error
    print("")
    print("L1 error:", np.sum(np.abs(err)) * h)
    print("L2 error:", np.sqrt(np.sum(err ** 2) * h))
    print("Linf error:", np.max(np.abs(err)))
    Psi = generate_analytic_solution()
    plt.figure()
    plt.plot(rp, Psih, '.', label="Numerical solution")
    plt.plot(rp, Psi0(rp), '--', label="Initial guess")
    plt.plot(rp, Psi(rp), '-', label="Analytical solution")
    plt.xlabel(r"$r$")
    plt.legend()
    plt.figure()
    plt.plot(rp, np.abs(err), '-', label="Error")
    plt.xlabel(r"$r$")
    plt.yscale("log")
    plt.legend()
    plt.show()


def convergence_study():
    # Maximum radius
    Rmax = 10

    # Initial guess
    def Psi0(r): return r ** 2 / (5 + r ** 3)

    npoints = np.array([25, 50, 100, 200, 400])
    err1, err2, errInf = [], [], []
    for N in npoints:
        print("-" * 73)
        print("N:", N)
        print("-" * 73)
        rp, Psih, err = solve(N, Rmax, Psi0)
        print("-" * 73)
        h = rp[1] - rp[0]
        err1.append(np.sum(np.abs(err)) * h)
        err2.append(np.sqrt(np.sum(err ** 2) * h))
        errInf.append(np.max(np.abs(err)))
    plt.figure()
    plt.loglog(npoints, err2[0] * (npoints[0] / npoints)
               ** (2), 'k-', label="Theory")
    plt.loglog(npoints, err2, 'ro', label="Results")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$L^2$-error")
    plt.show()


def main():
    solve_and_plot()


if __name__ == '__main__':
    main()

