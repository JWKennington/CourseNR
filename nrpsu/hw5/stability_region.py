# -*- coding: utf-8 -*-
# %%
import matplotlib.pyplot as plt
import numpy
import numpy as np


def amplification_factor(A, b, z):
    lhs = (numpy.eye(N=A.shape[0]) - z * A)
    lhs_inv = np.linalg.inv(lhs)
    rhs = z * numpy.ones(shape=(A.shape[0],))
    k_vec = numpy.matmul(lhs_inv, rhs)
    y_1 = numpy.matmul(b.T, k_vec)
    return numpy.sum(numpy.abs(y_1))


def tabulate_amplification_factor(A, b, z_re_grid, z_im_grid):
    @numpy.vectorize
    def amp_vec(z_re, z_im):
        return amplification_factor(A, b, z_re + z_im*1j)

    return amp_vec(z_re_grid, z_im_grid)


def plot_stability_region(A, b, zmax=5.0, nz=512):
    # Create a grid
    z1d = np.linspace(-zmax, zmax, nz)
    z_re, z_im = np.meshgrid(z1d, z1d, indexing='ij')
    fac = tabulate_amplification_factor(A, b, z_re, z_im)
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])
    ax.set_xlabel(r"Re")
    ax.set_ylabel(r"Im")
    ax.set_aspect('equal')
    ax.pcolormesh(z_re, z_im, fac < 1.0, cmap='cividis')
    ax.grid(visible=True)
    return fig


# %%
# Explicit Euler
A = np.array([[0]])
b = np.array([[1]])
fig = plot_stability_region(A, b)
fig.gca().set_title("Forward Euler Stability Region")
# %%
# Implicit Euler
A = np.array([[1]])
b = np.array([[1]])
fig = plot_stability_region(A, b)
fig.gca().set_title("Implicit Euler Stability Region")
# %%
# Heun's method
A = np.array([[0, 0], [1, 0]])
b = np.array([0.5, 0.5])
fig = plot_stability_region(A, b)
fig.gca().set_title("RK2 Stability Region")
# %%
# Classical RK4
A = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]])
b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
fig = plot_stability_region(A, b)
fig.gca().set_title("RK4 Stability Region")
# %%
# SSP-RK3
A = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1 / 4, 1 / 4, 0.0]])
b = np.array([1 / 6, 1 / 6, 2 / 3])
fig = plot_stability_region(A, b)
fig.gca().set_title("SSP-RK3 Stability Region")
# %%
# Gauss-Legendre RK2
s3 = np.sqrt(3.0)
A = np.array([
    [1 / 4, 1 / 4 - s3 / 6],
    [1 / 4 + s3 / 6, 1 / 4]])
b = np.array([1 / 2, 1 / 2])
fig = plot_stability_region(A, b)
fig.gca().set_title("Gauss-Legendre-RK2 Stability Region")
# %%
plt.show()
