"""Problem 2, symbolic calculations using the pystein library"""

import itertools
import sympy
from sympy.diffgeom import Manifold, Patch
from pystein import coords, metric, curvature
from pystein.utilities import tensor_pow as tpow, full_simplify

M = Manifold('M', dim=3)
P = Patch('origin', M)
r, theta, phi = sympy.symbols('r theta phi', nonnegative=True)
cs = coords.CoordSystem('spherical-polar', P, [r, theta, phi])
dr, dtheta, dphi = cs.base_oneforms()

ds2 = tpow(dr, 2) + r ** 2 * tpow(dtheta, 2) + r ** 2 * sympy.sin(theta) ** 2 * tpow(dphi, 2)
gamma = metric.Metric(twoform=ds2)

christoffels = [((i, j, k), full_simplify(curvature.christoffel_symbol_component(i, j, k, gamma)))
                for i, j, k in itertools.product(range(3), range(3), range(3))]
curvature.display_components([c for c in christoffels if c[1]])

riccis = [((i, j), full_simplify(curvature.ricci_tensor_component(i, j, gamma).doit()))
          for i, j in itertools.product(range(3), range(3))]
curvature.display_components(riccis)

M = sympy.symbols('M')
beta_j = sympy.Matrix([sympy.sqrt(2 * M / r), 0, 0])
beta_j

def cov_deriv_one_form(i, j, form, g):
    coord_syms = g.coord_system.base_symbols()
    v = sympy.Derivative(form[j], coord_syms[i])

    for k in range(len(coord_syms)):
        v -= curvature.christoffel_symbol_component(k, i, j, g) * form[k]

    return full_simplify(v.doit())

def k_ij(i, j, beta_j, g, alpha = 1):
    d_i_beta_j = cov_deriv_one_form(i, j, beta_j, g)
    d_j_beta_i = cov_deriv_one_form(j, i, beta_j, g)
    return - sympy.Rational(1, 2) / alpha * (d_i_beta_j + d_j_beta_i)

K_ij = sympy.Matrix([[k_ij(0, j, beta_j, gamma) for j in range(3)],
                     [k_ij(1, j, beta_j, gamma) for j in range(3)],
                     [k_ij(2, j, beta_j, gamma) for j in range(3)]])

sympy.trace(gamma.matrix.inv() * K_ij)
