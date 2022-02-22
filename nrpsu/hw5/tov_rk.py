"""
Tolman-Oppenheimer-Volkoff This solves the TOV equations using a simple FD approach

Code borrowed from NR course, modified for use in GR II
"""
import collections
import functools
import types
import typing

import numpy
from matplotlib import pyplot
from scipy import integrate, optimize

from nrpsu.hw5 import rk


def scale_density(rho_bar: float, index: float, gas_const: float):
    return rho_bar * (gas_const ** (-index))


def scale_pressure(pressure_bar: float, index: float, gas_const: float):
    return pressure_bar * (gas_const ** (-index))


def scale_radius(radius_bar: float, index: float, gas_const: float):
    return radius_bar * (gas_const ** (index / 2))


def polytropic_pressure(density_rest: float, index: float = 1.0, gas_const: float = 1.0):
    return gas_const * density_rest ** (1.0 + 1.0 / index)


def polytropic_density_rest(density: float = None, pressure: float = None, index: float = 1.0, gas_const: float = 1.0):
    if len([x for x in (density, pressure) if x is not None]) != 1:
        raise ValueError('Must specify only one of {{density_init, pressure}}')

    if pressure is not None:
        return (pressure / gas_const) ** (index / (index + 1))

    raise NotImplementedError


def polytropic_density(density_rest: float = None, pressure: float = None, index: float = 1.0, gas_const: float = 1.0):
    if len([x for x in (density_rest, pressure) if x is not None]) != 1:
        raise ValueError('Must specify only one of {{density_init, pressure}}')

    if density_rest is not None:
        return index * gas_const * density_rest ** (1.0 / index)

    return index * pressure + polytropic_density_rest(pressure=pressure)


def d_mass_d_rad(radius: float, density: float):
    return 4 * numpy.pi * radius ** 2 * density


def d_pressure_d_rad(radius: float, density: float, mass: float, pressure: float):
    if mass == 0:
        return 0
    return (-density / radius ** 2 * (1 + pressure / density) *
            (mass + 4 * numpy.pi * pressure * radius ** 3) / (1 - 2 * mass / radius))


def d_phi_d_rad(radius: float, density: float, mass: float, pressure: float):
    if mass == 0:
        return 0
    return (- 1 / density * d_pressure_d_rad(radius, density, mass, pressure) / (1 + pressure / density))


class State:
    __slots__ = 'mass pressure phi'.split()

    def __init__(self, mass, pressure, phi):
        self.mass = mass
        self.pressure = pressure
        self.phi = phi

    def to_array(self):
        return numpy.array([self.mass, self.pressure, self.phi])

    @staticmethod
    def from_array(arr: numpy.ndarray):
        return State(mass=arr[0], pressure=arr[1], phi=arr[2])


Solution = collections.namedtuple('Solution', 'radius mass pressure phi surface_radius interior_mass rest_mass')


def d_state_d_rad(radius: float, state: State, density: float):
    dm = d_mass_d_rad(radius, density)
    dpress = d_pressure_d_rad(radius, density, state.mass, state.pressure)
    dphi = d_phi_d_rad(radius, density, state.mass, state.pressure)
    return numpy.array([dm, dpress, dphi])


def integrate_manual(density_rest: float, d_rad: float = 0.01, poly_index: float = 1.0, poly_gas_const: float = 1.0, rad_max: float = 1.0):
    radius = 0.0
    pressure_cent = polytropic_pressure(density_rest, index=poly_index, gas_const=poly_gas_const)
    state = State(mass=0.0, pressure=pressure_cent, phi=0.0)

    radii = []
    masses = []
    pressures = []
    phis = []

    def save_state(radius: float, state: State):
        radii.append(radius)
        masses.append(state.mass)
        pressures.append(state.pressure)
        phis.append(state.phi)

    # initial state and first step to avoid r=0 singularity
    save_state(radius, state)
    radius += d_rad
    save_state(radius, state)

    while True:
        density = polytropic_density(pressure=state.pressure, index=poly_index, gas_const=poly_gas_const)
        deriv_arr = d_state_d_rad(radius, state, density=density)
        state = State.from_array(state.to_array() + deriv_arr * d_rad)
        save_state(radius, state)

        radius += d_rad

        if state.pressure < 0 or radius > rad_max:
            break

    raw_radii, raw_result = numpy.array(radii), numpy.array(list(zip(masses, pressures, phis)))
    return post_process(raw_radii, raw_result, d_rad, poly_index, poly_gas_const, solver=integrate_manual)


def integrate_scipy(density_rest: float, d_rad: float = 0.01, poly_index: float = 1.0, poly_gas_const: float = 1.0, rad_max: float = 1.0):
    radii = numpy.arange(0.0 + d_rad, rad_max + d_rad, d_rad)
    pressure_cent = polytropic_pressure(density_rest, index=poly_index, gas_const=poly_gas_const)
    state = State(mass=0.0, pressure=pressure_cent, phi=0.0).to_array()

    def integrand(radius, state_arr):
        state = State.from_array(state_arr)
        density = polytropic_density(pressure=state.pressure, index=poly_index, gas_const=poly_gas_const)
        dy = d_state_d_rad(radius=radius, state=state, density=density)
        return dy

    res = integrate.odeint(integrand, state, t=radii, tfirst=True)

    return post_process(radii, res, d_rad, poly_index, poly_gas_const, solver=integrate_scipy)


def integrate_rk(density_rest: float, d_rad: float = 0.01, poly_index: float = 1.0, poly_gas_const: float = 1.0,
                 rad_max: float = 1.0, tol=1e-2, max_iters=5000):
    pressure_cent = polytropic_pressure(density_rest, index=poly_index, gas_const=poly_gas_const)
    state = State(mass=0.0, pressure=pressure_cent, phi=0.0).to_array()

    def integrand(radius, state_arr):
        state = State.from_array(state_arr)
        density = polytropic_density(pressure=state.pressure, index=poly_index, gas_const=poly_gas_const)
        dy = d_state_d_rad(radius=radius, state=state, density=density)
        return dy

    radii, res = rk.runge_kutta(tableau=rk.CommonTableau.DormandPrince, func=integrand,
                                y0=state, t_max=rad_max, dt=d_rad, tol=tol, t0=d_rad, max_iters=max_iters)

    radii = numpy.array(radii)
    res = numpy.array(res)

    return post_process(radii, res, d_rad, poly_index, poly_gas_const, solver=integrate_scipy)


def post_process(radii: numpy.array, raw_result: numpy.ndarray, d_rad: float = 0.01, poly_index: float = 1.0, poly_gas_const: float = 1.0,
                 solver: types.FunctionType = integrate_manual) -> Solution:
    # trim at first NaN pressure
    if solver == integrate_manual:
        idx_first_neg = numpy.where(raw_result[:, 1] < 0)[0][0]

        radii = radii[:idx_first_neg + 1]
        result = raw_result[:idx_first_neg + 1, :]

        masses, pressures, phis = result[:, 0], result[:, 1], result[:, 2]

        # Use a linear approximation to find the location of the surface
        factor = pressures[-2] / (pressures[-1] - pressures[-2])
        surface_radius = radii[-2] * (1 - factor) + radii[-1] * factor
        interior_mass = masses[-2] * (1 - factor) + masses[-1] * factor

        radii, masses, pressures, phis = radii[:-1], masses[:-1], pressures[:-1], phis[:-1]

    elif solver == integrate_scipy:

        idx_first_nan = numpy.where(numpy.isnan(raw_result[:, 1]))[0][0]

        radii = radii[:idx_first_nan - 1]
        result = raw_result[:idx_first_nan - 1, :]

        masses, pressures, phis = result[:, 0], result[:, 1], result[:, 2]
        surface_radius = radii[-1]
        interior_mass = masses[-1]
    else:
        raise ValueError('Unknown solver')

    # trim
    trim_idx = 1

    # Compute baryonic mass
    rho0 = polytropic_density_rest(pressure=pressures[trim_idx:], index=poly_index, gas_const=poly_gas_const)
    _radii = radii[trim_idx:]
    _masses = masses[trim_idx:]
    vol = 4 * numpy.pi * _radii ** 2 / numpy.sqrt(1 - 2 * _masses / _radii)
    d_rad_vec = radii[1:] - radii[:-1]
    rest_mass = numpy.sum(rho0 * vol * d_rad_vec)

    return Solution(radius=radii, mass=masses, pressure=pressures, phi=phis, surface_radius=surface_radius, interior_mass=interior_mass, rest_mass=rest_mass)


class SolutionSequence:
    def __init__(self, densities_rest: numpy.ndarray, solutions: typing.Tuple[Solution]):
        self.densities_rest = densities_rest
        self.solutions = solutions

    def _slice_attr_(self, attr: str):
        return numpy.array([getattr(solution, attr) for solution in self.solutions])

    @property
    def rest_mass(self):
        return self._slice_attr_('rest_mass')

    @property
    def interior_mass(self):
        return self._slice_attr_('interior_mass')

    @property
    def surface_radius(self):
        return self._slice_attr_('surface_radius')

    @staticmethod
    def from_densities(densities_rest: numpy.ndarray, d_rad: float = 0.01, poly_index: float = 1.0, poly_gas_const: float = 1.0, rad_max: float = 1.0,
                       solver: types.FunctionType = integrate_manual):
        if isinstance(rad_max, list):
            solns = tuple(solver(density_rest, d_rad, poly_index, poly_gas_const, rad_max) for density_rest, rad_max in zip(densities_rest, rad_max))
        else:
            solns = tuple(solver(density_rest, d_rad, poly_index, poly_gas_const, rad_max) for density_rest in densities_rest)

        return SolutionSequence(densities_rest, solns)


def generate_plot(index: float, gas_const: float, d_rad: float, densities: numpy.ndarray, solver: types.FunctionType, rad_max: float):
    # opt = default_options()
    # eos = EOSPoly(opt["n"], opt["K"])
    # seq = TOVSequence(eos)
    # seq.generate(opt["rho0"], opt["dr"])

    seq = SolutionSequence.from_densities(densities, d_rad=d_rad, poly_index=index,
                                          poly_gas_const=gas_const, rad_max=rad_max, solver=solver)

    fig = pyplot.figure()
    ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])
    ax.plot(seq.densities_rest, seq.interior_mass, "k-", label=r"$M$")
    ax.plot(seq.densities_rest, seq.rest_mass, "k--", label=r"$M_0$")
    ax.legend(loc="best")
    ax.set_xlabel(r"$\rho_0$ [P.U.]")
    ax.set_ylabel(r"$M$ [P.U.]")
    # ax.set_ylim((0.0, 0.2))

    fig = pyplot.figure()
    ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])
    ax.plot(seq.surface_radius, seq.interior_mass, "k-")
    ax.set_xlabel(r"$R$ [P.U.]")
    ax.set_ylabel(r"$M$ [P.U.]")
    # ax.set_ylim((0.0, 0.2))
    # ax.set_xlim((0.4, 1.3))
    pyplot.show()


if __name__ == '__main__':
    densities = numpy.linspace(0.01, 1.5, 100)
    d_rad = 0.001
    rad_max = 2.0
    # solver = integrate_manual
    # solver = integrate_scipy

    tol = 1e-3
    Nmax = int(1e4)
    solver = functools.partial(integrate_rk, tol=tol, max_iters=Nmax)

    n = 1.0
    K = 1.0
    generate_plot(index=n, gas_const=K, d_rad=d_rad, densities=densities, solver=solver, rad_max=rad_max)
