"""Advection solver code from LeVeque 10.8"""
import functools
import typing

import numpy
from plotly import graph_objects as go, subplots


def boundary_condition_identity(x) -> numpy.ndarray:
    return x


def boundary_condition_constant(x, const: float) -> numpy.ndarray:
    y = x.copy()
    y[0] = y[-1] = const
    return y


def step(const_2_step: float, const_3_step: float, u_n: numpy.ndarray, boundary_condition: typing.Callable) -> numpy.ndarray:
    """Generalized stepping function in terms of linear combination of two distinct differencing terms

    Coefficient-primary formulation:

        u_new = c_2 (u_{i+1} - u_{i-1}) + c_3 (u_{i+1} - 2 u_i + u_{i+1})

    Index-primary formulation:

        u_new = (c_2 + c_3) u_{i+1} + (-2 c_3) u_i + (-c_2 + c_3) u_{i-1}

    Args:
        const_2_step:
            float, the scale factor for the two-index differencing term
        const_3_step:
            float, the scale factor for the three-index differencing term
        u_n:
            numpy.ndarray, the current state
        boundary_condition:
            Callable, a callable function that returns a copy of the state with boundary conditions set, accepts
            only a single argument (the state)

    Returns:
        numpy.ndarray, the evolved state vector
    """
    u_np1 = u_n.copy()

    # Compute both terms
    term_2_step = u_n[2:] - u_n[:-2]
    term_3_step = u_n[2:] - 2.0 * u_n[1:-1] + u_n[0:-2]

    # Set new step values based on two terms
    u_np1[1:-1] = u_n[1:-1] + const_2_step * term_2_step + const_3_step * term_3_step

    # Apply boundary condition and return
    return boundary_condition(u_np1)


def step_upwind(u_n: numpy.ndarray, dt: float, dx: float, a: float) -> numpy.ndarray:
    """Helper function around general stepper for case of Upwind method"""
    const = a * dt / dx
    const_3_step = const / 2
    const_2_step = -const_3_step
    return step(u_n=u_n, const_2_step=const_2_step, const_3_step=const_3_step,
                boundary_condition=functools.partial(boundary_condition_constant, const=0))


def step_lax_wendroff(u_n: numpy.ndarray, dt: float, dx: float, a: float) -> numpy.ndarray:
    """Helper function around general stepper for case of Lax-Wendroff method"""
    const = a / 2 * dt / dx
    const_2_step = -const
    const_3_step = 2 * const ** 2
    return step(u_n=u_n, const_2_step=const_2_step, const_3_step=const_3_step,
                boundary_condition=functools.partial(boundary_condition_constant, const=0))


def solve(a: float, x: numpy.ndarray, t_init: float, t_max: float, dt: float, u_init: numpy.ndarray, stepper: typing.Callable, slice_freq: int = 1):
    # Spacing constants
    dx = x[1] - x[0]

    # Initial state
    step = 0
    t = t_init
    u = u_init

    # state Memory
    ts = [t]
    us = [u]

    while t <= t_max:
        # Evolve with stepper
        step += 1
        t += dt
        u = stepper(u_n=u, dt=dt, dx=dx, a=a)

        # Optionally save out results
        if step % slice_freq == 0:
            ts.append(t)
            us.append(u)

    return ts, us


def soln_init(x):
    return soln_analytic(t=0, x=x, v=1)


def soln_analytic(t, x, v):
    return numpy.exp(-20.0 * ((x - v * t) - 2) ** 2) + numpy.exp(-((x - v * t) - 5) ** 2)


def plot(x, ts_up, us_up, ts_lw, us_lw, idxs: typing.Tuple[int, int], dt, dx):
    t1 = ts_up[idxs[0]]
    t2 = ts_up[idxs[1]]
    # scale analytic solution ? speed argument
    v_0 = dx / dt
    s = 0.793
    u_an_1 = soln_analytic(t=t1, x=x, v=s * v_0)
    u_an_2 = soln_analytic(t=t2, x=x, v=s * v_0)
    u_up_1 = us_up[idxs[0]]
    u_up_2 = us_up[idxs[1]]
    u_lw_1 = us_lw[idxs[0]]
    u_lw_2 = us_lw[idxs[1]]

    fig = subplots.make_subplots(rows=1, cols=2,
                                 subplot_titles=('$\mathrm{{Solution\\ at\\ }} t = {:.1e}$'.format(t1),
                                                 '$\mathrm{{Solution\\ at\\ }} t = {:.1e}$'.format(t2)))

    fig.add_trace(go.Scatter(x=x, y=u_an_1, mode='lines', line=dict(color='black', dash='solid'), name='Analytic'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=u_up_1, mode='lines', line=dict(color='red', dash='dash'), name='Upwind'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=u_lw_1, mode='lines', line=dict(color='blue', dash='dot'), name='Lax-Wendroff'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=u_an_2, mode='lines', line=dict(color='black', dash='solid')  # , name='Analytic'
                             , showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=u_up_2, mode='lines', line=dict(color='red', dash='dash')  # , name='Upwind'
                             , showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=u_lw_2, mode='lines', line=dict(color='blue', dash='dot')  # , name='Lax-Wendroff'
                             , showlegend=False),
                  row=1, col=2)

    fig.update_layout(
        #     yaxis_range=[0,10],
        #                   xaxis_range=[0,10],
        width=1200,
        height=700,
        showlegend=True,
        title_text=r'$\text{Solution of Advection Equation Solution}$',
        title_x=0.5,
        xaxis_title=r'$x \in (0, 25)$',
        yaxis_title=r'$u(x)$')
    # fig.show()
    fig.write_image('problem-2-solution.pdf')


def main():
    a = 1.0
    dx = 0.05
    x = numpy.arange(dx, 25.0, dx)
    dt = 0.8 * dx
    t_max = 17.0
    t_init = dt
    u_init = soln_init(x)

    ts_up, us_up = solve(a=a, x=x, t_init=t_init, t_max=t_max, dt=dt, u_init=u_init, stepper=step_upwind)
    ts_lw, us_lw = solve(a=a, x=x, t_init=t_init, t_max=t_max, dt=dt, u_init=u_init, stepper=step_lax_wendroff)

    idxs = (2, -2)

    plot(x, ts_up, us_up, ts_lw, us_lw, idxs=idxs, dt=dt, dx=dx)


if __name__ == '__main__':
    main()
