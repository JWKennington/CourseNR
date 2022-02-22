import functools
import types

import numpy
from plotly import graph_objects as go


class ButcherTableau:
    def __init__(self, a_matrix: numpy.ndarray, c_vector: numpy.ndarray, b_vector: numpy.ndarray, sizes: tuple):
        self.a = a_matrix
        self.b = b_vector
        self.c = c_vector
        self.order = sizes  # TODO infer from b
        self.multi_order = len(self.order) > 1
        self.max_order = max(self.order)


class CommonTableau:
    """class of constants for namespacing known tableaus"""
    DormandPrince = ButcherTableau(a_matrix=numpy.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    ]), c_vector=numpy.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]),
        b_vector=numpy.array([
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
            [5179 / 57600, 0, 7571 / 16695, 393 / 640, 92097 / 339200, 187 / 2100, 1 / 40],
        ]), sizes=(6, 7))


def estimate_slope_procedural(tableau: ButcherTableau, func: types.FunctionType, y_n: numpy.ndarray, t_n: float, dt: float):
    # Compute slopes
    k = numpy.zeros(shape=(tableau.max_order, y_n.size))

    for i in range(tableau.max_order):
        _k = func(t_n + dt * tableau.c[i],
                    # y_n + dt * numpy.sum(tableau.a[i, :] * k),
                    y_n + dt * numpy.matmul(k.T, tableau.a[i, :]),

                  )
        if not isinstance(_k, numpy.ndarray): # handle 1-D case
            _k = numpy.array([_k])
        k[i, :] = _k

    return k


def step_dormand_prince(tableau: ButcherTableau, y_n, t_n, dt, k, tol):
    y_s = y_n + dt * numpy.matmul(k[:6, :].T, tableau.b[0, :6])
    y_ss = y_n + dt * numpy.matmul(k[:7, :].T, tableau.b[1, :7])
    err = numpy.sum(numpy.abs(y_ss - y_s))

    if err > tol:
        y, t = y_n, t_n
        dt = dt / 2
        save = False
    else:
        y, t = y_s, t_n + dt
        save = True
        if err < tol / 2 ** 5:
            dt = 2 * dt

    return y, t, dt, save


def runge_kutta(tableau: ButcherTableau, func: types.FunctionType, y0: numpy.ndarray, t_max: float, dt: float,
                tol: float, t0: float = 0.0, slope_estimator: types.FunctionType = estimate_slope_procedural,
                stepper: types.FunctionType = step_dormand_prince, max_iters: int = 1000):
    y = y0
    t = t0
    n = 0

    ys = [y]
    ts = [t]

    while t <= t_max and n < max_iters:
        k = slope_estimator(tableau, func=func, y_n=y, t_n=t, dt=dt)

        y, t, dt, save = stepper(tableau=tableau, y_n=y, t_n=t, dt=dt, k=k, tol=tol)
        n += 1

        if save:
            ys.append(y)
            ts.append(t)

    return ts, ys


################################################################################
#                                                                              #
#                                SIMPLE EXAMPLE                                #
#                                                                              #
################################################################################

def simple_func(t, y, lam: float):
    return lam * y


def simple_rk_solve(y0, t0, dt, t_max, max_iters: int = 1000, lam: float = 1.0, tol: float = 1e-3):
    func = functools.partial(simple_func, lam=lam)  # make iterative func
    ts, ys = runge_kutta(tableau=CommonTableau.DormandPrince, func=func,
                         y0=numpy.array([y0]), t0=t0, tol=tol, max_iters=max_iters, t_max=t_max, dt=dt)
    return ts, ys


def plot(ts, ys):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=[y[0] for y in ys]))
    fig.update_layout(
        #     yaxis_range=[-5,5],
        #     xaxis_range=[0,3],
        width=800,
        height=600,
        #                   showlegend=True,
        title_text=r'$\text{Solution to Simple Problem using RK45 } \dot{y} = \lambda y$',
        title_x=0.5,
        xaxis_title=r'$t$',
        yaxis_title=r'$y(t)$')
    fig.show()
    # fig.write_image('simple-soln.pdf')


def main():
    lam = 1
    y0 = 1.0
    t0 = 0
    dt = 0.01
    t_max = 10.0
    tol = 1e-3
    max_iters = 1e4

    ts, ys = simple_rk_solve(y0, t0, dt, t_max, max_iters=int(max_iters), lam=lam, tol=tol)
    plot(ts, ys)


if __name__ == '__main__':
    main()
