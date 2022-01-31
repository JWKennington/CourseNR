"""This script uses the analytical solutions for the Oppenheimer-Snyder
model of spherical dust collapse to plot geodesics """

import numpy
import plotly.graph_objects as go


def star_rad(conf_time: float, rad_0: float) -> float:
    """Surface of star radial coordinate in conformal time"""
    return 0.5 * rad_0 * (1 + numpy.cos(conf_time))


def star_time(conf_time: float, rad_0: float, mass: float = 1) -> float:
    """Surface of star time coordinate in conformal time"""
    return numpy.sqrt((rad_0 ** 3) / (8 * mass)) * (conf_time + numpy.sin(conf_time))


def plot(rads, times):
    fig = go.Figure()

    for r_0, r, t in zip(r_0s, rads, times):
        fig.add_trace(go.Scatter(x=r, y=t, mode='lines', name='{:.2f}'.format(r_0)))

    fig.update_layout(
        yaxis_range=[0, 25], xaxis_range=[0,6],
        width=600, height=600,
        title_text=r'$\text{Oppenheimer Snyder }(\tau, R)$',
        title_x=0.5,
        xaxis_title=r'$R / M$',
        yaxis_title=r'$\tau/M$')
    fig.show()


def main():
    conf_times = numpy.arange(0, numpy.pi, 0.01)
    mass_fracs = [0.1, 0.25, 0.5, 1.0]
    r_full = 5.0
    r_0s = numpy.array(mass_fracs) ** (1 / 3) * r_full

    rads = []
    times = []

    for r_0, mass_frac in zip(r_0s, mass_fracs):
        rads.append(star_rad(conf_times, r_0))
        times.append(star_time(conf_times, r_0, mass=mass_frac))


if __name__ == '__main__':
    main()
