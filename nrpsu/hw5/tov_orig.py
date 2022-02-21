#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tolman-Oppenheimer-Volkoff
This solves the TOV equations using a simple FD approach
"""
from math import pi
import matplotlib.pyplot as plt
import numpy as np


class EOSPoly:
    """
    A class representing a polytropic equation of state
    """

    def __init__(self, n=1.0, K=1.0):
        self.n = n
        self.K = K

    def press_from_rho0(self, rho0):
        return self.K * rho0 ** (1.0 + 1.0 / self.n)

    def rho_from_press(self, press):
        return self.n * press + self.rho0_from_press(press)

    def rho0_from_press(self, press):
        return (press / self.K) ** (self.n / (self.n + 1))


class TOV:
    """
    A class representing a TOV
    """

    def __init__(self, rho0c, eos):
        self.rho0c = rho0c
        self.eos = eos

    def rhs(self, m, P, r):
        rho = self.eos.rho_from_press(P)
        dm = 4 * pi * r * r * rho
        dP = -rho / (r * r) * (1 + P / rho) * (m + 4 * pi * P * r * r * r) / (1 - 2 * m / r)
        return dm, dP

    def solve(self, dr=0.01):
        """
        Solve the TOV equations and store the results
        """
        # Central values
        m = [0.0]
        P = [self.eos.press_from_rho0(self.rho0c)]
        r = [0.0]
        # At r = 0, dm = dP = 0
        m.append(m[0])
        P.append(P[0])
        r.append(dr)
        # Step in radius until we reach the surface
        while True:
            mm, mP, mr = m[-1], P[-1], r[-1]
            dm, dP = self.rhs(mm, mP, mr)
            m.append(m[-1] + dr * dm)
            P.append(P[-1] + dr * dP)
            r.append(r[-1] + dr)
            # If the pressure is negative we have found the last point
            if P[-1] < 0:
                break
        # Use a linear approximation to find the location of the surface
        xi = P[-2] / (P[-1] - P[-2])
        self.R = r[-2] * (1 - xi) + r[-1] * xi
        self.M = m[-2] * (1 - xi) + m[-1] * xi
        # Store the solution as numpy arrays
        self.m = np.array(m[:-1])
        self.P = np.array(P[:-1])
        self.r = np.array(r[:-1])
        # Compute baryonic mass
        rho0 = self.eos.rho0_from_press(self.P[1:])
        r = self.r[1:]
        m = self.m[1:]
        vol = 4 * pi * r * r / np.sqrt(1 - 2 * m / r)
        self.M0 = np.sum(rho0 * vol * dr)


class TOVSequence:
    """
    A class representing a sequence of TOV
    """

    def __init__(self, eos):
        self.eos = eos

    def generate(self, rho0, dr=0.01):
        M = []
        M0 = []
        R = []
        for r in rho0:
            tov = TOV(r, self.eos)
            tov.solve(dr)
            M.append(tov.M)
            M0.append(tov.M0)
            R.append(tov.R)
        self.rho0 = np.array(rho0)
        self.M = np.array(M)
        self.M0 = np.array(M0)
        self.R = np.array(R)


def default_options():
    return {
        "K": 1.0,
        "n": 1.0,
        "dr": 0.001,
        "rho0": np.linspace(0.01, 1.5, 100)
    }


def __main__():
    opt = default_options()
    eos = EOSPoly(opt["n"], opt["K"])
    seq = TOVSequence(eos)
    seq.generate(opt["rho0"], opt["dr"])
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])
    ax.plot(seq.rho0, seq.M, "k-", label=r"$M$")
    ax.plot(seq.rho0, seq.M0, "k--", label=r"$M_0$")
    ax.legend(loc="best")
    ax.set_xlabel(r"$\rho_0$ [P.U.]")
    ax.set_ylabel(r"$M$ [P.U.]")
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])
    ax.plot(seq.R, seq.M, "k-")
    ax.set_xlabel(r"$R$ [P.U.]")
    ax.set_ylabel(r"$M$ [P.U.]")
    plt.show()


# Run the main function if executed as a script
if __name__ == '__main__':
    fig = __main__()
    plt.show()
