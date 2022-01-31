"""Integrating the TOV equations"""
import numpy


def dmass_drad(rad, dens):
    """TOV equation 1"""
    return 4 * numpy.pi * rad ** 2 * dens

def dpress_drad(rad, dens, mass, press):
    return - ((dens * mass) / (rad ** 2) * (1 + press / dens) *
              (1 + (4 * numpy.pi * press * rad ** 3) / mass) /
              (1 - (2 * mass) / rad))

def dPhi_drad(rad, dens, mass, press):
    return - (1 / dens) * dpress_drad(rad, dens, mass, press) / (1 + press / dens)


def polytropic_eos():
    pass


