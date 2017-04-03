import numpy as np


def Jy2K(S, theta, lam):
    """
    Convert Jansky/beam to Kelvin, taken from
    https://science.nrao.edu/facilities/vla/proposing/TBconv

    S: Flux in Jy/beam
    theta: FWHM of the telescope in radians
    lam: Wavelength of the observation in m
    """
    return 0.32e-3 * lam**2. / theta**2. * S


def K2Jy(T, theta, lam):
    """
    Inverse of Jy2K, check the corresponding docstring
    T: Brightness temperature in K
    theta: FWHM of the telescope in radians
    lam: Wavelength of the observation in m
    """

    return T / Jy2K(1., theta, lam)
