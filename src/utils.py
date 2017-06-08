import numpy as np
from astropy import units as u
from src import this_project as P


def Jy2K(S, theta, lam):
    """
    Convert Jansky/beam to Kelvin, taken from
    https://science.nrao.edu/facilities/vla/proposing/TBconv

    S: Flux in Jy/beam
    theta: FWHM of the telescope in radians
    lam: Wavelength of the observation in m

    Returns: Brightness temperature in K
    """
    return 0.32e-3 * lam**2. / theta**2. * S


def K2Jy(T, theta, lam):
    """
    Inverse of Jy2K, check the corresponding docstring
    T: Brightness temperature in K
    theta: FWHM of the telescope in radians
    lam: Wavelength of the observation in m

    Return: Flux/beam in Jy
    """

    return T / Jy2K(1., theta, lam)


def HImass2flux(S, D):
    """
    S: Flux in Jy km/s
    D: Distance in Mpc
    returns: HI mass in solar masses
    """
    return 2.36e5 * D**2 * S


# redshift-frequency conversion
def z2nu(z):
    return P.HI_RESTFREQ / (1. + z)


def nu2z(nu):
    return (P.HI_RESTFREQ / nu.to(u.Hz)) - 1.
