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


def flux2HImass(S, D):
    """
    Converts observed fluxes into HI masses, depending on the distance

    Inputs
    ------
    S : float or ndarray
        Flux in Jy km/s
    D : float
        Distance in Mpc
    Returns
    -------
    HI mass in solar masses

    """
    M_HI = 2.36e5 * D**2 * S

    return M_HI


def HImass2flux(M, D):
    """
    Converts HI masses into observed fluxes, depending on the distance
    Inputs
    ------
    M : float or ndarray
        HI mass in solar masses
    D : float
        Distance in Mpc
    Returns
    -------
    Flux in Jy km/s

    """
    S = M_HI / 2.36e5 / D**2

    return S


# redshift-frequency conversion
def z2nu(z):
    return P.HI_RESTFREQ / (1. + z)


def nu2z(nu):
    return (P.HI_RESTFREQ / nu.to(u.Hz)) - 1.
