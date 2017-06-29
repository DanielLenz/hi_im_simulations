import numpy as np
from astropy import units as u
from src import this_project as P
from scipy.constants import c


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

    Input
    -----
    S : float or ndarray
        Flux in Jy km/s
    D : float
        Distance in Mpc
    Returns
    -------
    m_hi : HI mass in solar masses

    """
    m_hi = 2.36e5 * D**2 * S

    return m_hi


def HImass2flux(m_hi, D):
    """
    Converts HI masses into observed fluxes, depending on the distance

    Input
    -----
    M : float or ndarray
        HI mass in solar masses
    D : float
        Distance in Mpc
    Returns
    -------
    S : Flux in Jy km/s

    """
    S = m_hi / 2.36e5 / D**2

    return S


# redshift-frequency-velocity conversion
def z2nu(z):
    """
    Converts redshifts to HI frequencies

    Input
    -----
    z : float or ndarray
        Redshift

    Returns
    -------
    nu : float or ndarray, same shape as z
        Observed frequency of HI emission in MHz

    """

    nu = (P.HI_RESTFREQ / (1. + z)).to(u.MHz).value

    return nu

def nu2z(nu):
    """
    Converts HI frequencies to redshifts

    Input
    -----
    nu : float or ndarray
        Observed frequency of HI emission in MHz

    Returns
    -------
    z : float or ndarray, same shape as nu
        Redshift

    """
    z = (P.HI_RESTFREQ / nu.to(u.Hz)) - 1.

    return z

def vrad2z(v):
    """
    Converts radial velocities to redshifts

    Input
    -----
    vrad : float or ndarray
        Radial velocity in km/s

    Returns
    -------
    z : float or ndarray, same shape as vrad
        Redshift

    """

    v_over_z = v/c
    z = np.sqrt((1. + v_over_z) / (1. - v_over_z)) - 1.
    return z


def z2vrad(z):
    """
    Converts redshifts to radial velocities

    Input
    -----
    z : float or ndarray
        Redshift

    Returns
    -------
    vrad : float or ndarray, same shape as z
        Radial velocity in km/s

    """

    vrad = (c*z**2. + 2.*c*z)/(z**2. + 2.*z + 2.)

    return vrad
