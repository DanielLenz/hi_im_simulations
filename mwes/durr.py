import numpy as np
import matplotlib.pyplot as pl
from scipy.constants import c
from scipy import stats
from astropy import units as u


from lineprofile.model import LineModel


def get_params():
    nsources = int(1.e3)

    # spatial
    pixsize = 10./60.  # in degree
    boxsize = 256  # 256^2 on the sky
    ra_min = 10.  # in degree
    ra_max = ra_min + pixsize * boxsize  # in degree
    d_ra = 1.  # additional volume for the catalogue
    dec_min = 10.  # in degree
    dec_max = dec_min + pixsize * boxsize  # in degree
    d_dec = 1.  # additional volume for the catalogue

    # spectral
    z_start = 0.1  # redshift start
    bandwidth = 1. * u.MHz  # bandwidth in MHz
    nu_start = HI_RESTFREQ / (1. + z_start)
    n_channels = 512  # 512 spectral channels
    d_nu = 10. * u.kHz  # spacing in frequency

    params = dict(
        nsources=nsources,
        pixsize=pixsize,
        boxsize=boxsize,
        ra_min=ra_min,
        ra_max=ra_max,
        d_ra=d_ra,
        dec_min=dec_min,
        dec_max=dec_max,
        d_dec=d_dec,
        z_start=z_start,
        bandwidth=bandwidth,
        nu_start=nu_start,
        n_channels=n_channels,
        d_nu=d_nu,
    )

    return params


if __name__ == '__main__':
    params = get_params()

    # get frequencies
    nus = np.arange(
        params['nu_start'].to(u.Hz).value,
        (params['nu_start'] + params['n_channels'] * params['d_nu']).to(u.Hz).value,
        params['n_channels']) * u.Hz
    nus = nus[::-1]  # high to low freqs, meaning increasing redshift

    velocities = nus.to(u.km/u.s, equivalencies=HI_VFRAME)

    # galaxy parameters
    gal_params = get_gal_params(velocities, n_samples=10)

    i = 1
    p = {k: v[i] for k, v in iter(gal_params.items())}
    parameters = np.array([
        np.log10(30.0),  # 30. Jy.km/s integrated flux density
        np.mean(velocities.to(u.km/u.s).value),  # Center the profile in data
        230.0,  # 230.0 km/s rotational broadening
        15.0,  # 15.0 km/s turbulent broadening
        0.2,  # 20 % solid body rotation
        -0.1,  # Slight asymmetry to lower velocities
    ])

    model = LineModel(
        velocities.to(u.km/u.s).value, n_disks=1, n_baseline=0)

    data = model.model(parameters)

    pl.plot(velocities, data)
    pl.show()


class Spectrum(object):

    _model = None
    _spectrum = None
    _shape_parameters = None

    def __init__(
            self,
            lon, lat, velos,
            velo_center, flux, v_rot, turb_velo, solid_rot, skewness):

        self.velos = velos
        self.flux = flux
        self.logflux = np.log10(self.flux)
        self.v_rot = v_rot
        self.turb_velo = turb_velo
        self.solid_rot = solid_rot
        self.skewness = skewness

    @property
    def model(self):
        if self._model is None:
            self._model = LineModel(self.velos, n_disks=1, n_baseline=0)
        return self._model

    @property
    def shape_parameters(self):
        """
        log10 of the flux in Jy.km/s
        apparent rotational velocity in km/s
        turbulent broadening in km/s
        fraction of the solid body rotation, between 0 and 1
        skewness, between -1 and +1
        """
        if self._shape_parameters is None:
            self._shape_parameters = np.array([
                self.logflux,
                self.v_rot,
                self.turb_velo,
                self.solid_rot,
                self.skewness
            ])
        return self._shape_parameters

    @property
    def spectrum(self):
        if self._spectrum is None:
            self._spectrum = self.model.model(self.shape_parameters)
        return self._spectrum
