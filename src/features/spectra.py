import numpy as np
import matplotlib.pyplot as pl
from scipy.constants import c
from scipy import stats
from astropy import units as u

import cygrid as cg

from lineprofile.model import LineModel


def generate_spectra(gal_params, ra, dec, velocity_grid, v_rad):
    n_samples = len(ra)
    # print(gal_params)
    # print(ra, dec, velocity_grid, v_rad)
    # return
    specs = []
    for i in range(n_samples):
        spectrum = Spectrum(
            lon=ra[i],
            lat=dec[i],
            velos=velocity_grid,
            velo_center=v_rad[i],
            flux=gal_params['flux'][i],
            v_rot=gal_params['v_rot'][i],
            turb_velo=gal_params['turb_velo'][i],
            solid_rot=gal_params['solid_rot'][i],
            skewness=gal_params['skewness'][i])
        specs.append(spectrum.spectrum)

    return np.array(specs)


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
