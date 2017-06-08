from astropy import units as u
from astropy.io import fits
import numpy as np

from src import this_project as P


def create_header(box):
    header = fits.Header()

    header['NAXIS'] = 3

    header['NAXIS1'] = box['boxsize']
    header['NAXIS2'] = box['boxsize']
    header['NAXIS3'] = box['n_channels']

    header['CDELT1'] = box['pixsize'].value
    header['CDELT2'] = box['pixsize'].value
    header['CDELT3'] = -box['delta_nu'].value

    header['CRPIX1'] = 0
    header['CRPIX2'] = 0
    header['CRPIX3'] = 0

    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['CUNIT3'] = 'Hz'

    header['CRVAL1'] = box['ra_min'].value
    header['CRVAL2'] = box['dec_min'].value
    header['CRVAL3'] = box['nu_grid'][0].value
    header['LATPOLE'] = 90.

    header['CTYPE1'] = 'RA---SFL'
    header['CTYPE2'] = 'DEC--SFL'
    header['CTYPE3'] = 'FREQ'

    return header


def parkes_type():
    ...


def get_test_params():

    # spatial
    pixsize = (10. / 60.) * u.deg  # in degree
    beamsize = (20. / 60.) * u.deg  # in degree
    boxsize = 256  # 256^2 on the sky
    ra_min = 10. * u.deg  # in degree
    ra_max = ra_min + pixsize * boxsize  # in degree
    d_ra = 1. * u.deg  # additional volume for the catalogue
    dec_min = 10. * u.deg  # in degree
    dec_max = dec_min + pixsize * boxsize  # in degree
    d_dec = 1. * u.deg  # additional volume for the catalogue

    # spectral
    z_start = 0.01  # redshift start
    bandwidth = (100 * u.MHz).to(u.Hz)  # bandwidth in MHz
    nu_start = P.HI_RESTFREQ / (1. + z_start)
    n_channels = 512  # 512 spectral channels
    # delta_nu = 10. * u.kHz  # spacing in frequency
    delta_nu = bandwidth / n_channels  # spacing in frequency

    # get frequencies, high to low freqs, meaning increasing redshift
    nu_grid = np.linspace(
        nu_start.to(u.Hz).value,
        (nu_start - n_channels * delta_nu).to(u.Hz).value,
        n_channels) * u.Hz

    # convert to v_lsr in km/s
    velocity_grid = nu_grid.to(u.km / u.s, equivalencies=P.HI_VFRAME)
    delta_v = velocity_grid[1] - velocity_grid[0]

    params = dict(
        pixsize=pixsize,
        boxsize=boxsize,
        beamsize=beamsize,
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
        delta_nu=delta_nu,
        delta_v=delta_v,
        nu_grid=nu_grid,
        velocity_grid=velocity_grid
    )

    params['header'] = create_header(params)

    return params
