import numpy as np
from astropy.io import fits
import healpy as hp

import cygrid


def grid_catalogue(ras, decs, spectra, header, box):
    # set up the gridder and kernel
    gridder = cygrid.WcsGrid(header)

    # set kernel
    kernelsize_fwhm = 3. * box['beamsize'].value
    kernelsize_sigma = kernelsize_fwhm / 2.355
    sphere_radius = 3. * kernelsize_sigma

    gridder.set_kernel(
        'gauss1d',
        (kernelsize_sigma,),
        sphere_radius,
        kernelsize_sigma / 2.)

    # grid
    gridder.grid(ras, decs, spectra)

    datacube = gridder.get_datacube()

    return datacube


def grid_hpx(healpix_data, header, beamsize):
    """
    healpix_data: 1D-array, valid nside
    header: fits header
    beamsize: FWHM in deg
    """

    nside = hp.get_nside(healpix_data)
    npix = hp.nside2npix(nside)

    if healpix_data.ndim == 1:
        h = header.copy()
        h['NAXIS3'] = 1
        d = healpix_data[:, None]
    else:
        h = header.copy()
        d = healpix_data
    # set up the gridder and kernel
    gridder = cygrid.WcsGrid(h)

    # set kernel
    kernelsize_fwhm = 3. * beamsize
    kernelsize_sigma = kernelsize_fwhm / 2.355
    sphere_radius = 3. * kernelsize_sigma

    gridder.set_kernel(
        'gauss1d',
        (kernelsize_sigma,),
        sphere_radius,
        kernelsize_sigma / 2.)

    # get hpx coordinates
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    lons = np.rad2deg(phi).astype(np.float64)
    lats = (90. - np.rad2deg(theta)).astype(np.float64)

    # grid
    gridder.grid(lons, lats, d)
    datacube = gridder.get_datacube()

    return np.squeeze(datacube)
