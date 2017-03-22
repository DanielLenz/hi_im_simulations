import numpy as np
from astropy.io import fits
import matplotlib.pyplot as pl
import cygrid


def create_dummy_cat(params, outname):
    spectra = np.random.random(size=(params['nsources'], params['boxdepth']))
    ras = np.random.uniform(
        low=params['ra_min'] - params['d_ra'],
        high=params['ra_max'] + params['d_ra'],
        size=params['nsources'])

    decs = np.random.uniform(
        low=params['dec_min'] - params['d_dec'],
        high=params['dec_max'] + params['d_dec'],
        size=params['nsources'])

    c1 = fits.Column(name='ra', format='D', array=ras)
    c2 = fits.Column(name='dec', format='D', array=decs)
    c3 = fits.Column(
        name='spectra',
        format='{bd}E'.format(bd=params['boxdepth']),
        array=spectra)

    coldefs = fits.ColDefs([c1, c2, c3])
    tbhdu = fits.BinTableHDU.from_columns(coldefs)
    tbhdu.writeto(outname, clobber=True)

    return 0


def grid_catalogue(params, inname, outname):
    # set up header
    header = fits.Header()

    header['NAXIS'] = 3

    header['NAXIS1'] = params['boxsize']
    header['NAXIS2'] = params['boxsize']
    header['NAXIS3'] = params['boxdepth']

    header['CDELT1'] = params['pixsize']
    header['CDELT2'] = params['pixsize']
    header['CDELT3'] = 1.

    header['CRPIX1'] = 0
    header['CRPIX2'] = 0
    header['CRPIX3'] = 0

    header['CRVAL1'] = params['ra_min']
    header['CRVAL2'] = params['dec_min']
    header['CRVAL3'] = 0.
    header['LATPOLE'] = 90.

    header['CTYPE1'] = 'RA---SFL'
    header['CTYPE2'] = 'DEC--SFL'
    header['CTYPE3'] = 'VRAD'

    # wcs = WCS(header)
    # set up the gridder and kernel
    gridder = cygrid.WcsGrid(header)

    kernelsize_fwhm = 3. * params['pixsize']
    kernelsize_sigma = kernelsize_fwhm / 2.355
    sphere_radius = 3. * kernelsize_sigma

    gridder.set_kernel(
        'gauss1d',
        (kernelsize_sigma,),
        sphere_radius,
        kernelsize_sigma / 2.
        )

    # open catalogue
    catalogue = fits.getdata(inname)
    # grid
    gridder.grid(catalogue['ra'], catalogue['dec'], catalogue['spectra'])

    datacube = gridder.get_datacube()
    fits.writeto(outname, datacube, header, clobber=True)

    return 0


def main():

    nsources = int(1.e6)
    pixsize = 10/60  # in degree
    boxsize = 256  # 256^2 on the sky
    boxdepth = 1024  # 1024 spectral channels
    ra_min = 10.  # in degree
    ra_max = ra_min + pixsize * boxsize  # in degree
    d_ra = 1.  # additional volume for the catalogue
    dec_min = 10.  # in degree
    dec_max = dec_min + pixsize * boxsize  # in degree
    d_dec = 1.  # additional volume for the catalogue

    # gather the parameters in a dictionary
    test_params = dict(
        nsources=nsources,
        pixsize=pixsize,
        boxsize=boxsize,
        boxdepth=boxdepth,
        ra_min=ra_min,
        ra_max=ra_max,
        d_ra=d_ra,
        dec_min=dec_min,
        dec_max=dec_max,
        d_dec=d_dec,
    )

    # create the catalogue
    create_dummy_cat(
        params=test_params,
        outname='data/interim/first_cat.fits')

    # put it on a regular grid
    grid_catalogue(
        params=test_params,
        inname='data/interim/first_cat.fits',
        outname='data/interim/first_cube.fits')


if __name__ == '__main__':
    main()
