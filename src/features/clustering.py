from astropy.io import fits
import numpy as np

from src import utils as ut


def load_clustering(infile):

    # load catalogue
    cat = fits.getdata(infile)

    # extract positions and masses
    ras = cat['ra']
    decs = cat['dec']
    zs = cat['z']

    n_samples = zs.shape[0]

    # frequencies in MHz
    nus = ut.z2nu(zs).to(u.MHz).value

    # in km/s
    velos = nus.to(u.km / u.s, equivalencies=P.HI_VFRAME)

    locations = dict(
        ras=ras,
        decs=decs,
        nus=nus,
        velos=velos,
        n_samples=n_samples)


def get_test_clustering(ra_range, dec_range, nu_range, n_samples):

    # ras in degree
    ras = np.random.uniform(
        low=ra_range[0],
        high=ra_range[1],
        size=n_samples)

    # decs in degree
    decs = np.random.uniform(
        low=dec_range[0],
        high=dec_range[1],
        size=n_samples)

    # frequencies in MHz
    nus = np.random.uniform(
        low=nu_range[0],
        high=nu_range[1],
        size=n_samples)

    # in km/s
    velos = nus.to(u.km / u.s, equivalencies=P.HI_VFRAME)

    locations = dict(
        ras=ras,
        decs=decs,
        nus=nus,
        velos=velos,
        n_samples=n_samples)

    return locations
