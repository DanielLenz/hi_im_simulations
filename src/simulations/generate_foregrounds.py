import numpy as np
import healpy as hp
from astropy.io import fits

from src import this_project as P
from src.features import box_params, gal_params, clustering, gridding


def apply_synchro_powerlaw(image, reffreq, freqs):
    def powerlaw(nu, nu0):
        return np.power((nu / nu0), -0.7)

    cube = np.array([image * powerlaw(nu, reffreq) for nu in freqs])

    return cube


def generate_synchro(box):
    # get synchrotron
    synchrotron_hpx = hp.read_map(
        P.PPJOIN(
            'data/external/synchrotron_foreground/synch_353p0_1024.fits'))

    # grid synchrotron
    gridded_image = gridding.grid_hpx(
        synchrotron_hpx,
        box['header'],
        box['beamsize'].value)

    # apply powerlaw
    synchro_cube = apply_synchro_powerlaw(
        gridded_image,
        reffreq=217.e9,
        freqs=box['nus'].value)

    return synchro_cube


def generate_foregrounds(box, gal_priors, locations):
    synchro_cube = generate_synchro(box)
    uncorr_pointsources = np.zeros_like(synchro_cube)
    corr_pointsources = np.zeros_like(synchro_cube)

    foregrounds = synchro_cube + uncorr_pointsources + corr_pointsources

    return foregrounds


if __name__ == '__main__':
    outdir = 'simulations/simple/'

    n_samples = int(200)

    # catalogue input
    # box properties
    box = box_params.get_test_params()

    # galaxy properties
    gal_priors = gal_params.get_simple_priors(
        v_range=[
            box['velocity_grid'][0].value, box['velocity_grid'][-1].value],
        n_samples=n_samples)

    # clustering of the galaxies
    locations = clustering.get_test_clustering(
        ra_range=[
            (box['ra_min'] - box['d_ra']).value,
            (box['ra_max'] + box['d_ra']).value],
        dec_range=[
            (box['dec_min'] - box['d_dec']).value,
            (box['dec_max'] + box['d_dec']).value],
        v_range=[
            (box['velocity_grid'][0]).value,
            (box['velocity_grid'][-1]).value],
        n_samples=n_samples)

    # sum up foregrounds
    foregrounds = generate_foregrounds(
        box=box,
        gal_priors=gal_priors,
        locations=locations)

    # write to disk
    fits.writeto(
        outdir + 'foregrounds.fits',
        foregrounds,
        box['header'],
        clobber=True)
