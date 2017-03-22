import numpy as np
import healpy as hp
from astropy.io import fits

from src import this_project as P
from src.features import box_params, gal_params, clustering, gridding


def generate_noise(box, gal_priors, locations):

    shape = box['n_channels'], box['boxsize'], box['boxsize']

    noise_cube = np.random.normal(loc=0, scale=1., size=shape)

    return noise_cube


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

    noise_cube = generate_noise(
        box=box, gal_priors=gal_priors, locations=locations)

    # write to disk
    fits.writeto(
        outdir + 'noise.fits',
        noise_cube,
        box['header'],
        clobber=True)
