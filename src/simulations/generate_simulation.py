import numpy as np

from src.features import box_params, gal_params, clustering, spectra, gridding
from src.simulations import generate_signal
from src.simulations import generate_foregrounds
from src.simulations import generate_noise

if __name__ == '__main__':
    outdir = 'simulations/simple/'

    n_samples = int(20)

    # catalogue input
    # box properties
    box = box_params.get_test_params()

    # galaxy priors
    gal_priors = gal_params.get_simple_priors(
        v_range=[box['velocity_grid'][0].value, box['velocity_grid'][-1].value],
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
        n_samples=n_samples,
    )

    # generate components
    foregrounds = generate_foregrounds.generate_foregrounds(
        box, gal_priors, locations)
    noise = generate_noise.generate_noise(
        box, gal_priors, locations)
    signal, catalogue = generate_signal.generate_signal(
        box, gal_priors, locations)

    # final cube
    total_sim = signal + foregrounds + noise
    fits.writeto(
        outdir + 'total_sim.fits',
        total_sim,
        header,
        clobber=True)
