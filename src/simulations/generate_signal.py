import numpy as np
from scipy import stats
from astropy.io import fits

from src.features import box_params, gal_params, clustering, spectra, gridding


def create_FITStable(ras, decs, specs):
    """
    ras, decs 1D array
    spectra with shape (ras.shape, n_channels)
    """

    c1 = fits.Column(name='ra', format='D', array=ras)
    c2 = fits.Column(name='dec', format='D', array=decs)
    c3 = fits.Column(
        name='spectra',
        format='{}E'.format(int(specs.shape[1])),
        array=specs)

    coldefs = fits.ColDefs([c1, c2, c3])
    tbhdu = fits.BinTableHDU.from_columns(coldefs)

    return tbhdu


def generate_catalogue(
        box, gal_priors, locations):

    # generate spectra with the HI profile code
    specs = spectra.generate_spectra(
        gal_priors,
        ra=locations['ras'],
        dec=locations['decs'],
        velocity_grid=box['velocity_grid'].value,
        v_rad=locations['velos'])

    catalogue = create_FITStable(
        ras=locations['ras'],
        decs=locations['decs'],
        specs=specs)

    return specs, catalogue


def generate_signal(box, gal_priors, locations):
    # catalogue
    specs, catalogue = generate_catalogue(
        box=box,
        gal_priors=gal_priors,
        locations=locations)

    # signal, foreground, and instrumental effects
    signal = gridding.grid_catalogue(
        ras=locations['ras'],
        decs=locations['decs'],
        spectra=specs,
        header=box['header'],
        box=box)
    signal = np.nan_to_num(signal)

    return signal, catalogue


if __name__ == '__main__':
    outdir = 'simulations/simple/'

    n_samples = int(20)

    # catalogue input
    # box properties
    box = box_params.get_test_params()

    # galaxy priors
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

    signal, catalogue = generate_signal(
        box=box, gal_priors=gal_priors, locations=locations)

    # write to disk
    fits.writeto(
        outdir + 'signal.fits',
        signal,
        box['header'],
        clobber=True)

    # write to disk
    catalogue.writeto(outdir + 'catalogue.fits', clobber=True)
