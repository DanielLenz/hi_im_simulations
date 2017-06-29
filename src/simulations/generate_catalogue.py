"""
Use the existing Bethermin+ (2017) catalogue to create a catalogue for
the HI intensity mapping simulations. This requires selecting specific
redshift and halo masses, as well as the computation of HI masses.
"""

import pandas as pd
import numpy as np
from astropy.io import fits

from src import this_project as P


def create_dataframe():
    with fits.open(
        P.PPJOIN('data/external/Mock_cat_Bethermin2017.fits')) as hdu:
        df = pd.DataFrame(hdu[2].data)
    
from astropy.table import Table
dat = Table.read('datafile', format='fits')
df = dat.to_pandas()
