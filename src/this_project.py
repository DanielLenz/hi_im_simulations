from astropy import units as u
import os

# paths
PROJECTPATH = '/users/dlenz/projects/hi_im_sims/'
BASEPATH = '/users/dlenz/projects/'
PPJOIN = lambda p: os.path.join(PROJECTPATH, p)
BPJOIN = lambda p: os.path.join(BASEPATH, p)

# HI restframe
HI_RESTFREQ = 1420.4058 * u.MHz
HI_VFRAME = u.doppler_radio(HI_RESTFREQ)
