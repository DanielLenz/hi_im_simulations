#!/usr/bin/python
"""
Generate a (Poisson-distributed) set of galaxies in a given volume, and assign 
HI masses to them.
"""
import numpy as np
import pylab as P
import scipy.interpolate
import scipy.integrate
from halomodel import HaloModel

L_HI = 6.27e-9 # Conversion factor, M_HI (M_sun) -> L_HI (L_sun)

np.random.seed(10)

# Survey volume parameters
L = 50. # Box size, Mpc
z_obs = 100. # Distance from observer to survey volume

# Initialise halo model
hm = HaloModel(pkfile="camb_pk_z0.dat")


def binned_mass_function(hm, mhbins):
    """
    Return halo mass function in a given set of halo mass bins.
    """
    # Calculate halo mass function
    mhvals = np.logspace(8., 16., 500)
    _dndlogm = hm.dndlogm(mhvals, z=0.) # Mpc^-3
    
    # Calculate cumulative n(>m) by integrating backwards, then interpolate
    _ncum = scipy.integrate.cumtrapz(_dndlogm[::-1], -np.log(mhvals)[::-1], 
                                     initial=0.)[::-1]
    ncum = scipy.interpolate.interp1d(np.log(mhvals), _ncum, kind='linear', 
                                      bounds_error=False)
    
    # Calculate binned halo number density (difference of cumulative mass fn.)
    nbins = ncum(np.log(mhbins[:-1])) - ncum(np.log(mhbins[1:]))
    
    # Sanity check on negative values
    nbins[np.where(nbins < 0.)] = np.nan
    return nbins

def realise_halos(mhbins, nbins, L):
    """
    Realise galaxy catalogue and per-galaxy parameters starting from a binned 
    halo mass function.
    
    Halo masses within each bin are assigned according to a log-uniform 
    distribution. Positions are unfirom random.
    """
    vol = L**3.
    
    # Realise a set of halos in each mass bin
    mh = []
    Ntot = 0
    for i in range(nbins.size):
        
        # Poisson draw of expected number of halos
        Nmean = vol * nbins[i]
        N = np.random.poisson(lam=Nmean)
        Ntot += N
        
        # Realise this no. of halos with log-uniform masses within mass interval
        _mh = np.random.uniform(low=np.log(mhbins[i]), 
                                high=np.log(mhbins[i+1]),
                                size=N)
        mh.append( np.exp(_mh) )    
    mh = np.concatenate(mh)
    return mh

def realise_himass(mh):
    """
    Return HI mass for a set of halos, given their halo mass.
    """
    # Interpolate HI mass - halo mass relation
    mhvals = np.logspace(np.log10(np.min(mh)), np.log10(np.max(mh)), 500)
    _mhi = hm.MHI(mhvals, z=0.)
    mhi_interp = scipy.interpolate.interp1d(np.log(mhvals), np.log(_mhi), 
                                            kind='linear', bounds_error=False,
                                            fill_value=0.)
    
    # Apply interpolation to set of halos
    mhi = np.exp( mhi_interp(np.log(mh)) )
    
    # Sanity check
    mhi[np.where(np.isnan(mhi))] = 0.
    
    return mhi
    
def realise_positions(N, L):
    """
    Realise random positions of halos, using a uniform distribution.
    """
    # Realise positions according to uniform distribution
    x = np.random.uniform(low=-L/2., high=L/2., size=N)
    y = np.random.uniform(low=-L/2., high=L/2., size=N)
    z = np.random.uniform(low=-L/2., high=L/2., size=N)
    return x, y, z

def realise_redshifts(z, h=0.67):
    """
    Get the cosmological redshift, given z-coord, using the simple Hubble 
    relation for low-redshift.
    """
    # z_red = H_0 d / c
    return 100.*h / 3e5 * z


def voxel_grid(L, Nx, Ny, Nz):
    """
    Define grid of voxels that covers the entire volume. Returns voxel 
    centroids in each 3D direction.
    """
    # Get grid cell edges
    xedges = np.linspace(-L/2., L/2., Nx + 1)
    yedges = np.linspace(-L/2., L/2., Ny + 1)
    zedges = np.linspace(-L/2., L/2., Nz + 1)
    
    # Return cell centroids
    x0 = xedges[:-1] + 0.5 * (xedges[1] - xedges[0])
    y0 = yedges[:-1] + 0.5 * (yedges[1] - yedges[0])
    z0 = zedges[:-1] + 0.5 * (zedges[1] - zedges[0])
    return x0, y0, z0


def galaxy_profile(pxgrid, pos, w, fsep):
    """
    3D profile of galaxy. Simple non-rotated Gaussian in angle; double Gaussian 
    in redshift direction.
    """
    x, y, z = pxgrid # Pixel centres
    x0, y0, z0 = pos # Centroid position
    wx, wy, wz = w # Width parameters of profiles
    
    # Angular profile
    p_sky = np.exp(-0.5 * (((x - x0) / wx)**2. + ((y - y0)/wy)**2.)) \
          / (2.*np.pi * wx * wy)
    
    # Radial profile (double Gaussian)
    dz = fsep * wz
    p_freq = (np.exp(-0.5 * ((z - z0 - 0.5*dz) / wz)**2.) \
            + np.exp(-0.5 * ((z - z0 + 0.5*dz) / wz)**2.) ) \
            / (2. * wz * np.sqrt(2.*np.pi))
    return p_sky * p_freq
    

def generate_datacube():
    """
    Add galaxies to datacube.
    """
    # TODO



if __name__ == '__main__':

    # Calculate binned mass function
    #mhbins = np.logspace(8.5, 16., 200) # Msun
    #mhbins = np.logspace(10., 10.5, 200) # Msun
    #mhbins = np.logspace(10.5, 16., 200) # Msun
    mhbins = np.logspace(10., 10.2, 200) # Msun
    mhc = np.exp( 0.5 * (np.log(mhbins[1:]) + np.log(mhbins[:-1])) ) # Centroids
    nbins = binned_mass_function(hm, mhbins)

    # Realise halos across all bins of interest
    print "Realising halos..."
    mh = realise_halos(mhbins, nbins, L)
    print "\tGenerated %2.2e halos." % mh.size

    # Realise HI masses
    print "Realising galaxy properties..."
    mhi = realise_himass(mh)
    x, y, z = realise_positions(mh.shape, L)
    zred = realise_redshifts(z, h=0.67)


    # Pixel grid definition
    pix = voxel_grid(L, Nx=1, Ny=300, Nz=300)
    pxgrid = np.meshgrid(pix[0], pix[1], pix[2])

    wx = 5e-2
    wy = 5e-2
    wz = 20e-2 # FIXME: What is a reasonable radial distance (in Mpc)?
    # FIXME: x,y size should depend on HI mass?
    # FIXME: Rotation curve should have shape in x,y

    # FIXME: This vaguely simulates different orientations
    fsep = np.random.uniform(low=0., high=3.5, size=x.size)

    print "N = %1.1e" % x.size

    prof = 0
    for i in range(x.size):
        if i % 100 == 0: print "%d / %d" % (i, x.size)
        
        # flux = L / 4 pi dl^2
        
        f = galaxy_profile(pxgrid, (0., y[i], z[i]), (wx, wy, wz), fsep[i]) \
          * L_HI * mhi[i] / (4.*np.pi * (z[i] + z_obs)**2.)
          # FIXME: Units are L_sun/Mpc^2!
        prof += f
        # FIXME: Normalisation!

    np.save("imtest", prof[:,0,:])


    print np.min(prof), np.max(prof)
    print prof.shape
    print pxgrid[0].shape

    def force_aspect(ax, aspect=1):
        im = ax.get_images()
        extent =  im[0].get_extent()
        ax.set_aspect(np.abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


    ax = P.subplot(111)
    ax.matshow(prof[:,0,:], cmap='YlOrRd', vmin=0., vmax=0.005)
    #P.colorbar(ax)
    force_aspect(ax)
    #P.axes().set_aspect(0.5)
    P.show()

