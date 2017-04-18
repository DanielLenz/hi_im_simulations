#!/usr/bin/python
"""
Halo mass function and halo bias model.
"""
import numpy as np
import scipy.integrate

class HaloModel(object):
    
    def __init__(self, pkfile, om=0.3, h=0.7, gamma=0.55, ampfac=1.):
        """
        Initialise HaloModel class.
        """
        # Cosmo params
        self.om = om
        self.h = h
        self.gamma=gamma
        
        # Define critical density and Sheth-Tormen params (see 
        # Sheth & Tormen 1999)
        self.delta_c = 1.68647
        self.a = 0.707
        self.A = 0.322
        self.p = 0.3
        
        # Define Tinker et al. parameters (see their Table 2)
        self.A0 = 0.186
        self.a0 = 1.47
        self.b0 = 2.57
        self.c = 1.19
        
        # Load matter power spectrum, P(k) (assumed to be in non-h^-1 units)
        self.k, self.pk = np.genfromtxt(pkfile).T[:2]
        self.pk *= ampfac**2.
        
        # Convert to non-h^-1 units
        #self.k *= 1. / self.h
        #self.pk *= self.h**3.
        
        
    def fgrowth(self, z):
        """
        Generalised form for the growth rate.
        """
        Ez2 = self.om * (1. + z)**3. + (1. - self.om)
        Oma = self.om * (1. + z)**3. / Ez2
        return Oma**self.gamma

    def growth_fn(self, z):
        """
        Calculate growth function, D(z), at a given redshift. Normalised to 
        D=1 at z=0.
        """
        _z = np.linspace(0., z, 200)
        a = 1. / (1. + _z)
        _f = self.fgrowth(_z)
        _D = np.concatenate( ([0.,], scipy.integrate.cumtrapz(_f, np.log(a))) )
        _D = np.exp(_D)
        return _D[-1]

    def M_for_R(self, R, z=0.):
        """
        Mass contained within a sphere of radius R in the background.
        """
        rho = self.h*self.h*self.om * (1. + z)**3. * 2.776e11 # in Msun/Mpc^3
        return 4./3. * np.pi * R**3. * rho

    def R_for_M(self, M, z=0.):
        """
        Comoving radius as a function of mass (mass in M_sun).
        """
        rho = self.h*self.h*self.om * (1. + z)**3. * 2.776e11 # in Msun/Mpc^3
        return (3.*M / (4. * np.pi * rho))**(1./3.)
    
    def sigma_R(self, R, z=0.):
        """
        Calculate the linear rms fluctuation, sigma(R), as a function of tophat 
        smoothing scale, R.
        """
        # FIXME: Doesn't deal with redshift properly
        k = self.k
        pk = self.pk
        D = self.growth_fn(z)
        W = 3. * (np.sin(k*R) - k*R*np.cos(k*R)) / (k*R)**3.
        
        # Integrate over window function
        sig_r = scipy.integrate.simps(pk*(D*k*W)**2., k)
        sig_r /= (2. * np.pi**2.)
        return np.sqrt(sig_r)

    def dlogsigM_dlogM(self, M, sig):
        """
        Logarithmic derivative of sigma(M) with respect to M, i.e.
        d log(sigma(M)) / d log(M)
        """
        coeffs = np.polyfit(np.log(M), np.log(sig), deg=4)
        p = np.poly1d(coeffs)
        return p.deriv()(np.log(M))

    def bias(self, M, z=0.):
        """
        Calculate the halo bias, b(M, z), using Eq. 12 of Sheth & Tormen 1999.
        """
        delta_c = self.delta_c
        A = self.A
        a = self.a
        p = self.p
        
        # Calculate sigma(R) for this mass scale
        R = self.R_for_M(M)
        sigR = np.array([self.sigma_R(_R, z) for _R in R])
        
        # Calculate Eulerian bias
        delta1 = delta_c # FIXME: Not strictly correct, should be fn. of Omega_m
        v1 = (delta1 / sigR)**2.
        b = 1. + (a*v1 - 1.) / delta1 + (2.*p/delta1) / (1. + (a*v1)**p)
        return b
    
    def dndlogm(self, M, z=0., type='tinker'):
        """
        Halo mass function in log mass.
        """
        if type == 'tinker':
            return M * self.n_tinker(M, z)
        else:
            return M * self.n_sheth_tormen(M, z)
    
    def n_sheth_tormen(self, M, z=0.):
        """
        Halo mass function, dn/dm, as a function of mass and redshift, from 
        Sheth & Tormen (1999).
        """
        delta_c = self.delta_c
        A = self.A
        a = self.a
        p = self.p
        
        rho = self.h*self.h*self.om * (1. + z)**3. * 2.776e11 # in Msun/Mpc^3
        
        # Integrate to find sigma(R)
        R = self.R_for_M(M)
        sigR = np.array([self.sigma_R(_R, z) for _R in R])
        
        # Get logarithmic derivative
        dlogsig = self.dlogsigM_dlogM(M, sigR)
        
        # Evaluate mass function shape
        v = (self.delta_c / sigR)**2.
        vfv = A * (1. + 1./(a*v)**p) * np.sqrt(a*v/(2.*np.pi)) * np.exp(-a*v/2.)
        
        # Evaluate halo mass function
        nm = -2. * rho/M**2. * vfv * dlogsig
        return nm
    
    def n_tinker(self, M, z=0.):
        """
        Halo mass function, dn/dm, as a function of mass and redshift. Taken 
        from Eqs. 1, 2, 5-8, and Table 2 of Tinker et al. 2008 [arXiv:0803.2706].
        """
        # Redshift scaling of parameters, from Eqs. 5-8 of Tinker et al.
        Delta = 200. # Define standard overdensity
        alpha = 10.**( -(0.75 / np.log10(Delta / 75.))**1.2 )
        A = self.A0 * (1. + z)**(-0.14)
        a = self.a0 * (1. + z)**(-0.06)
        b = self.b0 * (1. + z)**alpha
        c = self.c
        
        rho = self.h*self.h*self.om * (1. + z)**3. * 2.776e11 # in Msun/Mpc^3
        
        # Integrate to find sigma(R)
        R = self.R_for_M(M)
        sigR = np.array([self.sigma_R(_R, z) for _R in R])
        
        # Get logarithmic derivative
        dlogsig = self.dlogsigM_dlogM(M, sigR)
        
        # Evaluate shape function
        fsig = A * ((sigR/b)**(-a) + 1.) * np.exp(-c / sigR**2.)
        
        # Return halo mass function
        return -fsig * rho / M**2. * dlogsig
        

    def MHI(self, M, z=0.):
        """
        HI mass as a function of halo mass.
        (Eq. 3.2 of arXiv:1405.6713; Bagla model)
        """
        f3 = 0.014405 # Should be obtained by matching Omega_HI = obs ~ 10^-3.
        vmin = 30. # km/s
        vmax = 200. # km/s
        Mmin = 1e10 * (vmin/60.)**3. * ((1.+z)/4.)**-1.5
        Mmax = 1e10 * (vmax/60.)**3. * ((1.+z)/4.)**-1.5
        
        # Calculate M_HI(M)
        M_HI = f3 * M / (1. + (M / Mmax))
        M_HI[np.where(M < Mmin)] = 0.
        return M_HI
        
    def cumulative_hi_mass(self, M, z):
        """
        Cumulative fraction of total HI mass as a function of M_halo.
        """
        Ez2 = self.om * (1. + z)**3. + (1. - self.om)
        rho_c = self.h*self.h * Ez2 * 2.776e11 # in Msun/Mpc^3
        M_HI = self.MHI(M, z)
        nm = self.n(M, z)
        
        omega_hi = scipy.integrate.simps(nm*M_HI, M) / rho_c
        cumul_mhi = scipy.integrate.cumtrapz(nm*M_HI, M, initial=0.)
        return cumul_mhi / cumul_mhi[-1]

