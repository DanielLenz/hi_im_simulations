import numpy as np
from scipy import stats


def get_stewart2014(velocities, n_samples=1):

    # rotation velocities
    class v_rot_rv(stats.rv_continuous):
        def _pdf(self, x):
            return (
                np.exp(-0.5 * ((np.log10(x) - 2.2) / 0.3)**2.) /
                (0.3 * np.log(10.) * np.sqrt(2. * np.pi)))

    gal_params = dict(
        v_rot=eff_rot_velos,
        skewness=skewness,
        flux=fluxes,
        turb_velo=turb_velos,
        solid_rot=solid_rot
    )

    return gal_params


def get_simple_priors(v_range, n_samples=1):

    # radial velocities
    v_rad = np.random.uniform(*v_range, size=n_samples)

    # rotation velocities
    inclination = np.random.uniform(0, 360., n_samples)
    rot_velos = stats.norm.rvs(loc=200, scale=50, size=n_samples)
    rot_velos = np.clip(rot_velos, a_min=50., a_max=None)
    eff_rot_velos = rot_velos * np.sin(np.radians(inclination))

    # skewness, has to be within [-1, 1]
    skewness = stats.norm.rvs(loc=0, scale=0.3, size=n_samples)
    skewness = np.clip(skewness, a_min=-1., a_max=1.)

    # flux densities in Jy.km/s
    fluxes = stats.norm.rvs(loc=50, scale=10, size=n_samples)
    fluxes = np.clip(fluxes, a_min=0., a_max=None)

    # turbulent broadening in km/s
    turb_velos = stats.norm.rvs(loc=15., scale=5., size=n_samples)
    turb_velos = np.clip(turb_velos, a_min=0., a_max=None)

    # fraction of solid body rotation, must be between 0 and 1
    solid_rot = stats.norm.rvs(loc=0.2, scale=0.05, size=n_samples)
    solid_rot = np.clip(solid_rot, a_min=0., a_max=None)

    gal_params = dict(
        v_rot=eff_rot_velos,
        skewness=skewness,
        flux=fluxes,
        turb_velo=turb_velos,
        solid_rot=solid_rot,
        v_rad=v_rad
    )

    return gal_params
