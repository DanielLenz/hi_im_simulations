import numpy as np


def get_test_clustering(ra_range, dec_range, v_range, n_samples):

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

    # velocities in km/s
    velos = np.random.uniform(
        low=v_range[0],
        high=v_range[1],
        size=n_samples)

    locations = dict(
        ras=ras,
        decs=decs,
        velos=velos,
        n_samples=n_samples)

    return locations
