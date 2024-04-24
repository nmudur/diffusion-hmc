import numpy as np
from scipy.ndimage import gaussian_filter

def add_noise(truefields, loc=0, scale=1.0, seed=20):
    print(f'Adding noise to fields with mean, std = {loc}, {scale}')
    if scale==0:
        return truefields+loc # when you just want an offset

    else:
        rng = np.random.default_rng(seed)
        return truefields + rng.normal(loc, scale, truefields.shape)


def smooth(truefields, sigma=0.5):
    print(f'Smooth by sigma = {sigma}')
    assert len(truefields.shape)==3
    return np.vstack([gaussian_filter(truefields[a], sigma=sigma)[None, ...] for a in range(truefields.shape[0])])