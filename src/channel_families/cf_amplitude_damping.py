import numpy as np


def cf_amplitude_damping(dim: int, g: float):
    proy = np.zeros((dim**2, dim**2))
    proy[0,0] = 1
    
    damp = np.zeros((dim, dim, dim, dim))
    for l in range(dim-1):
        damp[l, l, l+1, l+1] = 1
    damp = damp.reshape((dim**2, dim**2))
    
    return (1-g) * np.identity(dim**2) + g * (proy+damp)
