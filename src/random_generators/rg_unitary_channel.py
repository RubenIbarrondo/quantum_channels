import numpy as np
from scipy.stats import unitary_group

def rg_unitary_channel(dim: int):
    u = unitary_group.rvs(dim)
    return np.kron(u, u.conjugate())