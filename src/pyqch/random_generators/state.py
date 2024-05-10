from scipy.stats import unitary_group
import numpy as np

def state(dim: int, rank:int = None):
    if rank is None:
        rank = dim
    u = unitary_group.rvs(dim * rank)
    purification = np.outer(u[0, :], u[0,:].conjugate())
    return np.trace(np.reshape(purification, (dim, rank, dim, rank)), axis1=1, axis2=3)
