import numpy as np
from scipy.stats import unitary_group, dirichlet


def state_dirichlet(dim:int, alpha:float):
    """ Generates random states sampling their spectrum from the homogeneous dirichlet distribution
    with concentration parameter alpha.
    The basis are rotated with Haar random unitary matrices.
    
    Note: alpha->0 pure states, alpha=1 uniform measure, alpha->infty maximally mixed
    
    """
    spec = dirichlet.rvs([alpha]*dim, size=1, random_state=1)[0]
    u = unitary_group.rvs(dim)
    rho = u @ np.diag(spec) @ u.T.conj()
    return rho
