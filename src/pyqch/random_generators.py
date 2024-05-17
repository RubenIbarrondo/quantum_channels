import numpy as np
from scipy.stats import unitary_group, dirichlet

def channel(dim_in:int, dim_out:int = None, kraus_rank:int = None):
    
    if dim_out is None:
        dim_out = dim_in
    if kraus_rank is None:
        kraus_rank = dim_out * dim_in

    if dim_out * kraus_rank < dim_in:
        raise ValueError(f"Channel specifications must satisfy dim_out * kraus_rank >= dim_in, but {dim_out} * {kraus_rank} < {dim_in}.")
    
    # Generate a random isometry from dim_in to dim_out * kraus_rank
    # 1) Create a random unitary of dimension dim_out * kraus_rank
    # 2) Take only dim_in columns, i.e. setting the initial state of the environment to 0
    v = unitary_group.rvs(dim_out * kraus_rank)[:, :dim_in]
    v = np.reshape(v, (dim_out, kraus_rank, dim_in))

    # Taking products and tracing out the final environment with dimension kraus_rank
    t_astensor = np.einsum("nik,mil->nmkl" , v, v.conj())
    return np.reshape(t_astensor, (dim_out**2, dim_in**2))


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


def state(dim: int, rank:int = None):
    if rank is None:
        rank = dim
    u = unitary_group.rvs(dim * rank)
    purification = np.outer(u[0, :], u[0,:].conjugate())
    return np.trace(np.reshape(purification, (dim, rank, dim, rank)), axis1=1, axis2=3)


def unitary_channel(dim: int):
    u = unitary_group.rvs(dim)
    return np.kron(u, u.conjugate())