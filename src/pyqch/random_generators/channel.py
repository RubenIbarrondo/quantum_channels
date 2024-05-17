import numpy as np
from scipy.stats import unitary_group

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
