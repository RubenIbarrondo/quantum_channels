import numpy as np
from scipy.stats import unitary_group

def channel(dim_in:int, dim_out:int = None, kraus_rank:int = None):
    if dim_out is None:
        dim_out = dim_in
    if kraus_rank is None:
        kraus_rank = dim_out * dim_in

    u = unitary_group.rvs(dim_out * dim_in * kraus_rank)
    u = np.reshape(u, (dim_out,dim_in, kraus_rank,dim_out, dim_in, kraus_rank))

    # The first dim_out should survive, for the second we should select the 0 axis
    # the first dim_in should be traced out, the first one survive
    # the first kraus rank should be traced out too, the first one should be set to the zero axis
    ak_arr = u[:, :, :, 0, :, 0] 
    
    t_astensor = np.einsum("ijkl,mjkn->imln", ak_arr, ak_arr.conjugate())
    return np.reshape(t_astensor, (dim_out**2, dim_in**2))

