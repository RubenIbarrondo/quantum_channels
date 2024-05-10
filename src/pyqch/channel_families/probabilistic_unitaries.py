import numpy as np

def probabilistic_unitaries(dim:int, ps:np.ndarray, us:np.ndarray):
    return (np.einsum("m,mpi,mqj->pqij", ps, us, us.conj())).reshape((dim**2, dim**2))
