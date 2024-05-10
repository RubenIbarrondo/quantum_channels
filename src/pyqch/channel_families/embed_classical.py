import numpy as np

def embed_classical(dim: int, stoch_mat: np.ndarray):
    a = np.identity(dim)
    return (np.einsum("ij,pq,pi->pqij", a, a, stoch_mat)).reshape((dim**2, dim**2))