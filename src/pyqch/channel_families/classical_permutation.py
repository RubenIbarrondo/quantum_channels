import numpy as np

def classical_permutation(dim: int, perm: np.ndarray):
    mat = np.zeros((dim, dim))
    mat[perm, np.arange(dim)] = 1
    return mat

