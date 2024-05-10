import numpy as np

def computational_basis(dim, index):
    state = np.zeros(dim, dtype=complex)
    state[index] = 1
    return state