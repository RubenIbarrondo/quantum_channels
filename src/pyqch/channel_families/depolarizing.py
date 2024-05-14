import numpy as np

def depolarizing(dim: int, p: float, r: np.ndarray = None):
    if r is None:
        r = np.identity(dim) / dim
    
    max_entang = np.reshape(np.identity(dim), dim**2)
    vr = np.reshape(r, dim**2)

    if p == 1:
        transfer_matrix = np.outer(vr, max_entang)
    elif p == 0:
        transfer_matrix = max_entang
    else:
        transfer_matrix = (1-p) * np.identity(dim**2, dtype=complex)
        transfer_matrix += p * np.outer(vr, max_entang)
    return transfer_matrix
