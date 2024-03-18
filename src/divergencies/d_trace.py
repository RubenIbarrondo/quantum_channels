import numpy as np

def d_trace(rho, sigma):
    return np.linalg.norm(rho - sigma) / 2