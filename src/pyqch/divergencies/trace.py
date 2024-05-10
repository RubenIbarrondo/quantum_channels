import numpy as np

def trace(rho, sigma):
    return np.linalg.norm(rho - sigma, 'nuc') / 2