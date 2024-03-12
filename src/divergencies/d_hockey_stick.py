import numpy as np

def d_hockey_stick(rho, sigma, gamma):
    return 1/2 * np.linalg.norm(rho- gamma * sigma, 'nuc')+1/2*np.trace(rho-gamma*sigma)