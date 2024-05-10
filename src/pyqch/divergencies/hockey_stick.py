import numpy as np

def hockey_stick(rho, sigma, gamma):
    return 1/2 * np.linalg.norm(rho- gamma * sigma, 'nuc')+1/2*np.trace(rho-gamma*sigma)