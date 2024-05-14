import numpy as np

def hs_dist(r, s):
    return np.linalg.norm(r-s, 'fro')
