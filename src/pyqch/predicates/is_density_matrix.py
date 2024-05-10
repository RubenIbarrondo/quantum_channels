import numpy as np
from scipy import linalg

def is_density_matrix(dm, tol=1e-6, show=False):
    dtr = np.max(np.abs(np.trace(dm)-1))
    dhrm = np.max(np.abs(dm-dm.conj().transpose()))
    # This can crash the Kernel for states of dimension higher
    # than 150. It shouldnt, but it does.
    dpos = np.min(linalg.eigvalsh(dm, subset_by_index=[0, 1]))
    if show:
        print("Trace diff: ", dtr)
        print("Hermiticity diff: ", dhrm)
        print("Minimum eigval: ", dpos)
    return (dtr <= tol) and (dhrm <= tol) and (dpos >= - tol)