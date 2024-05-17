import numpy as np
from scipy.linalg import eigvalsh
from .channel_operations import choi_state

def is_channel(t, tol=1e-6, show=False):
    choi = choi_state(t)

    if show:
        print("For the Choi matrix of the channel:")
    choi_is_state = is_density_matrix(choi, tol=tol, show=show)

    d2 = int(np.sqrt(t.shape[0]))
    d1 = int(np.sqrt(t.shape[1]))
    trace_preserving = np.allclose(np.identity(d1), 
                                   (t.T.conj() @ np.identity(d2).reshape(d2**2)).reshape((d1, d1)), atol=tol)

    if show:
        if trace_preserving:
            print("Channel is trace preserving.")
        else:
            print("Channel is not trace preserving.")
    return choi_is_state and trace_preserving


def is_density_matrix(dm, tol=1e-6, show=False):
    dtr = np.max(np.abs(np.trace(dm)-1))
    dhrm = np.max(np.abs(dm-dm.conj().transpose()))
    # This can crash the Kernel for states of dimension higher
    # than 150. It shouldnt, but it does.
    dpos = np.min(eigvalsh(dm, subset_by_index=[0, 1]))
    if show:
        print("Trace diff: ", dtr)
        print("Hermiticity diff: ", dhrm)
        print("Minimum eigval: ", dpos)
    return (dtr <= tol) and (dhrm <= tol) and (dpos >= - tol)