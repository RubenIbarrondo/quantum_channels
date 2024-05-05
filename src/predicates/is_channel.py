import numpy as np
from src.channel_operations.co_choi_state import co_choi_state
from src.predicates.is_density_matrix import is_density_matrix

def is_channel(t, tol=1e-6, show=False):
    choi = co_choi_state(t)

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