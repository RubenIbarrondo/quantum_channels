import numpy as np
from scipy import linalg

def co_fixed_points(t, tol=1e-6):
    # get vectors associated with eigenvalue 1
    w, v = linalg.eig(t)
    
    if not np.any(np.abs(w-1)<tol):
        raise ValueError("No fixed point was found")
    
    v_fixed_points = v[:,np.where(np.abs(w-1)<tol)]

    # if needed, reshape into positive, unit-trace matrices
    dim = int(np.sqrt(v_fixed_points.shape[0]))
    n = v_fixed_points.shape[1]
    ms = v_fixed_points.reshape((dim, dim, n))
    
    # become them hermitian
    ms = ms.transpose((1, 0, 2)).conj() + ms
    
    if n==1:
        # if single fixed point then the trace cannot be null
        # so we normalize and ensure positivity just by
        return ms / np.trace(ms, axis1=0, axis2=1)
    else:
        # we have to ensure positivity first.
        # Then normalize
        raise NotImplementedError("Transforming several fixed points into matrix form is not implemented")
