import numpy as np
from scipy import linalg

def fixed_points(t, tol=1e-6):
    # t has to be square matrix
    if not t.shape[0] == t.shape[1]:
        raise ValueError("Only defined for sqare channels.")

    # get vectors associated with eigenvalue 1
    w, v = linalg.eig(t)

    fp_mask = np.abs(w-1) < tol
    n = int(np.sum(fp_mask))
    
    if n == 0:
        raise ValueError("No fixed point was found")
    elif n > 1:
        raise NotImplementedError("Transforming several fixed points into matrix form is not implemented")
    
    v_fixed_points = v[:,fp_mask]

    # if needed, reshape into positive, unit-trace matrices
    dim = int(np.sqrt(v_fixed_points.shape[0]))
    ms = v_fixed_points.reshape((dim, dim, n))
    
    # become them hermitian
    ms = ms.transpose((1, 0, 2)).conj() + ms
    
    if n==1:
        # if single fixed point then the trace cannot be null
        # so we normalize and ensure positivity just by
        return ms.reshape((dim, dim)) / np.trace(ms, axis1=0, axis2=1)
    else:
        # we have to ensure positivity first.
        # Then normalize
        raise NotImplementedError("Transforming several fixed points into matrix form is not implemented")
