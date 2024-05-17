import numpy as np
from scipy import linalg


def choi_state(t):
    d2 = int(np.sqrt(t.shape[0]))
    d1 = int(np.sqrt(t.shape[1]))

    t = t.reshape((d2, d2, d1, d1))

    choi = 1/d1 * t.transpose((0, 2, 1, 3)).reshape((d1*d2, d1*d2))
    return choi


def tensor(t_arr, n=None):
    # n is only used if ts is not a list
    if isinstance(t_arr, list) and len(t_arr) == 2:
        t = t_arr[0]
        g = t_arr[1]
        td1 = int(np.sqrt(t.shape[1]))
        td2 = int(np.sqrt(t.shape[0]))
        gd1 = int(np.sqrt(g.shape[1]))
        gd2 = int(np.sqrt(g.shape[0]))
        
        tres = t.reshape((td2, td2, td1, td1))
        gres = g.reshape((gd2, gd2, gd1, gd1))
            
        tg = np.einsum("ijkl,mnop->imjnkolp", tres, gres)

        return tg.reshape(((td2*gd2)**2, (td1*gd1)**2))
    
    elif isinstance(t_arr, list) and len(t_arr) > 2:
        return tensor([t_arr[0], tensor(t_arr[1:])])
    elif n is not None:
        if n==1:
            return t_arr
        else:
            return tensor([t_arr]*n)
    else:
        raise ValueError()
    

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
