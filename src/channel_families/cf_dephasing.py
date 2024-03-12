import numpy as np

def cf_dephasing(dim: int, g: float, u: np.ndarray = None):
    """
    This only implements “given a basis dampen the off-diagonal terms”.
    For generalized dephasing one should define the corresponding PVMs 
    and include an identity with some dampening.

    u defaults to np.identity(dim)
    """
    a = np.identity(dim)

    if u is None:
        t1 = np.einsum("pq,pi,qj->pqij", a, a, a)
        t2 = np.einsum("pi,qj->pqij", a, a)
        return ((1-g) * t1 + g * t2).reshape((dim**2, dim**2))
    else:
        t1 = np.einsum("sp,si,sq,sj->pqij", u.conj(), u, u, u.conj())
        t2 = np.einsum("pi,qj->pqij", a, a)
        return ((1-g) * t1 + g * t2).reshape((dim**2, dim**2))