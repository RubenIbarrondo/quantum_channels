import numpy as np


def amplitude_damping(dim: int, g: float) -> np.array:
    proy = np.zeros((dim**2, dim**2))
    proy[0,0] = 1
    
    damp = np.zeros((dim, dim, dim, dim))
    for l in range(dim-1):
        damp[l, l, l+1, l+1] = 1
    damp = damp.reshape((dim**2, dim**2))
    
    return (1-g) * np.identity(dim**2) + g * (proy+damp)


def classical_permutation(dim: int, perm: np.ndarray):
    mat = np.zeros((dim, dim))
    mat[perm, np.arange(dim)] = 1
    return mat


def dephasing(dim: int, g: float, u: np.ndarray = None):
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
    

def depolarizing(dim: int, p: float, r: np.ndarray = None):
    if r is None:
        r = np.identity(dim) / dim
    
    max_entang = np.reshape(np.identity(dim), dim**2)
    vr = np.reshape(r, dim**2)

    if p == 1:
        transfer_matrix = np.outer(vr, max_entang)
    elif p == 0:
        transfer_matrix = max_entang
    else:
        transfer_matrix = (1-p) * np.identity(dim**2, dtype=complex)
        transfer_matrix += p * np.outer(vr, max_entang)
    return transfer_matrix    


def embed_classical(dim: int, stoch_mat: np.ndarray):
    a = np.identity(dim)
    return (np.einsum("ij,pq,pi->pqij", a, a, stoch_mat)).reshape((dim**2, dim**2))


def initializer(dim: int, states: np.ndarray, mode='c-q'):
    m = states.shape[0]
    a = np.identity(m)

    if mode == 'q-qc':
        mat = np.einsum('ij,ik,kl,kpq->ipjqkl', a, a, a, states)
        mat = mat.reshape(((m*dim)**2, m**2))
    elif mode == 'q-q':
        mat = np.einsum('kl,kpq->pqkl', a, states)
        mat = mat.reshape((dim**2, m**2))

    elif mode == 'c-qc':
        mat = np.einsum('ij,ik,kpq->ipjqk', a, a, states)
        mat = mat.reshape(((m*dim)**2, m))
    elif mode == 'c-q':
        mat = np.einsum('kpq->pqk', states)
        mat = mat.reshape((dim**2, m))
    
    return mat


def povm(dim: int, pos: np.ndarray, mode='q-q'):
    # Verify inputs
    # Cast pos to np array with proper shape

    # mode is either 'q-q', 'q-c', or 'q-qc'
    m = pos.shape[0]
    a = np.identity(m)

    if mode == 'q-q':
        mat = np.einsum('kij,klm->iljm', pos, pos.conj())
        mat = mat.reshape((dim**2, dim**2))
    elif mode == 'q-c':
        mat = np.einsum('ki,ksl,kj,kso->ijlo', a, pos, a.conj(), pos.conj())
        mat = mat.reshape((m**2, dim**2))
    elif mode == 'q-qc':
        mat = np.einsum('mi,mjk,ml,mno->ijlnko', a,pos, a, pos.conj())
        mat = mat.reshape(((m*dim)**2, dim**2))
    return mat


def probabilistic_unitaries(dim:int, ps:np.ndarray, us:np.ndarray):
    return (np.einsum("m,mpi,mqj->pqij", ps, us, us.conj())).reshape((dim**2, dim**2))