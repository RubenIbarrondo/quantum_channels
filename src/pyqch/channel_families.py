"""
channel_families
================

This module implements functions that return the transition matrices 
describing different families of quantum channels. These functions 
cover a variety of quantum operations including classical permutations, 
dephasing, depolarizing, embedding classical channels, initializers, 
POVMs, probabilistic damping, and probabilistic unitaries.
"""

import numpy as np


def classical_permutation(dim: int, perm: list) -> np.ndarray:
    """
    Returns the transition matrix representing a classical permutation 
    channel.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.
    perm : list
        The array defining the permutation that maps i to perm[i].

    Returns
    -------
    np.ndarray
        The transition matrix representing the classical permutation channel.

    Examples
    --------
    >>> from channel_families import classical_permutation
    >>> perm = [2, 0, 1]
    >>> T = classical_permutation(3, perm)
    >>> print(T)
    [[0. 1. 0.]
     [0. 0. 1.]
     [1. 0. 0.]]
    """
    mat = np.zeros((dim, dim))
    mat[perm, np.arange(dim)] = 1
    return mat


def dephasing(dim: int, g: float, u: np.ndarray = None) -> np.ndarray:
    """
    Returns the transition matrix for a dephasing channel.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.
    g : float
        Dephasing strength.
    u : np.ndarray, optional
        The unitary matrix defining the basis in which dephasing occurs.
        Defaults to the identity matrix.

    Returns
    -------
    np.ndarray
        The transition matrix for the dephasing channel.

    Notes
    -----
    This function implements dephasing by dampening the off-diagonal terms 
    in a given basis. For generalized dephasing, one should define the 
    corresponding PVMs and include an identity with some dampening, then 
    pass them to the `povm` function.

    Examples
    --------
    >>> from channel_families import dephasing
    >>> dim = 3
    >>> g = 0.5
    >>> T = dephasing(dim, g)
    >>> print(T)
    [[1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. ]])
    """
    a = np.identity(dim)

    if u is None:
        tdeph = np.einsum("pq,pi,qj->pqij", a, a, a)
        tid = np.einsum("pi,qj->pqij", a, a)
        return ((1-g) * tid + g * tdeph).reshape((dim**2, dim**2))
    else:
        tdeph = np.einsum("sp,si,sq,sj->pqij", u.conj(), u, u, u.conj())
        tid = np.einsum("pi,qj->pqij", a, a)
        return ((1-g) * tid + g * tdeph).reshape((dim**2, dim**2))
    

def depolarizing(dim: int, p: float, r: np.ndarray = None) -> np.ndarray:
    """
    Returns the transition matrix for a depolarizing channel.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.
    p : float
        Depolarizing probability.
    r : np.ndarray, optional
        The stationary state of the depolarizing channel. Defaults to the 
        completely mixed state.

    Returns
    -------
    np.ndarray
        The transition matrix for the depolarizing channel.

    Examples
    --------
    >>> from channel_families import depolarizing
    >>> dim = 3
    >>> p = 0.2
    >>> T = depolarizing(dim, p)
    >>> rho_in = np.diag([1, 0, 0])
    >>> rho_out = (T @ rho_in.reshape(dim**2)).reshape((dim, dim))
    >>> print(rho_out)
    [[0.86666667+0.j 0.        +0.j 0.        +0.j]
     [0.        +0.j 0.06666667+0.j 0.        +0.j]
     [0.        +0.j 0.        +0.j 0.06666667+0.j]]
    """
    if r is None:
        r = np.identity(dim) / dim
    
    max_entang = np.reshape(np.identity(dim), dim**2)
    vr = np.reshape(r, dim**2)

    if p == 1:
        transfer_matrix = np.outer(vr, max_entang)
    elif p == 0:
        transfer_matrix = np.identity(dim ** 2)
    else:
        transfer_matrix = (1-p) * np.identity(dim**2, dtype=complex)
        transfer_matrix += p * np.outer(vr, max_entang)
    return transfer_matrix    


def embed_classical(dim: int, stoch_mat: np.ndarray) -> np.ndarray:
    """
    Embeds a classical stochastic matrix into a quantum transition matrix.

    The resulting process sets off-diagonal terms to zero and acts on the diagonal
    terms following the classical stochastic process.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.
    stoch_mat : np.ndarray
        The classical column-stochastic matrix to be embedded.

    Returns
    -------
    np.ndarray
        The transition matrix representing the embedded classical channel.

    Examples
    --------
    >>> from channel_families import embed_classical
    >>> stoch_mat = np.array([[0.8, 0.3], [0.2, 0.7]])
    >>> T = embed_classical(2, stoch_mat)
    >>> print(T)
    [[0.8 0.  0.  0.3]
     [0.  0.  0.  0. ]
     [0.  0.  0.  0. ]
     [0.2 0.  0.  0.7]]
    """
    a = np.identity(dim)
    return (np.einsum("ij,pq,pi->pqij", a, a, stoch_mat)).reshape((dim**2, dim**2))


def initializer(dim: int, states: np.ndarray, mode='c-q') -> np.ndarray:
    """
    Returns a transition matrix for initializing a quantum state.

    The initializer channel takes a classical probability distribution as
    input and produces a density matrix.
    
    The `mode` argument defines the format of the input and output states.
    The initial state can be a classical array (c) or a density matrix whose
    diagonal elements are used as the input probability distribution (q), 
    discarding off-diagonal terms. The output state can store only the 
    initialized states (q) or also the input probability distribution (qc)

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.
    states : np.ndarray
        Array of states defining the choices for initialization.
        The expected shape is (number of states, dim, dim), such that
        states[i] represents the state prepared when the classical
        variable takes the value i.
    mode : str, optional
        Mode of initialization ('c-q' for classical to quantum,
        'c-qc' for classical to quantum and classical, 'q-q' for 
        quantum to quantum, 'q-qc' for quantum to quantum and
        classical). Defaults to 'c-q'.

    Returns
    -------
    np.ndarray
        The transition matrix for the initializer.

    Examples
    --------
    >>> from channel_families import initializer
    >>> states = np.array([[[1, 0], [0, 0]],
                            [[.5, .5], [.5, .5]]])
    >>> T = initializer(2, states)
    >>> print(T)
    [[1.  0.5]
     [0.  0.5]
     [0.  0.5]
     [0.  0.5]]
    """
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


def povm(dim: int, pos: np.ndarray, mode='q-q') -> np.ndarray:
    """
    Returns a transition matrix for a quantum channel defined by a
    positive operator-valued measure (POVM).

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.
    pos : np.ndarray
        The POVM elements defining the channel. The expected shape is
        (number of positive operators, dim, dim), such that pos[i] is
        the ith positive operator.
    mode : str, optional
        Mode of the channel ('q-q' for quantum to quantum, 
        'q-c' for quantum to classical, 'q-qc' for quantum to
        quantum and classical). Defaults to 'q-q'.

    Returns
    -------
    np.ndarray
        The transition matrix for the POVM-based quantum channel.

    Examples
    --------
    >>> from channel_families import povm
    >>> pos = np.array([np.diag([1,0]), np.diag([0,1])])
    >>> T = povm(2, pos)
    >>> print(T)
    [[1 0 0 0]
     [0 0 0 0]
     [0 0 0 0]
     [0 0 0 1]]
    """
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


def probabilistic_damping(dim: int, p: float) -> np.ndarray:
    """
    Returns a transition matrix for a probabilistic damping channel.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.
    p : float
        Damping probability.

    Returns
    -------
    np.ndarray
        The transition matrix for the probabilistic damping channel.

    Examples
    --------
    >>> from channel_families import probabilistic_damping
    >>> dim = 2
    >>> p = 0.1
    >>> T = probabilistic_damping(dim, p)
    >>> print(T)
    [[1.  0.  0.  0.1]
     [0.  0.9 0.  0. ]
     [0.  0.  0.9 0. ]
     [0.  0.  0.  0.9]]
    """
    proy = np.zeros((dim**2, dim**2))
    proy[0,0] = 1
    
    damp = np.zeros((dim, dim, dim, dim))
    for l in range(dim-1):
        damp[l, l, l+1, l+1] = 1
    damp = damp.reshape((dim**2, dim**2))
    
    return (1-p) * np.identity(dim**2) + p * (proy+damp)


def probabilistic_unitaries(dim:int, p_arr:np.ndarray, u_arr:np.ndarray) -> np.ndarray:
    """
    Returns a transition matrix for a channel that applies unitaries 
    probabilistically.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.
    p_arr : np.ndarray
        Probabilities for each unitary operation.
    u_arr : np.ndarray
        Array of unitary matrices. The expected shape is
        (number of unitaries, dim, dim), such that u_arr[i] is
        the ith possible unitary.

    Returns
    -------
    np.ndarray
        The transition matrix for the probabilistic unitary channel.

    Examples
    --------
    This example demonstrates the implementation of a bit-flip channel.

    >>> from channel_families import probabilistic_unitaries
    >>> p_arr = np.array([0.5, 0.5])
    >>> u_arr = np.array([np.eye(2), np.array([[0, 1], [1, 0]])])
    >>> T = probabilistic_unitaries(2, p_arr, u_arr)
    >>> print(T)
    [[0.5 0.  0.  0.5]
     [0.  0.5 0.5 0. ]
     [0.  0.5 0.5 0. ]
     [0.5 0.  0.  0.5]]
    """
    return (np.einsum("m,mpi,mqj->pqij", p_arr, u_arr, u_arr.conj())).reshape((dim**2, dim**2))


def transposition(dim: int, u: np.ndarray = None) -> np.ndarray:
    """
    Returns a transition matrix for the transposition operator. This is positive but
    not completely positive, thus it is not a quantum channel.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.
    u : np.ndarray or None
        Basis in which the transposition is performed, if None the canonical basis is used.
        Defaults to None.
    
    Returns
    -------
    np.ndarray
        The transition matrix for the transposition.
    """

    tmat = np.identity(dim**2).reshape((dim,)*4)
    tmat = np.einsum("ijkl->jikl", tmat).reshape((dim**2, dim**2))
    
    if u is None:
        return tmat
    else:
        return np.kron(u.T.conj(), u.T) @ tmat @ np.kron(u, u.conj())
