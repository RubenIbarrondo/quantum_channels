"""
channel_families
================

This module implements functions that return the transition matrices or
Kraus representation describing different families of quantum channels.

These functions cover a variety of channels including classical permutations, 
dephasing, depolarizing, embedding classical channels, initializers, 
POVMs, probabilistic damping, and probabilistic unitaries.
"""

import numpy as np
import pyqch.channel_operations as co

def amplitude_damping(dim: int, lamb: float, x: np.ndarray | int = 1, y: np.ndarray | int = 0, kraus_representation: bool = False) -> np.ndarray:
    """
    Amplitude damping in arbitrary dimension.

    It describes the coherent damping from state |x> to |y>.
    
    Reduces to the usual qubit amplitude damping for dim=0, x=1, y=0.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.
    lamb : float
        The damping parameter ranging from 0 to 1.
    x : np.ndarray | int, optional
        The origin state for the damping, by default 1. If int, it is interpreted as a basis state.
    y : np.ndarray | int, optional
        The target state for the damping, by default 0. If int, it is interpreted as a basis state.
    kraus_representation : bool, optional
        If true, the function returns the Kraus representation of the channel with shape 
        (Kraus operator index, dim2, dim1), defaults to False.
        
    Returns
    -------
    np.ndarray
        The transition matrix or Kraus operators representing the amplitude damping channel.

    Notes
    -----
    For the general d-dimensional formulation we considered Definition 1 in [3]_.

    References
    ----------
    .. [3] Frederik vom Ende, (2024) "A Sufficient Criterion for Divisibility of Quantum Channels". arXiv: 2407.17103

    Raises
    ------
    ValueError
        If lambda not in [0, 1]; x or y are int and are not in [0, dim), or are np.ndarray and shape not compatible with (dim,1).
    """
    # Assert argument integrity
    if not (0<= lamb <= 1):
        raise ValueError(f"Invalid value for argument lamb, {lamb} does not satisfy 0 <= lamb <= 1.")
    if isinstance(x, (int, np.integer)):
        if not (0<= x < dim):
            raise ValueError(f"Invalid value for argument x, {x} does not satisfy 0 <= x <= dim={dim}.")
    else:
        if not (x.shape == (dim, 1) or x.shape == (dim,)):
            raise ValueError(f"Invalid shape for array x, {x.shape} not compatible with ({dim},).")
    if isinstance(y, (int, np.integer)):
        if not (0<= y < dim):
            raise ValueError(f"Invalid value for argument y, {y} does not satisfy 0 <= y <= dim={dim}.")
    else:
        if not (y.shape == (dim, 1) or y.shape == (dim,)):
            raise ValueError(f"Invalid shape for array y, {y.shape} not compatible with ({dim},).")

    # Define the Kraus operators
    Klamb = np.identity(dim, dtype=complex)
    Llamb = np.zeros((dim, dim), dtype=complex)

    if isinstance(x, (int, np.integer)):
        Klamb[x, x] = np.sqrt(1-lamb)
        if isinstance(y, (int, np.integer)):
            Llamb[y, x] = np.sqrt(lamb)
        else:
            Llamb[:, x] = np.sqrt(lamb) * y
   
    else:
        Klamb -= (1 - np.sqrt(1-lamb)) * np.outer(x, x.conj())
        if isinstance(y, (int, np.integer)):
            Llamb[y, :] = np.sqrt(lamb) * x.conj()
        else:
            Llamb = np.sqrt(lamb) * np.outer(y, x.conj())
    
    if kraus_representation:
        return np.array([Klamb, Llamb])
    else:
        # Construct the transfer matrix
        mat = np.kron(Klamb, Klamb.conj()) + np.kron(Llamb, Llamb.conj())
        return mat


def classical_permutation(dim: int, perm: list, kraus_representation: bool = False) -> np.ndarray:
    """
    Returns the transition matrix representing a classical permutation 
    channel.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.
    perm : list
        The array defining the permutation that maps i to perm[i].
    kraus_representation : bool, optional
        Defaults to False. WARNING: This map does not admit a Kraus representation, if it
        is set to True it will rise a ValueError.
    
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
    if kraus_representation:
        raise ValueError('This map does not admit the usual Kraus representation.')
    
    mat = np.zeros((dim, dim))
    mat[perm, np.arange(dim)] = 1
    return mat


def dephasing(dim: int, g: float | np.ndarray, u: np.ndarray = None, kraus_representation: bool = False) -> np.ndarray:
    """
    Returns the transition matrix for a dephasing channel.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.
    g : float | np.ndarray
        Dephasing strength. If float, it is used as an uniform damping for all
        off-diagonal terms; if np.ndarray, g[i,j] describes the damping of the
        element (i,j).
        The validity of g to define an appropriate dephasing channel is not verified.
    u : np.ndarray, optional
        The unitary matrix defining the basis in which dephasing occurs.
        Defaults to the identity matrix.
    kraus_representation : bool, optional
        If true, the function returns the Kraus representation of the channel with shape 
        (Kraus operator index, dim2, dim1), defaults to False.
    
    Returns
    -------
    np.ndarray
        The transition matrix or Kraus operators for the dephasing channel.

    Notes
    -----
    This function implements dephasing by dampening the off-diagonal terms 
    in a given basis. If g is a matrix, it should be real, symmetric, positive
    semi-definite and all diagonal terms equal to one.
    
    For other generalized dephasing, one could define the corresponding 
    PVMs and include an identity with some dampening, then pass them to 
    the `povm` function.

    Examples
    --------
    >>> from channel_families import dephasing
    >>> dim = 3
    >>> g = 0.4
    >>> T = dephasing(dim, g)
    >>> print(T)
    [[1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0.6, 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0.6, 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0.6, 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0.6, 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0.6, 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.6, 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. ]])
    """
    if kraus_representation:
        if u is not None:
            kraus_ops_id = dephasing(dim, g, u = None, kraus_representation=True)
            kraus_ops = np.einsum('ji,ljp,pq->liq', u.conj(), kraus_ops_id, u)
            return kraus_ops
        else:
            if np.isscalar(g):
                if not (0 <= g <= 1):
                    raise ValueError('This map does not admit the usual Kraus representation with g = {g}.')
                # This form is simple and intuitive
                # although not optimal in Kraus rank
                # (at least rank d can be obtained)
                kraus_ops = np.zeros((dim+1, dim, dim))
                kraus_ops[0] = np.diag(np.full(dim, np.sqrt(1 - g)))

                for k in range(dim):
                    kraus_ops[k+1, k, k] = np.sqrt(g)
                
                return kraus_ops

            else:
                # Diagonalize g and check whether it is positive
                eigs, eigvecs = np.linalg.eigh(g)

                # For the Kraus form to be equivalent to the transition matrix
                # g has to be positive. This is not required for the transition matrix.
                if not np.all(0 <= eigs):
                    raise ValueError('This map does not admit the usual Kraus representation with g = {g}.')

                kraus_ops = np.zeros((dim, dim, dim))

                for k in range(dim):
                    kraus_ops[k] = np.diag(np.sqrt(eigs[k]) * eigvecs[:, k])
                return kraus_ops
    
    else:
        a = np.identity(dim)

        if np.isscalar(g):
            
            if u is None:
                tdeph = np.einsum("pq,pi,qj->pqij", a, a, a)
                tid = np.einsum("pi,qj->pqij", a, a)
                return ((1-g) * tid + g * tdeph).reshape((dim**2, dim**2))
            else:
                tdeph = np.einsum("sp,si,sq,sj->pqij", u.conj(), u, u, u.conj())
                tid = np.einsum("pi,qj->pqij", a, a)
                return ((1-g) * tid + g * tdeph).reshape((dim**2, dim**2))
        else: 
            if u is None:
                return np.einsum('pi,qj,ij->pqij', a, a, g).reshape((dim**2, dim**2))
            else:
                return np.einsum('rp,ri,sq,sj,rs->pqij', u.conj(), u, u, u.conj(), g).reshape((dim**2, dim**2))
    

def depolarizing(dim: int, p: float, r: np.ndarray = None, kraus_representation: bool = False, kraus_atol: float = 1e-7) -> np.ndarray:
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
    kraus_representation : bool, optional
        If true, the function returns the Kraus representation of the channel with shape 
        (Kraus operator index, dim2, dim1), defaults to False.
    kraus_atol : float, optional
        The tolerance to neglect a Kraus operator, defaults to 1e-7.
    
    Returns
    -------
    np.ndarray
        The transition matrix or Kraus operators for the depolarizing channel.

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
    
    if not kraus_representation:
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
    else:
        # Diagonalize r or use it if it is already diagonal
        if np.allclose(np.diag(np.diag(r)), r):
            w = np.diag(r)
            v = np.identity(dim)
        else:
            w, v = np.linalg.eigh(r)

        # Get the effective rank of r and the mask
        # for those eigenvalues
        mask = (w > kraus_atol)
        rank = np.count_nonzero(mask)

        # Ensure normalization
        w = w / np.sum(w[mask])

        # Define the Kraus operators
        # (This may not be optimal in Kraus rank)
        kraus_ops = np.zeros((dim * rank + 1, dim, dim))
        kraus_ops[0] = np.sqrt(1-p) * np.identity(dim)

        kl_count = 1
        for k in np.where(mask)[0]:
            for l in range(dim):
                kraus_ops[kl_count, :, l] = np.sqrt(p * w[k]) * v[:, k]
                kl_count += 1
        return kraus_ops
        

def embed_classical(dim: int, stoch_mat: np.ndarray, kraus_representation: bool = False,  kraus_atol: float = 1e-7) -> np.ndarray:
    """
    Embeds a classical stochastic matrix into a quantum transition matrix.

    The resulting process sets off-diagonal terms to zero and acts on the diagonal
    terms following the classical stochastic process.

    Parameters
    ----------
    dim : int
        The dimension of the input Hilbert space.
    stoch_mat : np.ndarray
        The classical column-stochastic matrix to be embedded.
    kraus_representation : bool, optional
        If true, the function returns the Kraus representation of the channel with shape 
        (Kraus operator index, dim2, dim1), defaults to False.
    kraus_atol : float, optional
        The tolerance to neglect a Kraus operator, defaults to 1e-7.

    Returns
    -------
    np.ndarray
        The transition matrix or Kraus operators representing the embedded classical channel.

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
    
    dim_in = dim
    dim_out = stoch_mat.shape[0]

    if kraus_representation:
        krank = np.count_nonzero(stoch_mat >  kraus_atol)

        kraus_ops = np.zeros((krank, dim_out, dim_in))
        kl_count = 0
        for k in range(dim_out):
            for l in range(dim_in):
                if stoch_mat[k, l] > kraus_atol:
                    kraus_ops[kl_count, k, l] = np.sqrt(stoch_mat[k, l])
                    kl_count += 1
                
        return kraus_ops
    else:
        a_in = np.identity(dim_in)
        a_out = np.identity(dim_out)
        return (np.einsum("ij,pq,pi->pqij", a_in, a_out, stoch_mat)).reshape((dim_out**2, dim_in**2))


def initializer(dim: int, states: np.ndarray, mode='c-q', kraus_representation: bool = False, kraus_atol: float = 1e-7) -> np.ndarray:
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
        The dimension of the output Hilbert space.
    states : np.ndarray
        Array of states defining the choices for initialization.
        The expected shape is either (number of states, dim, dim) or
        (number of states, dim) , such that states[i] represents the
        state prepared when the classical variable takes the value i (described
        as a density matrix or as a vector).
    mode : str, optional
        Mode of initialization ('c-q' for classical to quantum,
        'c-qc' for classical to quantum and classical, 'q-q' for 
        quantum to quantum, 'q-qc' for quantum to quantum and
        classical). Defaults to 'c-q'.
    kraus_representation : bool, optional
        If true, the function returns the Kraus representation of the channel with shape 
        (Kraus operator index, dim2, dim1), defaults to False.
        Some modes do not admit a Kraus representation.
    kraus_atol : float, optional
        The tolerance to neglect a Kraus operator, defaults to 1e-7.
    
    Returns
    -------
    np.ndarray
        The transition matrix or Kraus operators for the initializer.

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

    if kraus_representation:
        if mode.startswith('c'):
            raise ValueError(f'This map does not admit the usual Kraus representation with mode {mode}.')
        
        # Obtain the spectral decomposition of each state
        if len(states.shape) == 2:
            w = [np.ones(1)] * m
            vecs = [state for state in states]
        else:
            w = []
            vecs = []
            for state in states:
                eigvals, eigvecs = np.linalg.eigh(state) 

                state_rank = np.count_nonzero(eigvals > kraus_atol)
                w.append(eigvals[-state_rank:])
                vecs.append(eigvecs[:,-state_rank:])
        
        krank = np.sum([len(eigvals) for eigvals in w])
        
        if mode == 'q-q':
            kraus_ops = np.zeros((krank, dim, m))
        else:
            kraus_ops = np.zeros((krank, dim * m, m))

        state_ind = 0
        cum_ind = 0
        for eigvals, eigvecs in zip(w, vecs):
            state_rank = eigvals.shape[0]
            
            if mode == 'q-q':
                kraus_ops[cum_ind:cum_ind+state_rank,:, state_ind] = (np.sqrt(eigvals) * eigvecs).T
            else:
                kraus_ops[cum_ind:cum_ind+state_rank,state_ind::m, state_ind] = (np.sqrt(eigvals) * eigvecs).T

            state_ind += 1
            cum_ind += state_rank

        return co.kraus_operators(initializer(dim, states, mode, kraus_representation=False))
    
    if len(states.shape) == 2:
        states_as_dm = np.array([np.outer(s, s.conj()) for s in states])
        return initializer(dim=dim, states=states_as_dm, mode=mode)
    else:
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
        else:
            raise ValueError(f'Invalid mode {mode}.')
        return mat


def povm(dim: int, pos: np.ndarray, mode='q-q', kraus_representation: bool = False) -> np.ndarray:
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
    kraus_representation : bool, optional
        If true, the function returns the Kraus representation of the channel with shape 
        (Kraus operator index, dim2, dim1), defaults to False.
        Some modes do not admit a Kraus representation.
    
    Returns
    -------
    np.ndarray
        The transition matrix or Kraus operators for the POVM-based quantum channel.

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
    # Verify inputs ?
    # Cast pos to np array with proper shape ?
    m = pos.shape[0]

    if kraus_representation:
        if mode.endswith('-c'):
            raise ValueError(f'This map does not admit the usual Kraus representation with mode {mode}.')
        elif mode == 'q-q':
            return pos
        else:
            kraus_ops = np.zeros((m, dim * m, dim))
            for k, po in enumerate(pos):
                kraus_ops[k,k::m,:] = po
            return kraus_ops
        
    a = np.identity(m)

    if mode == 'q-q':
        mat = np.einsum('kij,klm->iljm', pos, pos.conj())
        mat = mat.reshape((dim**2, dim**2))
    elif mode == 'q-c':
        mat = np.einsum('ksl,kj,kso->jlo', pos, a.conj(), pos.conj())
        mat = mat.reshape((m, dim**2))
    elif mode == 'q-qc':
        mat = np.einsum('mi,mjk,ml,mno->ijlnko', a,pos, a, pos.conj())
        mat = mat.reshape(((m*dim)**2, dim**2))
    else:
        raise ValueError(f'Invalid mode {mode}.')
    return mat


def probabilistic_damping(dim: int, p: float, kraus_representation: bool = False) -> np.ndarray:
    """
    Returns a transition matrix for a probabilistic damping channel.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.
    p : float
        Damping probability.
    kraus_representation : bool, optional
        If true, the function returns the Kraus representation of the channel with shape 
        (Kraus operator index, dim2, dim1), defaults to False.
    
    Returns
    -------
    np.ndarray
        The transition matrix or Kraus operators for the probabilistic damping channel.

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
    if kraus_representation:
        if p == 0.0:
            kraus_ops = np.zeros((1, dim, dim))
            kraus_ops[0] = np.identity(dim)
            return kraus_ops
        
        kraus_ops = np.zeros((dim+1, dim, dim))
        
        np.fill_diagonal(kraus_ops[0], np.sqrt(1-p))

        kraus_ops[1,0,0] = np.sqrt(p)

        for k in range(1, dim):
            kraus_ops[k+1,k-1,k] = np.sqrt(p)

        if p == 1.0:
            return kraus_ops[1:]
        else:
            return kraus_ops
        
    proy = np.zeros((dim ** 2, dim ** 2))
    proy[0,0] = 1
    
    damp = np.zeros((dim, dim, dim, dim))
    for l in range(dim-1):
        damp[l, l, l+1, l+1] = 1
    damp = damp.reshape((dim**2, dim**2))
    
    return (1-p) * np.identity(dim**2) + p * (proy+damp)


def probabilistic_unitaries(dim:int, p_arr:np.ndarray, u_arr:np.ndarray, kraus_representation: bool = False) -> np.ndarray:
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
    kraus_representation : bool, optional
        If true, the function returns the Kraus representation of the channel with shape 
        (Kraus operator index, dim2, dim1), defaults to False.
    
    Returns
    -------
    np.ndarray
        The transition matrix or Kraus operators for the probabilistic unitary channel.

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
    if kraus_representation:
        return np.einsum('p,pij->pij', np.sqrt(p_arr), u_arr)
    return (np.einsum("m,mpi,mqj->pqij", p_arr, u_arr, u_arr.conj())).reshape((dim**2, dim**2))


def transposition(dim: int, u: np.ndarray = None, kraus_representation: bool = False) -> np.ndarray:
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
    kraus_representation : bool, optional
        Defaults to False. WARNING: This map does not admit a Kraus representation, if it
        is set to True it will rise a ValueError.
    
    Returns
    -------
    np.ndarray
        The transition matrix for the transposition.
    """
    if kraus_representation:
        raise ValueError('This map does not admit the usual Kraus representation.')
    tmat = np.identity(dim**2).reshape((dim,)*4)
    tmat = np.einsum("ijkl->jikl", tmat).reshape((dim**2, dim**2))
    
    if u is None:
        return tmat
    else:
        return np.kron(u.T.conj(), u.T) @ tmat @ np.kron(u, u.conj())
