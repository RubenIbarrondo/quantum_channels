"""
channel_operations
==================

This module implements various operations on quantum channels. It includes 
functions for obtaining the Choi state of a channel, performing tensor 
products of channel representations, and finding fixed points of a channel.
"""

import numpy as np
from scipy import linalg
import cvxpy as cp
from pyqch import channel_families as cf


def choi_state(t:np.ndarray) -> np.ndarray:
    """
    Returns the Choi state representation of a given quantum channel.

    Parameters
    ----------
    t : np.ndarray
        The transition matrix representing the quantum channel.

    Returns
    -------
    np.ndarray
        The Choi state corresponding to the quantum channel.

    Examples
    --------
    >>> from channel_operations import choi_state
    >>> from channel_families import depolarizing
    >>> dim = 2
    >>> p = .5
    >>> t = depolarizing(dim, p)
    >>> choi = choi_state(t)
    >>> print(choi)
    [[0.375+0.j 0.   +0.j 0.   +0.j 0.25 +0.j]
     [0.   +0.j 0.125+0.j 0.   +0.j 0.   +0.j]
     [0.   +0.j 0.   +0.j 0.125+0.j 0.   +0.j]
     [0.25 +0.j 0.   +0.j 0.   +0.j 0.375+0.j]]
    """
     
    d2 = int(np.sqrt(t.shape[0]))
    d1 = int(np.sqrt(t.shape[1]))

    t = t.reshape((d2, d2, d1, d1))

    choi = 1/d1 * t.transpose((0, 2, 1, 3)).reshape((d1*d2, d1*d2))
    return choi


def kraus_operators(t:np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """
    Returns the Kraus operators of a given quantum channel.

    Parameters
    ----------
    t : np.ndarray
        The transition matrix representing the quantum channel,
        with shape (dim2**2, dim1**2)
    atol : float, optional
        The absolute tolerance for the norm of a Kraus operator to be
        considered as a non-trivial Kraus operator, defaults to 1e-6.

    Returns
    -------
    np.ndarray
        The Kraus operators corresponding to the quantum channel,
        with shape (kraus_rank, dim2, dim1).

    Notes
    -----
    Currently, the method uses the spectral decomposition of the Choi matrix and discards
    eigenvalues that are smaller than atol. This is equivalent to discarding Kraus operators
    with np.trace(K.T @ K) <= atol * dim1.
    
    Using the matrix square root of the Choi matrix is a valid alternative if
    the canonical form is not needed. It seems more efficient in cases where the
    matrix is sparse.
    """
     
    d2 = int(np.sqrt(t.shape[0]))
    d1 = int(np.sqrt(t.shape[1]))

    choi = choi_state(t)

    # Using eigen-decomposition
    w, v = np.linalg.eigh(choi)
    
    # Only keeping operators up to some norm
    kraus_rank = int(np.sum(w > atol))
    w[:-kraus_rank] = 0  # only preserve highest values

    # Renormalize the operators so that the channel is still trace preserving
    # (or trace decreasing or whatever it was)
    w = w / np.sum(w) * np.trace(choi)
    w *= d1  # required to fix the scale of the Choi representation

    kraus_ops = ((v * np.sqrt(w)).T)[-kraus_rank:, :].reshape((kraus_rank, d2, d1))

    return kraus_ops


def tensor(t_arr: np.ndarray | list[np.ndarray], n: int = 1) -> np.ndarray:
    """
    Returns the tensor product of quantum channels represented by their 
    transition matrices.

    This function can be used in two ways:

    1. Provide a list of transition matrices to compute their tensor product.
    
    2. Provide a single transition matrix and a value for `n` to compute the 
       self-tensor product applied `n` times.

    Parameters
    ----------
    t_arr : np.ndarray or list of np.ndarray
        A single transition matrix or a list of transition matrices.
    n : int, optional
        Number of times to apply the tensor product to a single transition 
        matrix. Ignored if `t_arr` is a list of matrices. Defaults to 1.

    Returns
    -------
    np.ndarray
        The resulting tensor product of the transition matrices.

    Examples
    --------
    Define a qubit-depolarizing and a qubit identity channel.

    >>> from channel_operations import tensor
    >>> from channel_families import depolarizing
    >>> dim = 2
    >>> p = 0.5
    >>> tdepol = depolarizing(dim, p)
    >>> tid = np.identity(dim**2)

    Example 1: Tensor product of a list of matrices
    Demonstrates how to construct an inhomogeneous local depolarizer that 
    acts on two qubits, but only adds noise to the state of the first qubit.

    >>> tensor_product = tensor([tdepol, tid])
    >>> tensor_product.shape
    (16, 16)

    Example 2: Self-tensor product of a single matrix applied n times
    Demostrates how to build a multi-qubit homogeneous depolarizing channel.

    >>> n = 3
    >>> local_depol = tensor(tdepol, n)
    >>> local_depol.shape
    (64, 64)
    """
    # n is only used if ts is not a list
    if isinstance(t_arr, list):
        if len(t_arr) == 1:
            return t_arr[0]
    
        elif len(t_arr) == 2:
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
        
        elif len(t_arr) > 2:
            return tensor([t_arr[0], tensor(t_arr[1:])])
        elif len(t_arr) == 0:
            raise ValueError("Expected len(t_arr) > 0.")
   
    elif isinstance(t_arr, np.ndarray):
        if n==1:
            return t_arr
        else:
            return tensor([t_arr]*n)
    else:
        raise TypeError("t_arr must be either a list of np.ndarray or a single np.ndarray")
    
    raise RuntimeError("Unexpected end of function: No return value")
    

def fixed_points(t:np.ndarray, tol:float=1e-6) -> np.ndarray:
    """
    Returns the fixed points of a given quantum channel.

    Parameters
    ----------
    t : np.ndarray
        The transition matrix representing the quantum channel.
    tol : float, optional
        Tolerance for determining fixed points. Defaults to 1e-6.

    Returns
    -------
    np.ndarray
        The fixed points of the quantum channel.

    Raises
    ------
    NotImplementedError
        If more than one fixed point is detected.

    Examples
    --------
    >>> from channel_operations import fixed_points
    >>> from channel_families import depolarizing
    >>> dim = 3
    >>> p = 0.5
    >>> rho_ref = np.diag([.5, .3, .2])
    >>> t = depolarizing(dim, p, rho_ref)
    >>> fixed_pt = fixed_points(t)
    >>> np.allclose(fixed_pt, rho_ref)
    True
    """

    no_multi_fp_msg = "Transforming multiple fixed points into matrix form is not implemented"
    # t has to be square matrix
    if not t.shape[0] == t.shape[1]:
        raise ValueError("Only defined for sqare channels.")

    # get vectors associated with eigenvalue 1
    w, v = linalg.eig(t)

    fp_mask = np.abs(w-1) < tol
    n = int(np.sum(fp_mask))
    
    if n == 0:
        raise RuntimeError("No fixed point was found")
    elif n > 1:
        raise NotImplementedError(no_multi_fp_msg)
    
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
        raise NotImplementedError(no_multi_fp_msg)


def twirling(t: np.ndarray, r_in: list[np.ndarray], r_out: list[np.ndarray]):
    """
    Applies a twirling operation to a quantum channel.

    The input is two representations of the finite group G, given as arrays of unitary matrices.

    Parameters
    ----------
    t : np.ndarray
        The transition matrix representing the quantum channel.
    r_in : list of np.ndarray
        Input representation of the finite group with each element labeled by an integer.
    r_out : list of np.ndarray
        Output representation of the finite group with each element labeled by an integer.

    Returns
    -------
    np.ndarray
        The transition matrix of the twirled quantum channel.

    Raises
    ------
    ValueError
        If r_in and r_out do not have the same size.

    Examples
    --------
    >>> from channel_operations import twirling
    >>> from random_generators import channel
    >>> t = channel(2, k_rank = 1)
    >>> r_in = [np.eye(2), np.array([[0, 1], [1, 0]])]
    >>> r_out = [np.eye(2), np.array([[0, -1j], [1j, 0]])]
    >>> t_twirl = twirling(t, r_in, r_out)
    >>> print(t_twirl)

    Notes
    -----
    This function applies a twirling operation, which averages the quantum channel over a group of unitary transformations.

    .. math::

        T_G(\mathcal{E})(\cdot) = \frac{1}{|G|} \sum_{g \in G} R_{out}(g^{-1}) \mathcal{E}(R_{in}(g) \cdot R_{in}(g)^\dagger) R_{out}(g^{-1})^\dagger

    """
    # r_in, r_out: and output representation of a finite group with each element is labeled by an integer
    # There should be a more efficient way...
    if len(r_in) != len(r_out):
        raise ValueError("If finite with list representation, r_in and r_out must have same size.")
    t_twirl = np.mean([np.kron(r_out[g].T.conj(), r_out[g].conj()) 
                       @ t @ 
                       np.kron(r_in[g], r_in[g].T)  for g in range(len(r_in))], axis=0)
    return t_twirl


def doeblin_coefficient(channel: np.ndarray, transpose: bool = False, subspace_projection: np.ndarray = None):
    """
    Computes the Doeblin coefficient of a quantum channel.

    The Doeblin coefficient is the maximum erasure probability for which an erasure channel 
    can be degraded into the given channel. This implementation uses the SDP construction 
    described in reference [1].

    Parameters
    ----------
    channel : np.ndarray
        The transition matrix representing the quantum channel.
    transpose : bool, optional
        Whether to check transpose-degradability instead of the usual degradability.
        Defaults to False.
    subspace_projection : np.ndarray, optional
        An orthogonal projector into a subspace of dimension at least 2. If provided, 
        it restricts the search for the coefficient to that subspace. Defaults to None.

    Returns
    -------
    float
        The Doeblin coefficient of the quantum channel. Returns None if the problem is 
        infeasible or unbounded.

    Notes
    -----
    This function implements the SDP (Semidefinite Programming) construction given 
    in reference [1] to find the Doeblin coefficient.

    The subspace restriciton is not detailed in [1].

    See Also
    --------
    choi_state : Function to compute the Choi state of the channel.
    channel_families.transposition : Function to obtain the transition matrix of matrix transposition.

    References
    ----------
    [1] : C. Hirche (2024), "Quantum Doeblin coefficients: A simple upper bound on contraction
    coefficients" (arXiv: 2405.00105)
    """

    d1, d2 = int(np.sqrt(channel.shape[1])), int(np.sqrt(channel.shape[0]))

    if not transpose:
        J = choi_state(channel)
    else:
        transpose = cf.transposition(d2)
        J = choi_state(transpose @ channel)

    # Restrict to the subspace projection if needed
    if subspace_projection is not None:
        J = np.kron(np.eye(d2), subspace_projection) @ J @ np.kron(np.eye(d2), subspace_projection)

    # Define the identity matrix I1 of dimension d1 or the subspace projection
    if subspace_projection is None:
        I1 = np.eye(d1)
    else:
        I1 = subspace_projection

    # Define the optimization variable sigma with shape (d2, d2)
    sigma = cp.Variable((d2, d2), PSD=True)

    # Define the objective function to maximize Tr(sigma)
    objective = cp.Maximize(cp.trace(sigma))

    # Define the constraint: kron(sigma, I1/d1) <= J
    kron_product = cp.kron(sigma, I1 / d1)
    constraints = [J - kron_product >> 0]

    # Formulate the problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem
    prob.solve()

    # Check if the problem is solved successfully
    if prob.status not in ["infeasible", "unbounded"]:
        return prob.value
    else:
        return None
