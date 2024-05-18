"""
channel_operations
==================

This module implements various operations on quantum channels. It includes 
functions for obtaining the Choi state of a channel, performing tensor 
products of channel representations, and finding fixed points of a channel.
"""

import numpy as np
from scipy import linalg


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
