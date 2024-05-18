"""
predicates
==========

This module contains functions to check whether a given matrix is a density matrix 
or a quantum channel. 
"""

import numpy as np
from scipy.linalg import eigvalsh
from .channel_operations import choi_state

def is_channel(t: np.ndarray, tol: float = 1e-6, show: bool = False) -> bool:
    """
    Checks if the given matrix is a valid quantum channel.

    Parameters
    ----------
    t : np.ndarray
        The transition matrix representing the quantum channel.
    tol : float, optional
        Tolerance for the numerical checks. Defaults to 1e-6.
    show : bool, optional
        If True, prints the results of the various tests performed. Defaults to False.

    Returns
    -------
    bool
        True if the matrix is a valid quantum channel, False otherwise.

    Examples
    --------
    >>> from predicates import is_channel
    >>> from channel_families import depolarizing 
    >>> dim = 3
    >>> p = 0.5
    >>> tdepol = depolarizing(dim, p)
    >>> is_channel(tdepol)
    True
    >>> t = 2.5 * tdepol
    >>> result = is_channel(t, show = True)
    For the Choi matrix of the channel:
    Trace diff:  1.4999999999999996
    Hermiticity diff:  0.0
    Minimum eigval:  0.1388888888888888
    Channel is not trace preserving.
    >>> result
    False
    """
    choi = choi_state(t)

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


def is_density_matrix(dm: np.ndarray, tol: float = 1e-6, show: bool = False) -> bool:
    """
    Checks if the given matrix is a valid density matrix.

    Parameters
    ----------
    dm : np.ndarray
        The matrix to be checked.
    tol : float, optional
        Tolerance for the numerical checks. Defaults to 1e-6.
    show : bool, optional
        If True, prints the results of the various tests performed. Defaults to False.

    Returns
    -------
    bool
        True if the matrix is a valid density matrix, False otherwise.

    Examples
    --------
    >>> from predicates import is_density_matrix
    >>> rho = np.array([[0.7, 0], [0, 0.3]])
    >>> result = is_density_matrix(dm, show=True)
    Trace diff:  0.0
    Hermiticity diff:  0.0
    Minimum eigval:  0.3
    >>> result
    True
    >>> m = np.array([[1.3, 0], [0, -0.3]])
    >>> result = is_density_matrix(m, show=True)
    Trace diff:  0.0
    Hermiticity diff:  0.0
    Minimum eigval:  -0.3
    >>> result
    False
    """
    dtr = np.max(np.abs(np.trace(dm)-1))
    dhrm = np.max(np.abs(dm-dm.conj().transpose()))
    # This can crash the Kernel for states of dimension higher
    # than 150. It shouldnt, but it does.
    dpos = np.min(eigvalsh(dm, subset_by_index=[0, 1]))
    if show:
        print("Trace diff: ", dtr)
        print("Hermiticity diff: ", dhrm)
        print("Minimum eigval: ", dpos)
    return (dtr <= tol) and (dhrm <= tol) and (dpos >= - tol)