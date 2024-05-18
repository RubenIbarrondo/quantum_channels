"""
divergences
===========

This module implements various quantum divergences, which are bivariate functions 
that assign a dissimilarity measure to quantum states.
"""

import numpy as np

def hockey_stick(rho: np.ndarray, sigma: np.ndarray, gamma: float) -> np.float64:
    """
    Computes the hockey-stick divergence between two quantum states.

    Parameters
    ----------
    rho : np.ndarray
        The first quantum state (density matrix).
    sigma : np.ndarray
        The second quantum state (density matrix).
    gamma : float
        The parameter gamma used in the divergence calculation.

    Returns
    -------
    np.float64
        The hockey-stick divergence between the two states.

    Examples
    --------
    >>> rho = np.array([[0.5, 0], [0, 0.5]])
    >>> sigma = np.array([[0.4, 0.1], [0.1, 0.6]])
    >>> gamma = 1.25
    >>> divergence = hockey_stick(rho, sigma, gamma)
    0.051776695296636865
    """
    return 1/2 * np.linalg.norm(rho- gamma * sigma, 'nuc')+1/2*np.trace(rho-gamma*sigma)


def hs_dist(rho: np.ndarray, sigma: np.ndarray) -> np.float64:
    """
    Computes the Hilbert-Schmidt distance between two quantum states.

    Parameters
    ----------
    rho : np.ndarray
        The first quantum state (density matrix).
    sigma : np.ndarray
        The second quantum state (density matrix).

    Returns
    -------
    np.float64
        The Hilbert-Schmidt distance between the two states.

    Examples
    --------
    >>> rho = np.array([[0.5, 0], [0, 0.5]])
    >>> sigma = np.array([[0.4, 0.1], [0.1, 0.6]])
    >>> hs_dist(rho, sigma)
    0.19999999999999998
    """
    return np.linalg.norm(rho-sigma, 'fro')


def max_relative_entropy(rho: np.ndarray, sigma: np.ndarray, basis: int = 2) -> np.float64:
    """
    Computes the max-relative entropy between two quantum states.

    Parameters
    ----------
    rho : np.ndarray
        The first quantum state (density matrix).
    sigma : np.ndarray
        The second quantum state (density matrix).
    basis : int, optional
        The logarithmic base for the entropy calculation. Defaults to 2.

    Returns
    -------
    np.float64
        The max-relative entropy between the two states.

    Examples
    --------
    >>> rho = np.array([[0.5, 0], [0, 0.5]])
    >>> sigma = np.array([[0.4, 0.1], [0.1, 0.6]])
    >>> max_relative_entropy(rho, sigma)
    0.4796385281957086
    """
    tol = 1e-7

    # Check support condition
    rvals, rvecs = np.linalg.eigh(rho)
    if any(abs(rvals.imag) >= tol):
        raise ValueError("Input rho has non-real eigenvalues.")
    rvals = rvals.real
    svals, svecs = np.linalg.eigh(sigma)
    if any(abs(svals.imag) >= tol):
        raise ValueError("Input sigma has non-real eigenvalues.")
    svals = svals.real

    # Calculate inner products of eigenvectors and return +inf if kernel
    # of sigma overlaps with support of rho.
    P = abs(rvecs.T @ svecs.conj()) ** 2
    if (rvals >= tol) @ (P >= tol) @ (svals < tol):
        return np.inf
    
    # If true
    s = np.log2(np.max(np.linalg.eigvals(np.linalg.pinv(sigma) @ rho))) / np.log2(basis)
    return s


def relative_entropy(rho: np.ndarray, sigma: np.ndarray, basis: int = 2) -> np.float64:
    """
    Computes the relative entropy (Kullback-Leibler divergence) between two quantum states.

    Parameters
    ----------
    rho : np.ndarray
        The first quantum state (density matrix).
    sigma : np.ndarray
        The second quantum state (density matrix).
    basis : int, optional
        The logarithmic base for the entropy calculation. Defaults to 2.

    Returns
    -------
    np.float64
        The relative entropy between the two states.

    Examples
    --------
    >>> rho = np.array([[0.5, 0], [0, 0.5]])
    >>> sigma = np.array([[0.4, 0.1], [0.1, 0.6]])
    >>> entropy = relative_entropy(rho, sigma)
    0.060147116858855876
    """
    tol = 1e-7
    # S(rho || sigma) = sum_i(p_i log p_i) - sum_ij(p_i P_ij log q_i)
    #
    # S is +inf if the kernel of sigma (i.e. svecs[svals == 0]) has non-trivial
    # intersection with the support of rho (i.e. rvecs[rvals != 0]).
    rvals, rvecs = np.linalg.eigh(rho)
    if any(abs(rvals.imag) >= tol):
        raise ValueError("Input rho has non-real eigenvalues.")
    rvals = rvals.real
    svals, svecs = np.linalg.eigh(sigma)
    if any(abs(svals.imag) >= tol):
        raise ValueError("Input sigma has non-real eigenvalues.")
    svals = svals.real

    # Calculate inner products of eigenvectors and return +inf if kernel
    # of sigma overlaps with support of rho.
    P = abs(rvecs.T @ svecs.conj()) ** 2
    if (rvals >= tol) @ (P >= tol) @ (svals < tol):
        return np.inf
    # Avoid -inf from log(0) -- these terms will be multiplied by zero later
    # anyway
    svals[abs(svals) < tol] = 1
    nzrvals = rvals[abs(rvals) >= tol]
    # Calculate S
    S = (nzrvals @ np.log2(nzrvals) - rvals @ P @ np.log2(svals) )/  np.log2(basis)
    # the relative entropy is guaranteed to be >= 0, so we clamp the
    # calculated value to 0 to avoid small violations of the lower bound.
    return max(0, S)


def tr_dist(rho: np.ndarray, sigma: np.ndarray) -> np.float64:
    """
    Computes the trace distance between two quantum states.

    Parameters
    ----------
    rho : np.ndarray
        The first quantum state (density matrix).
    sigma : np.ndarray
        The second quantum state (density matrix).

    Returns
    -------
    np.float64
        The trace distance between the two states.

    Examples
    --------
    >>> rho = np.array([[0.5, 0], [0, 0.5]])
    >>> sigma = np.array([[0.4, 0.1], [0.1, 0.6]])
    >>> tr_dist(rho, sigma)
    0.1414213562373095
    """
    return np.linalg.norm(rho - sigma, 'nuc') / 2

