"""
state_families
==============

Provides functions to generate quantum states belonging to 
specific families. Currently under development.
"""

import numpy as np

def computational_basis(dim: int, index: int, as_density_matrix: bool = False) -> np.ndarray:
    """
    Generates a quantum state in the computational basis.

    Parameters
    ----------
    dim : int
        Dimension of the Hilbert space.
    index : int
        Index of the desired basis state.
    as_density_matrix: bool
        Whether the state is returned as a density matrix, by default False.

    Returns
    -------
    np.ndarray
        The quantum state in the computational basis.
    """
    if not as_density_matrix:
        state = np.zeros(dim, dtype=complex)
        state[index] = 1
        return state
    else:
        state = np.zeros((dim, dim), dtype=complex)
        state[index, index] = 1
        return state
    

def maximally_entangled(dim: int, as_density_matrix: bool = False) -> np.ndarray:
    """
    Generates a maximally entangled state in the computational basis.

    Parameters
    ----------
    dim : int
        Dimension of the Hilbert space.
    as_density_matrix : bool, optional
        Whether the state is returned as a density matrix, by default False.

    Returns
    -------
    np.ndarray
        The maximally entangled state.
    """
    
    if not as_density_matrix:
        state = np.identity(dim, dtype=complex).reshape((dim**2,)) / np.sqrt(dim)
        return state
    else:
        state_vec = maximally_entangled(dim, as_density_matrix=False)
        state = np.outer(state_vec, state_vec.conj())
        return state