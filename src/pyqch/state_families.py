"""
state_families
==============

Provides functions to generate quantum states belonging to 
specific families. Currently under development.
"""

import numpy as np

def computational_basis(dim: int, index: int) -> np.ndarray:
    """
    Generates a quantum state in the computational basis.

    Parameters
    ----------
    dim : int
        Dimension of the Hilbert space.
    index : int
        Index of the desired basis state.

    Returns
    -------
    np.ndarray
        The quantum state in the computational basis.
    """
    state = np.zeros(dim, dtype=complex)
    state[index] = 1
    return state