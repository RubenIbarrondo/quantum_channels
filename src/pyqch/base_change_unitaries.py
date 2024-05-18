"""
base_change_unitaries
=====================

This module implements functions that return unitaries for transforming 
density matrices between different bases. It currently includes transformations 
related to the generalized Gell-Mann basis and the element-wise (computational 
or canonical) basis, with potential for future extensions to other transformations.
"""

import numpy as np

def gm_el(dim: int) -> np.ndarray:
    """
    Returns the unitary matrix that transforms from the coefficients in the
    generalized Gell-Mann normalized basis to the element-wise (computational) basis.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.

    Returns
    -------
    np.array
        The unitary matrix for the basis transformation.

    See Also
    --------
    el_gm : Returns the inverse transformation.

    Notes
    -----
    Gell-Mann matrices are a generalization of the Pauli matrices used to 
    describe higher-dimensional systems in quantum mechanics [1]_. They form a 
    complete, orthogonal basis for the space of Hermitian matrices.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gell-Mann_matrices

    Examples
    --------
    >>> import numpy as np
    >>> from base_change_unitaries import gm_el
    >>> U_el_gm = el_gm(2)  # Transforms from the normalized Pauli basis
    >>> # Define a density matrix in the Pauli basis
    >>> rho_pauli = np.array([1/2, 1/2, 0, 0])
    >>> # Transform the density matrix to the element-wise basis
    >>> rho = (U_el_gm @ rho_pauli).reshape((2,2))
    >>> print("rho in element-wise basis:")
    >>> print(rho)
    [[ 0.5  0.5 ]
     [ 0.5  0.5 ]]
    """
    if dim == 2:
        return __get_pauli_unitary()
    elif dim == 3:
        return __get_gellmannd3_unitary()
    else:
        ugm = np.zeros((dim,)*4, dtype=complex)
        
        ugm[np.arange(dim), np.arange(dim), 0, 0] = 1 / np.sqrt(dim)
        
        for l in range(1, dim):
            for k in range(l):
                # x's
                ugm[k, l, k, l] = 1/np.sqrt(2)
                ugm[l, k, k, l] = 1/np.sqrt(2)

                # y's
                ugm[k, l, l, k] = -1j/np.sqrt(2)
                ugm[l, k, l, k] = 1j/np.sqrt(2)

                # z's
                ugm[l, l, l, l] = - l / np.sqrt(l*(l+1))
                ugm[np.arange(l), np.arange(l), l, l] = 1 / np.sqrt(l*(l+1))
                
        ugm = ugm.reshape((dim**2, dim**2))
        return ugm

def el_gm(dim: int) -> np.ndarray:
    """
    Returns the unitary matrix that transforms from the element-wise 
    (computational) basis to the coefficients in the generalized
    Gell-Mann normalized basis.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.

    Returns
    -------
    np.array
        The unitary matrix for the basis transformation.

    See Also
    --------
    gm_el : Returns the inverse transformation.

    Examples
    --------
    >>> import numpy as np
    >>> from base_change_unitaries import el_gm
    >>> dim = 3
    >>> U_el_gm = gm_el(dim)
    >>> # Define a density matrix in the computational basis
    >>> rho = np.diag([0.6, 0.3, 0.1])
    >>> # Transform the density matrix to the Gell-Mann basis
    >>> rho_gm = U_el_gm @ rho.reshape(dim*2)
    >>> print("rho in Gell-Mann basis:")
    >>> print(rho_gm)
    """
    return gm_el(dim).T.conj()

def __get_pauli_unitary():
    upauli = np.zeros((4, 4), dtype=complex)
    upauli[:, 0] = np.array([1,0,0,1])
    upauli[:, 1] = np.array([0,1,1,0])
    upauli[:, 2] = np.array([0,-1j,1j,0])
    upauli[:, 3] = np.array([1,0,0,-1])
    return upauli / np.sqrt(2)


def __get_gellmannd3_unitary():
    gm3 = np.zeros((9, 9), dtype=complex)
    gm3[:, 0] = np.identity(3).reshape(9) * np.sqrt(2/3)
    
    gm3[[1, 3], 1] = 1
    gm3[[1, 3], 2] = [-1j, 1j]
    gm3[:, 3] = np.diag([1,-1,0]).reshape(9)

    gm3[[2, 6], 4] = 1
    gm3[[2, 6], 5] = [-1j, 1j]

    gm3[[5, 7], 6] = 1
    gm3[[5, 7], 7] = [-1j, 1j]

    gm3[:, 8] = np.diag([1, 1, -2]).reshape(9)/np.sqrt(3)

    return gm3 / np.sqrt(2)