"""
random_generators
=================

This module contains functions to generate random quantum channels and states by sampling from various distributions.
"""

import numpy as np
from scipy.stats import unitary_group, dirichlet


def channel(dim_in : int, dim_out : int = None, kraus_rank : int = None, random_state: int | np.random.Generator = None) -> np.ndarray:
    """
    Generates a random quantum channel by generating a random isometry 
    (truncated Haar-random unitary) and tracing out the environment.

    Parameters
    ----------
    dim_in : int
        Input dimension of the quantum channel.
    dim_out : int, optional
        Output dimension of the quantum channel. If None, defaults to dim_in.
    kraus_rank : int, optional
        Rank of the Kraus operators representing the quantum channel. If None, defaults to dim_in * dim_out.
    random_state: int | np.random.Generator, optional
        Used for drawing random variates. If seed is None, the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded with seed. If seed is 
        already a RandomState or Generator instance, then that object is used. Default is None.

    Returns
    -------
    np.ndarray
        The transition matrix representing the random quantum channel.

    Examples
    --------
    >>> from random_generators import channel
    >>> dim_in, dim_out = 2, 3
    >>> T = channel(dim_in, dim_out)
    >>> T.shape
    (9, 4)
    """
    if random_state is not None:
        random_state = np.random.default_rng(random_state)
    
    if dim_out is None:
        dim_out = dim_in
    if kraus_rank is None:
        kraus_rank = dim_out * dim_in

    if dim_out * kraus_rank < dim_in:
        raise ValueError(f"Channel specifications must satisfy dim_out * kraus_rank >= dim_in, but {dim_out} * {kraus_rank} < {dim_in}.")
    
    # Generate a random isometry from dim_in to dim_out * kraus_rank
    # 1) Create a random unitary of dimension dim_out * kraus_rank
    # 2) Take only dim_in columns, i.e. setting the initial state of the environment to 0
    v = unitary_group.rvs(dim_out * kraus_rank, random_state=random_state)[:, :dim_in]
    v = np.reshape(v, (dim_out, kraus_rank, dim_in))

    # Taking products and tracing out the final environment with dimension kraus_rank
    t_astensor = np.einsum("nik,mil->nmkl" , v, v.conj())
    return np.reshape(t_astensor, (dim_out**2, dim_in**2))


def state_dirichlet(dim : int, alpha : float, random_state: int | np.random.Generator = None) -> np.ndarray:
    """
    Generates random quantum states by sampling their spectrum from the homogeneous Dirichlet distribution
    with concentration parameter alpha. The basis are rotated with Haar random unitary matrices.

    Parameters
    ----------
    dim : int
        Dimension of the quantum state.
    alpha : float
        Concentration parameter of the Dirichlet distribution.
    random_state: int | np.random.Generator, optional
        Used for drawing random variates. If seed is None, the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded with seed. If seed is 
        already a RandomState or Generator instance, then that object is used. Default is None.

    Returns
    -------
    np.ndarray
        The random quantum state.

    Note
    ----
    As alpha -> 0 generated states are more pure, alpha = 1 is
    equivalent to a Hilbert-Schmidt uniform measure, and as 
    alpha -> infty it concentrates around the maximally mixed 
    state.

    Examples
    --------
    >>> from random_generators import state_dirichlet
    >>> dim = 2
    >>> alpha = 0.5
    >>> rho = state_dirichlet(dim, alpha)
    >>> print(rho)
    [[ 0.33869694+0.j        -0.24593237-0.4043495j]
     [-0.24593237+0.4043495j  0.66130306+0.j       ]]
    """
    if random_state is not None:
        random_state = np.random.default_rng(random_state)

    spec = dirichlet.rvs([alpha]*dim, size=1, random_state=random_state)[0]
    u = unitary_group.rvs(dim, random_state=random_state)
    rho = u @ np.diag(spec) @ u.T.conj()
    return rho


def state(dim: int, rank: int = None, random_state: int | np.random.Generator = None) -> np.ndarray:
    """
    Generates a random quantum state by sampling Haar random
    pure states on dimension dim * rank and computing the
    partiual trace on rank degrees of freedom.

    Parameters
    ----------
    dim : int
        Dimension of the quantum state.
    rank : int, optional
        Rank of the state. If None, defaults to dim.
    random_state: int | np.random.Generator, optional
        Used for drawing random variates. If seed is None, the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded with seed. If seed is 
        already a RandomState or Generator instance, then that object is used. Default is None.

    Returns
    -------
    np.ndarray
        The random quantum state.

    Examples
    --------
    >>> from random_generators import state
    >>> dim = 2
    >>> rank = 1
    >>> rho = state(dim, rank)
    >>> print(rho)
    [[ 0.7984795 +3.77722651e-19j -0.21614672+3.37920970e-01j]
     [-0.21614672-3.37920970e-01j  0.2015205 -3.08140272e-18j]]
    """
    if random_state is not None:
        random_state = np.random.default_rng(random_state)
    
    if rank is None:
        rank = dim
        
    u = unitary_group.rvs(dim * rank, random_state=random_state)
    purification = np.outer(u[0, :], u[0,:].conjugate())
    return np.trace(np.reshape(purification, (dim, rank, dim, rank)), axis1=1, axis2=3)


def unitary_channel(dim: int, random_state: int | np.random.Generator = None) -> np.ndarray:
    """
    Generates a random unitary quantum channel.

    Parameters
    ----------
    dim : int
        Dimension of the Holbert space.
    random_state: int | np.random.Generator, optional
        Used for drawing random variates. If seed is None, the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded with seed. If seed is 
        already a RandomState or Generator instance, then that object is used. Default is None.

    Returns
    -------
    np.ndarray
        The transition matrix representing the random unitary quantum channel.

    Examples
    --------
    >>> from random_generators import unitary_channel
    >>> dim = 2
    >>> U = unitary_channel(dim)
    >>> U.shape
    (4, 4)
    >>> np.allclose(U @ U.T.conj(), np.identity(dim**2))
    True
    """
    if random_state is not None:
        random_state = np.random.default_rng(random_state)

    u = unitary_group.rvs(dim, random_state=random_state)
    return np.kron(u, u.conjugate())