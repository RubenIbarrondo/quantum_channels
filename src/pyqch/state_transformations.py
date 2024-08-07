"""
state_transformations
==========

This module contains functions to directly transform density matrices. Such as,
subsystem permutations or applying a quantum channel in a subsystem without 
having to build the transfer matrix for the whole system. 
"""

import numpy as np
from .predicates import is_system_compatible

def subsystem_reshape(state: np.ndarray, system: tuple[int]) -> np.ndarray:
    """
    Reshapes the density matrix to the given system structure.

    Parameters
    ----------
    state : np.ndarray
        The denstity matrix to be reshaped.
    system : tuple[int]
        The system structure represented by a tupe with the local
        dimension of the constituent subsystems.

    Returns
    -------
    np.ndarray
        The reshaped density matrix.

    Raises
    ------
    ValueError
        If state and system are not compatible.
    
    """
    try:
        state = state.reshape(system + system)
        return state
    except ValueError as ve:
        if not is_system_compatible(state, system):
            raise ValueError(f"state with shape {state.shape} is uncompatible with system with structure {system}")
        else:
            raise ve

def subsystem_permutation(state: np.ndarray, system: tuple[int], permutation: tuple[int], inverse: bool = False) -> np.ndarray:
    """
    Performs the permutation of the subsystems on the state.

    Parameters
    ----------
    state : np.ndarray
        The density matrix of the state.
    system : tuple[int]
        The system structure represented by a tupe with the local
        dimension of the constituent subsystems.
    permutation : tuple[int]
        The permutation that maps i to permutation[i].
    inverse : bool
        Whether to perform the inverse of the permutation.

    Returns
    -------
    np.ndarray
        The density matrix of the satate after applying the permutation.

    Raises
    ------
    ValueError
        If the system and permutation are incompatible.
    """
    dim = state.shape[0]
    
    if len(system) != len(permutation):
        raise ValueError(f"invalid permutation {permutation} for system structure {system}.")
    
    state = subsystem_reshape(state, system)
    subsystem_num = len(system)
    
    if not inverse:
        perm = [0] * len(permutation)
        for i in range(len(permutation)):
            perm[permutation[i]] = i
        perm = tuple(perm)
    else:
        perm = permutation

    state = np.transpose(state, perm + tuple(p+subsystem_num for p in perm))
    
    return state.reshape((dim, dim))


def partial_trace(state: np.ndarray, system: tuple[int], traced_sites: tuple[int] | int, keep_sites: bool = False) -> np.ndarray:
    """
    Computes the partial trace on the state.

    Parameters
    ----------
    state : np.ndarray
        Density matrix of the state to be traced.
    system : tuple[int]
        The system structure represented by a tuple with the local
        dimension of the constituent subsystems.
    traced_sites : tuple[int] | int
        Site(s) in the system structure that will be traced out.
    keep_sites : bool (optional)
        If True, `traced_sites` specifies the subsystems to keep rather than trace out. Defaults to False.

    Returns
    -------
    np.ndarray
        The density matrix of the state resulting from the partial trace.
    """
    if keep_sites:
        if isinstance(traced_sites, (int, np.integer)):
            new_traced_out = list(range(len(system)))
            del new_traced_out[traced_sites]
            new_traced_out = tuple(new_traced_out)
        else:
            new_traced_out = tuple([i for i in range(len(system)) if i not in traced_sites])
        return partial_trace(state, system, new_traced_out, keep_sites=False)
    else:
        # Other approach could be to apply a permutation and then trace by bipartition
        subsystem_num = len(system)
        ptrstate = subsystem_reshape(state, system)

        if isinstance(traced_sites, (int, np.integer)):
            ptrstate = np.trace(ptrstate, axis1=traced_sites, axis2=traced_sites+subsystem_num)
            new_dim = np.prod([s for i_s, s in enumerate(system) if i_s != traced_sites])
        else:
            for tr_site in sorted(traced_sites, reverse=True):
                ptrstate = np.trace(ptrstate, axis1=tr_site, axis2=tr_site+subsystem_num)
                subsystem_num -= 1
        
            new_dim = np.prod([s for i_s, s in enumerate(system) if i_s not in traced_sites])
        return ptrstate.reshape((new_dim, new_dim))


def local_channel(state: np.ndarray, system: tuple[int], active_sites: tuple[int] | int, channel: np.ndarray) -> np.ndarray:
    """
    Applies the channel to the state in the chosen sites.

    Parameters
    ----------
    state : np.ndarray
        Density matrix of the state.
    system : tuple[int]
        The system structure represented by a tuple with the local
        dimension of the constituent subsystems.
    active_sites : tuple[int] | int
        The sites in which the channel acts on. If the channel's input and output dimensions
        are equal, the order of the sites is used to arrange the sites before applying the channel
        and recover them back. If the channel's output dimension differs from the input active_sites collapse
        to a single site in the relative position corresponding to active_sites[0].
    channel : np.ndarray
        The transition matrix of the quantum channel.

    Returns
    -------
    np.ndarray
        Density matrix of the state resulting from applying the channel.
    """
    
    is_square_channel = channel.shape[0] == channel.shape[1]

    if isinstance(active_sites, int):
        active_sites = (active_sites,)
  
    # Apply a permutation to the state so that the idle subsystems are in the initial
    # position and the active sites are ordered as in active_sites
    idle = tuple(i_s for i_s in range(len(system)) if i_s not in active_sites)
    inv_perm = idle + active_sites  # inverse of (idle | active ordered as in active_sites)

    state = subsystem_permutation(state, system, idle + active_sites, inverse=True)
    
    # Get an effective bipartition between the sites that are idle and the
    # sites where the channel acts
    bipartition = (int(np.prod([system[i] for i in idle])),
                   int(np.prod([system[i] for i in active_sites])))
    rstate = subsystem_reshape(state, bipartition)

    # Acting with the channel only on the second subsystem
    dout = int(np.sqrt(channel.shape[0]))
    din = int(np.sqrt(channel.shape[1]))
    rchannel = np.reshape(channel, (dout, dout, din, din))
    t_rstate = np.einsum("ijnm,pnrm->pirj", rchannel, rstate)

    new_dim_idle = t_rstate.shape[0]
    new_dim_active = t_rstate.shape[1]
    t_state = t_rstate.reshape((new_dim_idle*new_dim_active, )*2)
    
    idle_system = tuple(system[i] for i in idle)
    if is_square_channel:
        recovering_permutation = inv_perm
        active_system = tuple(system[i] for i in active_sites)
    else:
        
        new_idle_pre_active = tuple(i for i, idl in enumerate(idle) if (idl < active_sites[0]))
        new_active_first = len(new_idle_pre_active)
        new_idle_post_active = tuple(i + 1 for i, idl in enumerate(idle) if (idl > active_sites[0]))
        recovering_permutation = new_idle_pre_active + new_idle_post_active + (new_active_first,)
        active_system = (new_dim_active,)

    t_state = subsystem_permutation(t_state, idle_system + active_system, recovering_permutation)
    return t_state

def twirling(state: np.ndarray,
             r: list[np.ndarray] = None,
             decomposition: list[np.ndarray] = None):
    if r is not None:
        # r representation of a finite group with each element is labeled by an integer
        # There should be a more efficient way...
        state_twirl = np.mean([r[g] @ state @ r[g].conj().T for g in range(len(r))], axis=0)

    elif decomposition is not None:
        # decomposition = [W, d_arr, m_arr]
        W, d_arr, m_arr = decomposition
        
        # W: is a unitary transformation or 1, standing for the identity.
        assert (isinstance(W, int) and W == 1) or np.allclose(W @ W.T.conj(), np.identity(W.shape[0]))
        
        # d_arr are the dimensions of the subsystems of the subspaces where the group acts non-trivially
        # m_arr are the dimension where they act trivially
        # For consistency
        assert d_arr.dtype == int and m_arr.dtype == int
        assert len(d_arr) == len(m_arr)
        assert (isinstance(W, int) and W == 1) or (d_arr @ m_arr == W.shape[0])

        # Map the state to the basis of the decomposition
        if not isinstance(W, int):
            state = W.T.conj() @ state @ W
        
        # Go blockwise
        state_twirl = np.zeros_like(state, dtype=complex)
        for i, dimi in enumerate(zip(d_arr, m_arr)):
            di, mi = dimi
            di_less = d_arr[:i] @ m_arr[:i] if i > 0 else 0

            # Get the state corresponding to the i-th block
            rhoi = state[di_less: di_less + di * mi,
                         di_less: di_less + di * mi]

            # Get the reduced state of the part that is invariant
            rho_mi = partial_trace(rhoi, (di, mi), 0)

            # Get the full state of the site by the tensor product with the identity / di
            for j_di in range(di):
                state_twirl[di_less + j_di * mi: di_less + (j_di + 1) * mi,
                            di_less + j_di * mi: di_less + (j_di + 1) * mi] = rho_mi / di
        
        # Back to the original basis
        if not isinstance(W, int):
            state_twirl = W @ state_twirl @ W.T.conj()
    else:
        raise ValueError("Either r or decomposition have to be provided, but both were None.")
    
    return state_twirl