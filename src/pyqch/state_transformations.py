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


def partial_trace(state: np.ndarray, system: tuple[int], traced_sites: tuple[int] | int) -> np.ndarray:
    """
    Computes the partial trace on the state.

    Parameters
    ----------
    state : np.ndarray
        Density matrix of the state to be traced.
    system : tuple[int]
        The system structure represented by a tupe with the local
        dimension of the constituent subsystems.
    traced_sites : tuple[int] | int
        Site(s) in the system structure that will be traced out.

    Returns
    -------
    np.ndarray
        The density matrix of the state resulting from the partial trace.
    """
    
    # Other approach coul be to apply a permutation and then trace by biparititon
    subsystem_num = len(system)
    ptrstate = subsystem_reshape(state, system)

    if isinstance(traced_sites, int):
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
    Applies the channel to the satate in the chosen sites.

    Parameters
    ----------
    state : np.ndarray
        Density matrix of the state.
    system : tuple[int]
        The system structure represented by a tupe with the local
        dimension of the constituent subsystems.
    active_sites : tuple[int] | int
        The sites in which the channel acts on. If the channel's input and output dimensions
        are equal, the order of the sites is used to arrange the sites before applaying the channel
        and recover them back. If the channel's output dimension differs from the input active_sites collapse
        to a single site in the relative position correspoinding to active_sites[0].
    channel : np.ndarray
        The transition matrix of the quantum channel.

    Returns
    -------
    np.ndarray
        Density matrix of the state resulting from applyting the channel.
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