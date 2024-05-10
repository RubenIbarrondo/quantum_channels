from .gm_el import gm_el

def el_gm(dim: int):
    return gm_el(dim).T.conj()

