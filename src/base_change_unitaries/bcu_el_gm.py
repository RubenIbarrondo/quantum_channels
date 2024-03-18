from src.base_change_unitaries.bcu_gm_el import bcu_gm_el

def bcu_el_gm(dim: int):
    return bcu_gm_el(dim).T.conj()

