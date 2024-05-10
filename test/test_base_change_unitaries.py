import unittest
import numpy as np

class TestGellMann(unittest.TestCase):

    def test_paulis(self):
        from  src.pyqch.base_change_unitaries import gm_el
        # For dim 2 the unitaries should give the Pauli matrices
        pauli_ref = np.array([[[1,0],[0,1]],
                            [[0,1],[1,0]],
                            [[0, -1j],[1j, 0]],
                            [[1, 0],[0, -1]]
                            ]) / np.sqrt(2)
        pauli_ref = pauli_ref.transpose((1, 2, 0))
        dim = 2
        ugm = gm_el(dim)
        pauli_gm = ugm.reshape((2, 2, 4)) # Reshape to get matrices

        self.assertTrue(np.allclose(pauli_gm, pauli_ref))
    
    def test_inverse_relation(self):
        from  src.pyqch.base_change_unitaries import el_gm
        from  src.pyqch.base_change_unitaries import gm_el

        # For all dimensions, gm to el and el to gm shold be inverses
        max_dim = 10
        units = np.empty(max_dim-2, dtype=bool)
        for dim in range(2, max_dim):
            ugm = el_gm(dim)
            ugm_dag = gm_el(dim)

            units[dim-2] = np.allclose(ugm @ ugm_dag, np.identity(dim**2))
        self.assertTrue(all(units), msg=f"Unitarity failure in dimensions {np.arange(2, max_dim)[~units]}.")

    def test_traceless_hermitian(self):
        from  src.pyqch.base_change_unitaries import gm_el

        # For all dimensions, el to gm should give GellMann matrices
        # which are Hermitian and traceless (except for the first one which should be proportional to the identity)
        max_dim = 10
        hermiticity = np.empty(max_dim-2, dtype=bool)
        traceless = np.empty(max_dim-2, dtype=bool)
        for dim in range(2, max_dim):
            ugm = gm_el(dim).reshape((dim, dim, dim**2))
            
            hermiticity[dim-2] = np.allclose(ugm, ugm.transpose((1,0,2)).conj())
            trs = np.zeros(dim**2)
            trs[0] = np.sqrt(dim)
            traceless[dim-2] = np.allclose(np.trace(ugm, axis1=0, axis2=1), trs)

        self.assertTrue(all(hermiticity), msg=f"Hermiticity failure in dimensions {np.arange(2, max_dim)[~hermiticity]}.")
        self.assertTrue(all(traceless), msg=f"Traceless failure in dimensions {np.arange(2, max_dim)[~traceless]}.")

