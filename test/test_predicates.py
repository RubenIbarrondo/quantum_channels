import unittest
import numpy as np

class TestPredicates(unittest.TestCase):

    def test_is_state(self):
        from  src.pyqch.predicates import is_density_matrix

        dim = 3
    
        # The pure state -> True
        rho = np.zeros((dim, dim))
        rho[0, 0] = 1
        self.assertTrue(is_density_matrix(rho))

        # A diagonal state -> True
        rho = np.diag(np.arange(1, 1+dim, dtype=float))
        rho /= np.trace(rho)
        self.assertTrue(is_density_matrix(rho))

        # An arbitrary dense state -> True
        r = dim//2
        vrho = np.arange(1, 1+dim*r, dtype=float).reshape((dim, r)) 
        rho = vrho @ vrho.T.conj()
        rho = rho / np.trace(rho)
        self.assertTrue(is_density_matrix(rho))

        # An operator that is not hermitian -> False
        A = np.arange((dim*dim), dtype=float).reshape((dim, dim))
        self.assertFalse(is_density_matrix(A))

        # An operator that is not positive -> False
        A = np.identity(dim)
        A[0, 0] = -1
        self.assertFalse(is_density_matrix(A))

        # An operator that is not trace 1 -> False
        A = np.diag(np.arange(1, 1+dim, dtype=float))
        self.assertFalse(is_density_matrix(A))


    def test_is_channel(self):
        from  src.pyqch.predicates import is_channel

        dim = 3
        id_mat = np.identity(dim)

        # The identity channel -> True
        T = np.identity(dim**2)
        self.assertTrue(is_channel(T))

        # A classical stochastic channel-> True
        stoch_mat = np.arange(1, 1 + dim**2, dtype=float).reshape((dim, dim))
        stoch_mat /= np.sum(stoch_mat, axis=0)
        T = (np.einsum("ij,pq,pi->pqij", id_mat, id_mat, stoch_mat)).reshape((dim**2, dim**2))
        self.assertTrue(is_channel(T))

        # A dephasing channel -> True
        g = .5
        t1 = np.einsum("pq,pi,qj->pqij", id_mat, id_mat, id_mat)
        t2 = np.einsum("pi,qj->pqij", id_mat, id_mat)
        T =  ((1-g) * t1 + g * t2).reshape((dim**2, dim**2))
        self.assertTrue(is_channel(T))

        # An operator that is not hermiticity preserving -> False
        A = np.identity(dim**2, dtype=complex)
        A[1,1] = 1j
        self.assertFalse(is_channel(A))

        # An operator that is not positivity preserving -> False
        A = np.identity(dim**2, dtype=complex)
        A[0,0] = -1
        self.assertFalse(is_channel(A))

        # An operator that is not trace preserving -> False
        A = np.identity(dim**2, dtype=complex)
        A[0,0] = 0
        self.assertFalse(is_channel(A))