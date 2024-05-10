import unittest
import numpy as np

class TestStateFamilies(unittest.TestCase):

    def test_computational_basis(self):
        from  src.pyqch.state_families import computational_basis
        from  src.pyqch.predicates import is_density_matrix

        dim = 3
        index = dim//2

        psi = computational_basis(dim, index)

        self.assertEqual(psi.dtype, complex)
        self.assertEqual(psi.shape, (dim,))

        self.assertAlmostEqual(psi[index], 1)
        self.assertAlmostEqual(np.sum(np.abs(psi)), 1)

        rho = np.outer(psi, psi.T.conj())
        self.assertTrue(is_density_matrix(rho))