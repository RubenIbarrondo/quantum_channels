import unittest
import numpy as np

class TestStateFamilies(unittest.TestCase):

    def test_computational_basis(self):
        from src.state_families.sf_computational_basis import sf_computational_basis
        from src.predicates.is_density_matrix import is_density_matrix

        dim = 3
        index = dim//2

        psi = sf_computational_basis(dim, index)

        self.assertEqual(psi.dtype, complex)
        self.assertEqual(psi.shape, (dim,))

        self.assertAlmostEqual(psi[index], 1)
        self.assertAlmostEqual(np.sum(np.abs(psi)), 1)

        rho = np.outer(psi, psi.T.conj())
        self.assertTrue(is_density_matrix(rho))