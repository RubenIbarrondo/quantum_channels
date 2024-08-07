import unittest
import numpy as np

class TestStateFamilies(unittest.TestCase):

    def test_computational_basis_pure(self):
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
    

    def test_computational_basis_mixed(self):
        from  src.pyqch.state_families import computational_basis
        from  src.pyqch.predicates import is_density_matrix

        dim = 3
        index = dim//2

        psi = computational_basis(dim, index, as_density_matrix=True)

        self.assertEqual(psi.dtype, complex)
        self.assertEqual(psi.shape, (dim, dim))

        self.assertAlmostEqual(psi[index, index], 1)
        self.assertAlmostEqual(psi[0, 1], 0)

        self.assertTrue(is_density_matrix(psi))


    def test_maximally_entangled_pure(self):
        from  src.pyqch.state_families import maximally_entangled
        from  src.pyqch.predicates import is_density_matrix

        dim = 3

        omega = maximally_entangled(dim)

        self.assertEqual(omega.dtype, complex)
        self.assertEqual(omega.shape, (dim**2,))

        self.assertAlmostEqual(omega[0], 1/np.sqrt(dim))
        self.assertAlmostEqual(np.sum(np.abs(omega)**2), 1)

        rho = np.outer(omega, omega.T.conj())
        self.assertTrue(is_density_matrix(rho))
    

    def test_maximally_entangled_mixed(self):
        from  src.pyqch.state_families import maximally_entangled
        from  src.pyqch.predicates import is_density_matrix

        dim = 3

        omega = maximally_entangled(dim, as_density_matrix=True)

        self.assertEqual(omega.dtype, complex)
        self.assertEqual(omega.shape, (dim**2, dim**2))

        self.assertAlmostEqual(omega[0, 0], 1 / dim)
       
        self.assertTrue(is_density_matrix(omega))