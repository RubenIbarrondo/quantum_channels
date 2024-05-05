import unittest
import numpy as np

class TestRandomGenerators(unittest.TestCase):

    def test_rg_state(self):
        from src.random_generators.rg_state import rg_state
        from src.predicates.is_density_matrix import is_density_matrix
        
        np.random.seed(1)
        dim = 3
        rank = dim

        rho = rg_state(dim, rank)

        self.assertEqual(rho.shape, (dim, dim))
        self.assertEqual(rho.dtype, complex)

        self.assertTrue(is_density_matrix(rho))

    def test_dirichlet(self):
        from src.random_generators.rg_state_dirichlet import rg_state_dirichlet
        from src.predicates.is_density_matrix import is_density_matrix
        np.random.seed(1)
        dim = 3
        rank = dim

        rho = rg_state_dirichlet(dim, rank)

        self.assertEqual(rho.shape, (dim, dim))
        self.assertEqual(rho.dtype, complex)

        self.assertTrue(is_density_matrix(rho))

    def test_rg_channel(self):
        from src.random_generators.rg_channel import rg_channel
        from src.predicates.is_channel import is_channel
        np.random.seed(1)
        dim = 3
        rank = dim

        t = rg_channel(dim, rank)

        self.assertEqual(t.shape, (dim**2, dim**2))
        self.assertEqual(t.dtype, complex)

        self.assertTrue(is_channel(t))

    def test_rg_unitary_channel(self):
        from src.random_generators.rg_unitary_channel import rg_unitary_channel
        from src.predicates.is_channel import is_channel
        np.random.seed(1)
        dim = 3
        rank = dim

        t = rg_unitary_channel(dim)

        self.assertEqual(t.shape, (dim**2, dim**2))
        self.assertEqual(t.dtype, complex)

        self.assertTrue(is_channel(t))

        self.assertTrue(np.allclose(np.identity(dim),
                                    (t @ np.identity(dim).reshape(dim**2)).reshape((dim, dim))))
        
        self.assertTrue(np.allclose(np.identity(dim**2),
                                    (t @ t.T.conj())))


if __name__ == "__main__":
    unittest.main()

