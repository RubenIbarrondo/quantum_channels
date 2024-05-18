import unittest
import numpy as np

class TestRandomGenerators(unittest.TestCase):

    def test_rg_state(self):
        from  src.pyqch.random_generators import state
        from  src.pyqch.predicates import is_density_matrix
        
        np.random.seed(1)
        dim = 3
        rank = dim

        # Assert basic structure
        num_instances = 5
        for instance in range(num_instances):
            rho = state(dim, rank)

            self.assertEqual(rho.shape, (dim, dim))
            self.assertEqual(rho.dtype, complex)

            self.assertTrue(is_density_matrix(rho))
        
        # Assert that rank 1 is a pure state
        rho = state(dim, 1)
        np.testing.assert_almost_equal(rho, rho @ rho)

        # Assert the rank of the state is ok
        num_instances = 5
        step = max([dim // num_instances, 1])  # upper bounds the number of rank instances to consider
        for rank in range(1, dim+1, step):
            rho = state(dim, rank)
            self.assertEqual(np.linalg.matrix_rank(rho, tol=1e-6, hermitian=True), rank)

    def test_dirichlet(self):
        from  src.pyqch.random_generators import state_dirichlet
        from  src.pyqch.predicates import is_density_matrix
        np.random.seed(1)
        dim = 3
        rank = dim
        
        rho = state_dirichlet(dim, rank)

        self.assertEqual(rho.shape, (dim, dim))
        self.assertEqual(rho.dtype, complex)

        self.assertTrue(is_density_matrix(rho))

    def test_rg_channel(self):
        from src.pyqch.random_generators import channel
        from src.pyqch.predicates import is_channel
        from src.pyqch.channel_operations import choi_state
        
        np.random.seed(1)
        dim = 3
        dim2 = dim + 1
        rank = dim

        # Assert value error when rank * dim_out < dim_in
        with self.assertRaises(ValueError):
            channel(dim+1, dim, 1)
        with self.assertRaises(ValueError):
            channel(2*dim, dim, 1)

        # Asserting output format, type and CPTP
        num_instances = 5
        for instance in range(num_instances):
            t = channel(dim, dim2, rank)

            self.assertEqual(t.shape, (dim2**2, dim**2))
            self.assertEqual(t.dtype, complex)
            self.assertTrue(is_channel(t))
        
        # Asserting rank = 1 produces a unitary evolution
        tu = channel(dim, dim, 1)
        np.testing.assert_almost_equal(tu @ tu.conj().T, np.identity(dim**2))

        # Assert the output channel has the desired Kraus rank
        num_instances = 5
        step = max([dim*dim2 // num_instances, 1])  # upper bounds the number of rank instances to consider
        for rank in range(1, dim*dim2+1, step):
            trank = channel(dim, dim2, rank)

            tau = choi_state(trank)
            self.assertEqual(np.linalg.matrix_rank(tau, tol=1e-6, hermitian=True), rank)

    def test_rg_unitary_channel(self):
        from  src.pyqch.random_generators import unitary_channel
        from  src.pyqch.predicates import is_channel
        np.random.seed(1)
        dim = 3
        rank = dim

        t = unitary_channel(dim)

        self.assertEqual(t.shape, (dim**2, dim**2))
        self.assertEqual(t.dtype, complex)

        self.assertTrue(is_channel(t))

        np.testing.assert_almost_equal(np.identity(dim), (t @ np.identity(dim).reshape(dim**2)).reshape((dim, dim)))
        
        np.testing.assert_almost_equal(np.identity(dim**2), (t @ t.T.conj()))


if __name__ == "__main__":
    unittest.main()

