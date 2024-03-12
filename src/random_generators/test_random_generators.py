import unittest
import numpy as np

class TestRandomGenerators(unittest.TestCase):

    def test_rg_state(self):
        from src.random_generators.rg_state import rg_state
        np.random.seed(1)
        dim = 3
        rank = dim

        rho = rg_state(dim, rank)

        self.assertAlmostEqual(np.trace(rho), 1)
        self.assertTrue(np.allclose(rho, rho.T.conj()))
        
        w = np.linalg.eigvalsh(rho)
        self.assertTrue(np.all(w >= 0))

    def test_dirichlet(self):
        from src.random_generators.rg_state_dirichlet import rg_state_dirichlet
        pass

    def test_rg_channel(self):
        from src.random_generators.rg_channel import rg_channel
        pass

    def test_rg_unitary_channel(self):
        from src.random_generators.rg_unitary_channel import rg_unitary_channel
        pass


if __name__ == "__main__":
    unittest.main()

