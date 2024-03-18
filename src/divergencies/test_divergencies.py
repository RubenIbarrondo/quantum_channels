import unittest
import numpy as np

class TestChannelFamilies(unittest.TestCase):

    def test_relative_entropy(self):
        from src.divergencies.d_relative_entropy import d_relative_entropy

        dim = 3

        rho = np.zeros((dim, dim))
        rho[0,0] = 1
        sigma = np.zeros((dim, dim))
        sigma[1,1] = 1

        self.assertEqual(d_relative_entropy(rho, sigma), np.inf)

        rho = np.zeros((dim, dim))
        rho[0,0] = 1
        sigma = np.identity(dim) / dim
        self.assertAlmostEqual(d_relative_entropy(rho, sigma), np.log2(dim))
        self.assertAlmostEqual(d_relative_entropy(sigma, rho), np.inf)

    def test_max_relative_entropy(self):
        from src.divergencies.d_max_relative_entropy import d_max_relative_entropy

        dim = 3

        rho = np.zeros((dim, dim))
        rho[0,0] = 1
        sigma = np.zeros((dim, dim))
        sigma[1,1] = 1

        self.assertEqual(d_max_relative_entropy(rho, sigma), np.inf)

        rho = np.zeros((dim, dim))
        rho[0,0] = 1
        sigma = np.identity(dim) / dim
        self.assertAlmostEqual(d_max_relative_entropy(rho, sigma), np.log2(dim))
        self.assertAlmostEqual(d_max_relative_entropy(sigma, rho), np.inf)