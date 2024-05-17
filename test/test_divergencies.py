import unittest
import numpy as np

class TestDivergencies(unittest.TestCase):

    def _orthogonal_pure_states(self, dim):
        rho = np.zeros((dim, dim))
        rho[0,0] = 1
        sigma = np.zeros((dim, dim))
        sigma[1,1] = 1
        return rho, sigma
    
    def _pure_and_maximally_mixed(self, dim):
        rho = np.zeros((dim, dim))
        rho[0,0] = 1
        sigma = np.identity(dim) / dim
        return rho, sigma

    def _dense_states_and_rotated(self, dim):
        r = dim//2

        vrho = np.arange(1, 1+dim*r).reshape((dim, r)) 
        rho = vrho @ vrho.T.conj()
        rho = rho / np.trace(rho)

        vsigma =np.arange(dim*2*r, 0, -1).reshape((dim, 2*r))
        sigma = vsigma @ vsigma.T.conj()
        sigma = sigma / np.trace(sigma)

        u = np.zeros((dim, dim), dtype=complex)
        for i in range(dim//2):
            j = i+dim//2
            u[i, i] = 1 / np.sqrt(2)
            u[i, j] = 1 / np.sqrt(2)
            u[j, i] = 1 / np.sqrt(2)
            u[j, j] = - 1 / np.sqrt(2)
        if dim % 2 != 0:
            u[-1, -1] = np.exp(-1j * np.pi / 4)
        return rho, sigma, u @ rho @ u.T.conj(), u @ sigma @ u.T.conj()


    def test_relative_entropy(self):
        from  src.pyqch.divergencies import relative_entropy

        dim = 3

        rho, sigma = self._orthogonal_pure_states(dim)
        self.assertEqual(relative_entropy(rho, sigma), np.inf)

        rho, sigma = self._pure_and_maximally_mixed(dim)
        self.assertAlmostEqual(relative_entropy(rho, sigma), np.log2(dim))
        self.assertAlmostEqual(relative_entropy(sigma, rho), np.inf)

        rho, sigma, urho, usigma = self._dense_states_and_rotated(dim)
        self.assertAlmostEqual(relative_entropy(rho, sigma), relative_entropy(urho, usigma))

    def test_max_relative_entropy(self):
        from  src.pyqch.divergencies import max_relative_entropy

        dim = 3

        rho, sigma = self._orthogonal_pure_states(dim)
        self.assertEqual(max_relative_entropy(rho, sigma), np.inf)

        rho, sigma = self._pure_and_maximally_mixed(dim)
        self.assertAlmostEqual(max_relative_entropy(rho, sigma), np.log2(dim))
        self.assertAlmostEqual(max_relative_entropy(sigma, rho), np.inf)

        rho, sigma, urho, usigma = self._dense_states_and_rotated(dim)
        self.assertAlmostEqual(max_relative_entropy(rho, sigma), max_relative_entropy(urho, usigma))
    
    def test_trace(self):
        from  src.pyqch.divergencies import tr_dist

        dim = 3

        rho, sigma = self._orthogonal_pure_states(dim)
        self.assertAlmostEqual(tr_dist(rho, sigma), 1)

        rho, sigma = self._pure_and_maximally_mixed(dim)
        self.assertAlmostEqual(tr_dist(rho, sigma), 1-1/dim)
        self.assertAlmostEqual(tr_dist(sigma, rho), 1-1/dim)

        rho, sigma, urho, usigma = self._dense_states_and_rotated(dim)
        self.assertAlmostEqual(tr_dist(rho, sigma), tr_dist(urho, usigma))

    def test_hockey_stick(self):
        from  src.pyqch.divergencies import hockey_stick

        dim = 3
        gamma_arr = [.5/dim, 1., 2.]

        rho1, sigma1 = self._orthogonal_pure_states(dim)
        rho2, sigma2 = self._pure_and_maximally_mixed(dim)
        rho3, sigma3, urho3, usigma3 = self._dense_states_and_rotated(dim)

        for gamma in gamma_arr:

            self.assertAlmostEqual(hockey_stick(rho1, sigma1, gamma), 1)

            self.assertAlmostEqual(hockey_stick(rho2, sigma2, gamma), 1-gamma/dim)
            self.assertAlmostEqual(hockey_stick(sigma2, rho2, gamma), 1-1/dim + np.max([0, 1/dim-gamma]))

            self.assertAlmostEqual(hockey_stick(rho3, sigma3, gamma), hockey_stick(urho3, usigma3, gamma))
        
    def test_hs_dist(self):
        from src.pyqch.divergencies import hs_dist

        dim = 3

        rho, sigma = self._orthogonal_pure_states(dim)
        self.assertAlmostEqual(hs_dist(rho, sigma), np.sqrt(2))

        rho, sigma = self._pure_and_maximally_mixed(dim)
        self.assertAlmostEqual(hs_dist(rho, sigma), np.sqrt((1-1/dim)**2 + (dim-1)*(1/dim)**2))
        self.assertAlmostEqual(hs_dist(sigma, rho), np.sqrt((1-1/dim)**2 + (dim-1)*(1/dim)**2))

        rho, sigma, urho, usigma = self._dense_states_and_rotated(dim)
        self.assertAlmostEqual(hs_dist(rho, sigma), hs_dist(urho, usigma))
