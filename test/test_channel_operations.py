import unittest
import numpy as np

class TestChannelOperations(unittest.TestCase):

    def _unitary_channel(self, dim):
        u = np.zeros((dim, dim), dtype=complex)
        for i in range(dim//2):
            j = i+dim//2
            u[i, i] = 1 / np.sqrt(2)
            u[i, j] = 1 / np.sqrt(2)
            u[j, i] = 1 / np.sqrt(2)
            u[j, j] = - 1 / np.sqrt(2)
        if dim % 2 != 0:
            u[-1, -1] = np.exp(-1j * np.pi / 4)

        return np.kron(u, u.conj())
    
    def _replacer_channel_and_state(self, dim):
        return self._replacer_channel_and_state_diffdim(dim, dim)
    
    def _replacer_channel_and_state_diffdim(self, dim1, dim2):
        rho = np.diag(np.arange(1, dim2+1, dtype=complex))
        rho /= np.trace(rho)

        t = np.outer(rho.reshape(dim2**2), np.identity(dim1).reshape(dim1**2))
        return t, rho
    
    def _dephasing_channel(self, dim):
        id_mat = np.identity(dim)
        return np.einsum("pq,pi,qj->pqij", id_mat, id_mat, id_mat).reshape((dim**2, dim**2))
    
    def _depolarizing_channel(self, dim, p):
        t = np.outer(np.identity(dim).reshape(dim**2)/dim, np.identity(dim).reshape(dim**2))
        return (1-p) * np.identity(dim**2) + p * t

    def test_choi_state(self):
        from  src.pyqch.channel_operations import choi_state

        dim = 3
        dim2 = dim+1

        # Unitary channel -> has to be a pure and maximally entangled state
        tu = self._unitary_channel(dim)
        choi_tu = choi_state(tu)

        self.assertEqual(choi_tu.dtype, tu.dtype)
        self.assertEqual(choi_tu.shape, (dim**2, dim**2))

        np.testing.assert_almost_equal(choi_tu, choi_tu @ choi_tu)
        np.testing.assert_almost_equal(np.identity(dim) / dim,
                                    np.trace(choi_tu.reshape((dim,)*4), axis1=1, axis2=3))

        # Replacer channel -> has to be the fixed state times the maximally mixed
        tr, rho = self._replacer_channel_and_state(dim)
        choi_tr = choi_state(tr)

        self.assertEqual(choi_tr.dtype, tr.dtype)
        self.assertEqual(choi_tr.shape, (dim**2, dim**2))

        np.testing.assert_almost_equal(choi_tr,
                                    np.kron(rho, np.identity(dim)/dim))
        
        # Replacer channel d2!=d1 -> has to be the fixed state times the maximally mixed
        trdiff, rhodiff = self._replacer_channel_and_state_diffdim(dim, dim2)
        choi_trdiff = choi_state(trdiff)

        self.assertEqual(choi_trdiff.dtype, trdiff.dtype)
        self.assertEqual(choi_trdiff.shape, (dim*dim2, dim*dim2))

        np.testing.assert_almost_equal(choi_trdiff,
                                    np.kron(rhodiff, np.identity(dim)/dim))

        # Dephasing channel -> has to be a maximally correlated state
        tdeph = self._dephasing_channel(dim)
        choi_tdeph = choi_state(tdeph)

        self.assertEqual(choi_tdeph.dtype, tdeph.dtype)
        self.assertEqual(choi_tdeph.shape, (dim**2, dim**2))

        correlated_state = np.zeros((dim, dim, dim, dim))
        for k in range(dim):
            correlated_state[k,k,k,k]=1/dim
        correlated_state = correlated_state.reshape((dim**2, dim**2))

        np.testing.assert_almost_equal(choi_tdeph,
                                    correlated_state)

        # Depolarizing channel -> has to be the corresponding convex combination
        p =.5
        tdepol = self._depolarizing_channel(dim, p)
        choi_tdepol = choi_state(tdepol)

        self.assertEqual(choi_tdepol.dtype, tdepol.dtype)
        self.assertEqual(choi_tdepol.shape, (dim**2, dim**2))
        
        max_entang = np.identity(dim).reshape(dim**2) / np.sqrt(dim)
    
        np.testing.assert_almost_equal(choi_tdepol,
                                    p * np.identity(dim**2)/dim**2 + (1-p) * np.outer(max_entang, max_entang.T.conj()))
        

    def test_fixed_points(self):
        from  src.pyqch.channel_operations import fixed_points

        dim = 3
        dim2 = dim + 1
        # Unitary channel -> raise an error due to multiple-fixed points not being implemented
        tu = self._unitary_channel(dim)

        with self.assertRaises(NotImplementedError):
            fixed_points(tu)

        # Replacer channel -> has to be the fixed state
        tr1, rho1 = self._replacer_channel_and_state(dim)

        fp_tr1 = fixed_points(tr1)

        self.assertEqual(fp_tr1.dtype, complex)
        self.assertEqual(fp_tr1.shape, (dim, dim))
        np.testing.assert_almost_equal(fp_tr1, rho1)

        tr2, _ = self._replacer_channel_and_state_diffdim(dim, dim2)

        with self.assertRaises(ValueError):
            fixed_points(tr2)

        # Dephasing channel -> raise an error due to multiple-fixed points not being implemented
        tdephas = self._dephasing_channel(dim)

        with self.assertRaises(NotImplementedError):
            fixed_points(tdephas)

        # Depolarizing channel -> has to be the maximally mixed state
        p = .5
        tdepol = self._depolarizing_channel(dim, p)
        fp_depol = fixed_points(tdepol)
        
        self.assertEqual(np.real_if_close(fp_depol).dtype, float)
        self.assertEqual(fp_depol.shape, (dim, dim))
        np.testing.assert_almost_equal(fp_depol, np.identity(dim)/dim)

        # a * Identity with a!=1 -> raise an error becaouse it doesn't have a fixed point
        tamply = 2 * np.identity(dim**2)

        with self.assertRaises(ValueError):
            fixed_points(tamply)
        
    def test_tensor(self):
        from  src.pyqch.channel_operations import tensor
        dim = 3
        dim2 = dim + 1

        # Product of Unitary channels -> again unitary
        tu = self._unitary_channel(dim)
        tensor_tu = tensor(tu, 2)

        self.assertEqual(tensor_tu.shape, (dim**4, dim**4))
        self.assertEqual(tensor_tu.dtype, tu.dtype)

        np.testing.assert_almost_equal(np.identity(dim**2),
                                    (tensor_tu @ np.identity(dim**2).reshape(dim**4)).reshape((dim**2, dim**2)))
        
        np.testing.assert_almost_equal(np.identity(dim**4),
                                    (tensor_tu @ tensor_tu.T.conj()))
        np.testing.assert_almost_equal(tensor_tu, (np.transpose(tensor_tu.reshape((dim,)*(2*4)), (1,0,3,2, 5,4,7,6))).reshape((dim**4, dim**4)))

        # Product of Replacer channels -> also a replacer towards the tensor of fixed states
        tr, rho = self._replacer_channel_and_state(dim)

        tensor_tr = tensor(tr, 2)
        self.assertEqual(tensor_tr.shape, (dim**4, dim**4))
        self.assertEqual(tensor_tr.dtype, tr.dtype)

        np.testing.assert_almost_equal(tensor_tr,
                                    np.outer(np.kron(rho, rho).reshape(dim**4), np.identity(dim**2).reshape(dim**4)))

        # Product of Replacer channels diffdim-> also a replacer towards the tensor of fixed states
        tr2, rho2 = self._replacer_channel_and_state_diffdim(dim, dim2)

        tensor_tr2 = tensor([tr, tr2])
        self.assertEqual(tensor_tr2.shape, (dim**2*dim2**2, dim**4))
        self.assertEqual(tensor_tr2.dtype, (tr[0,0]*tr2[0,0]).dtype)

        np.testing.assert_almost_equal(tensor_tr2,
                                    np.outer(np.kron(rho, rho2).reshape(dim**2*dim2**2), np.identity(dim**2).reshape(dim**4)))

        # Product of Dephasing channels -> dephasing in the larger system
        tdephas = self._dephasing_channel(dim)
        tdephas2 = self._dephasing_channel(dim**2)
        tensor_tdephas = tensor(tdephas, 2)

        self.assertEqual(tensor_tdephas.shape, (dim**4, dim**4))
        self.assertEqual(tensor_tdephas.dtype, tdephas.dtype)

        np.testing.assert_almost_equal(tensor_tdephas, tdephas2)

        # Product of Depolarizing channels -> particular transformations of some states
        p = .5
        tdepol = self._depolarizing_channel(dim, p)
        tenspor_depol = tensor(tdepol, 2)

        self.assertEqual(tenspor_depol.shape, (dim**4, dim**4))
        self.assertEqual(tenspor_depol.dtype, tdepol.dtype)

        np.testing.assert_almost_equal(np.identity(dim**2).reshape(dim**4),
                                    tenspor_depol @ np.identity(dim**2).reshape(dim**4))

        # Longer products, check formating 
        tensor_3 = tensor([tdepol, tdephas, tu])

        self.assertEqual(tensor_3.shape, (dim**(2*3), dim**(2*3)))
        self.assertEqual(tensor_3.dtype, (tdepol[0,0]*tdephas[0,0]*tu[0,0]).dtype)