import unittest
import numpy as np
import  src.pyqch.random_generators as rg

class TestRandomGenerators_state(unittest.TestCase):

    def test_rg_state_basic_struct(self):
        from  src.pyqch.predicates import is_density_matrix
        
        dim = 3
        rank = dim

        # Assert basic structure
        num_instances = 5
        for instance in range(num_instances):
            rho = rg.state(dim, rank)

            self.assertEqual(rho.shape, (dim, dim))
            self.assertEqual(rho.dtype, complex)

            self.assertTrue(is_density_matrix(rho))
    
    def test_state_purestates(self):
        dim = 3
        # Assert that rank 1 is a pure state
        rho = rg.state(dim, 1)
        np.testing.assert_almost_equal(rho, rho @ rho)

    def test_state_rankisok(self):
        dim = 3
        rank = dim
        # Assert the rank of the state is ok
        num_instances = 5
        step = max([dim // num_instances, 1])  # upper bounds the number of rank instances to consider
        for rank in range(1, dim+1, step):
            rho = rg.state(dim, rank)
            self.assertEqual(np.linalg.matrix_rank(rho, tol=1e-6, hermitian=True), rank)

    def test_state_fixedseed_rng(self):
        seed = 24052024
        rng = np.random.default_rng(seed)
        dim = 3
        rank = 4

        state_ref = np.array([[ 0.45120698-3.63115313e-18j, -0.29729348+6.77260026e-02j, 0.12297973+2.73712147e-02j],
                              [-0.29729348-6.77260026e-02j,  0.32106235-1.02566936e-18j, -0.11659089-6.27375225e-02j],
                              [ 0.12297973-2.73712147e-02j, -0.11659089+6.27375225e-02j, 0.22773067-6.28566493e-18j]])
        state_gen = rg.state(dim, rank, rng)
        np.testing.assert_array_almost_equal(state_gen, state_ref)
    
    def test_state_fixedseed_int(self):
        seed = 24052024
        dim = 3
        rank = 4

        state_ref = np.array([[ 0.45120698-3.63115313e-18j, -0.29729348+6.77260026e-02j, 0.12297973+2.73712147e-02j],
                              [-0.29729348-6.77260026e-02j,  0.32106235-1.02566936e-18j, -0.11659089-6.27375225e-02j],
                              [ 0.12297973-2.73712147e-02j, -0.11659089+6.27375225e-02j, 0.22773067-6.28566493e-18j]])
        state_gen = rg.state(dim, rank, seed)
        np.testing.assert_array_almost_equal(state_gen, state_ref)
    
    def test_state_fixedglobalseed(self):
        seed = 24052024
        dim = 3
        rank = 4

        np.random.seed(seed)
        state_gen_1 = rg.state(dim, rank, None)
        
        np.random.seed(seed)
        state_gen_2 = rg.state(dim, rank, None)
        np.testing.assert_array_almost_equal(state_gen_1, state_gen_2)


class TestRandomGenerators_state_dirichlet(unittest.TestCase):
    
    def test_rg_state_dirichlet_basic_struct(self):
        from  src.pyqch.predicates import is_density_matrix
        
        dim = 3
        alpha = .5

        # Assert basic structure
        num_instances = 5
        for instance in range(num_instances):
            rho = rg.state_dirichlet(dim, alpha)

            self.assertEqual(rho.shape, (dim, dim))
            self.assertEqual(rho.dtype, complex)

            self.assertTrue(is_density_matrix(rho))
    
    def test_state_dirichlet_almostpurestates(self):
        dim = 3
        alpha = 0.000000001
        tol_decimal = 6
        # Assert that alpha = 0 is a pure state
        rho = rg.state_dirichlet(dim, alpha)
        np.testing.assert_almost_equal(rho, rho @ rho, decimal=tol_decimal)

    def test_state_dirichlet_fixedseed_rng(self):
        seed = 24052024
        rng = np.random.default_rng(seed)
        dim = 3
        alpha = .5

        state_ref = np.array([[ 0.43576134+6.93889390e-18j, -0.02721049-8.85036648e-03j, 0.01860752+7.05206604e-02j],
                              [-0.02721049+8.85036648e-03j,  0.20508533+0.00000000e+00j, 0.19818465+9.23607230e-02j],
                              [ 0.01860752-7.05206604e-02j,  0.19818465-9.23607230e-02j, 0.35915333+1.38777878e-17j]])
        state_gen = rg.state_dirichlet(dim, alpha, rng)
        np.testing.assert_array_almost_equal(state_gen, state_ref)
    
    def test_state_dirichlet_fixedseed_int(self):
        seed = 24052024
        dim = 3
        alpha = .5

        state_ref = np.array([[ 0.43576134+6.93889390e-18j, -0.02721049-8.85036648e-03j, 0.01860752+7.05206604e-02j],
                              [-0.02721049+8.85036648e-03j,  0.20508533+0.00000000e+00j, 0.19818465+9.23607230e-02j],
                              [ 0.01860752-7.05206604e-02j,  0.19818465-9.23607230e-02j, 0.35915333+1.38777878e-17j]])
        state_gen = rg.state_dirichlet(dim, alpha, seed)
        np.testing.assert_array_almost_equal(state_gen, state_ref)
    
    def test_state_dirichlet_fixedglobalseed(self):
        seed = 24052024
        dim = 3
        alpha = 3

        np.random.seed(seed)
        state_gen_1 = rg.state_dirichlet(dim, alpha, None)
        
        np.random.seed(seed)
        state_gen_2 = rg.state_dirichlet(dim, alpha, None)
        np.testing.assert_array_almost_equal(state_gen_1, state_gen_2)

    def test_state_dirichlet_alphabig(self):
        # For big alpha the state should concentrate near maximally mixed
        seed = 24052024
        dim = 3
        alpha = 100000
        tol_decimal = 2

        state_gen = rg.state_dirichlet(dim, alpha, seed)
        state_ref = np.identity(dim) / dim
        np.testing.assert_array_almost_equal(state_gen, state_ref, decimal=tol_decimal)


class TestRandomGenerators_channel(unittest.TestCase):
    def test_channel_format(self):
        from src.pyqch.predicates import is_channel
        
        np.random.seed(1)
        dim = 3
        dim2 = dim + 1
        rank = dim

        # Asserting output format, type and CPTP
        num_instances = 5
        for instance in range(num_instances):
            t = rg.channel(dim, dim2, rank)

            self.assertEqual(t.shape, (dim2**2, dim**2))
            self.assertEqual(t.dtype, complex)
            self.assertTrue(is_channel(t))

    def test_channel_errorifbadrank(self):
        dim = 3
        dim2 = dim + 1
        # Assert value error when rank * dim_out < dim_in
        with self.assertRaises(ValueError):
            rg.channel(dim+1, dim, 1)
        with self.assertRaises(ValueError):
            rg.channel(2*dim, dim, 1)

    def test_channel_unitary(self):
        dim = 3
        # Asserting rank = 1 produces a unitary evolution
        tu = rg.channel(dim, dim, 1)
        np.testing.assert_almost_equal(tu @ tu.conj().T, np.identity(dim**2))

    def test_channel_krausrank(self):
        from src.pyqch.channel_operations import choi_state
        dim = 3
        dim2 = dim + 1
        # Assert the output channel has the desired Kraus rank
        num_instances = 5
        step = max([dim*dim2 // num_instances, 1])  # upper bounds the number of rank instances to consider
        for rank in range(1, dim*dim2+1, step):
            trank = rg.channel(dim, dim2, rank)

            tau = choi_state(trank)
            self.assertEqual(np.linalg.matrix_rank(tau, tol=1e-6, hermitian=True), rank)

    def test_channel_fixedseed(self):
        seed = 24052024
        dim_in, dim_out, kraus_rank = 2, 3, 4

        channel_ref = np.array([[ 0.18494598+0.j        , -0.10095978-0.06593522j, -0.10095978+0.06593522j,  0.28326041+0.j        ],
                                [-0.13991956+0.19688071j,  0.03644306-0.03253516j, 0.08843988-0.10019202j,  0.04661948+0.2000453j ],
                                [ 0.14104275+0.0658656j , -0.00162202+0.07251312j, 0.06064763+0.04085394j,  0.0632398 -0.10138501j],
                                [-0.13991956-0.19688071j,  0.08843988+0.10019202j, 0.03644306+0.03253516j,  0.04661948-0.2000453j ],
                                [ 0.40436078+0.j        ,  0.0831271 +0.07714417j, 0.0831271 -0.07714417j,  0.36327774+0.j        ],
                                [-0.01326471-0.22780949j,  0.06046184-0.05677724j, 0.0292701 -0.0287341j , -0.11199429+0.06324956j],
                                [ 0.14104275-0.0658656j ,  0.06064763-0.04085394j, -0.00162202-0.07251312j,  0.0632398 +0.10138501j],
                                [-0.01326471+0.22780949j,  0.0292701 +0.0287341j , 0.06046184+0.05677724j, -0.11199429-0.06324956j],
                                [ 0.41069324+0.j        ,  0.01783268-0.01120896j, 0.01783268+0.01120896j,  0.35346184+0.j        ]])
        channel_gen = rg.channel(dim_in, dim_out, kraus_rank, random_state=seed)
        np.testing.assert_array_almost_equal(channel_gen, channel_ref)

class TestRandomGenerators_unitary_channel(unittest.TestCase):
    def test_unitary_channel_structure(self):
        from  src.pyqch.predicates import is_channel
        dim = 3

        t = rg.unitary_channel(dim)

        self.assertEqual(t.shape, (dim**2, dim**2))
        self.assertEqual(t.dtype, complex)

        self.assertTrue(is_channel(t))

    def test_unitary_channel_unitarity(self):
        dim = 3
        t = rg.unitary_channel(dim)

        np.testing.assert_almost_equal(np.identity(dim), (t @ np.identity(dim).reshape(dim**2)).reshape((dim, dim)))
        
        np.testing.assert_almost_equal(np.identity(dim**2), (t @ t.T.conj()))
    
    def test_unitary_channel_fixedseed(self):
        dim = 2
        seed = 24052024

        u_ref = np.array([[ 0.12559325+2.38989619e-18j, -0.32163866+7.98007155e-02j, -0.32163866-7.98007155e-02j,  0.87440675-2.55239885e-17j],
                          [ 0.32871257-4.20432152e-02j,  0.11707548-4.54642210e-02j, -0.86853196-1.01189950e-01j, -0.32871257+4.20432152e-02j],
                          [ 0.32871257+4.20432152e-02j, -0.86853196+1.01189950e-01j, 0.11707548+4.54642210e-02j, -0.32871257-4.20432152e-02j],
                          [ 0.87440675-2.63303972e-17j,  0.32163866-7.98007155e-02j, 0.32163866+7.98007155e-02j,  0.12559325+2.30853254e-18j]])
        u_gen = rg.unitary_channel(dim, random_state=seed)
        np.testing.assert_array_almost_equal(u_gen, u_ref)
