import unittest
import numpy as np

class TestStateTransformations(unittest.TestCase):

    def test_subsystem_reshape(self):
        from pyqch.state_transformations import subsystem_reshape
        # (dim**n, dim**n) to (dim,)*n
        dim = 2
        n = 3
        state = np.identity(dim**n) / dim**n
        rstate = subsystem_reshape(state, (dim,)*n)
        self.assertTupleEqual(rstate.shape, (dim,)*(2*n))

        # (dim1*dim2, dim1*dim2) to (dim1, dim2, dim1, dim2)
        dim1 = 2
        dim2 = 3
        state = np.identity(dim1*dim2) / dim1*dim2
        rstate = subsystem_reshape(state, (dim1, dim2))
        self.assertTupleEqual(rstate.shape, (dim1,dim2, dim1, dim2))

        # Incompatible dimensions raising an error
        dim = 2
        n = 3
        state = np.identity(dim**n) / dim**n
        with self.assertRaises(ValueError):
            rstate = subsystem_reshape(state, (dim,)*(n+1))

    def test_subsystem_permutation(self):
        from pyqch.state_transformations import subsystem_permutation
        
        # Identity
        rho1 = np.array([[1,0],[0,0]])
        rho2 = np.array([[0,0],[0,1]])
        rho = np.kron(rho1, rho2)
        rho_id = subsystem_permutation(rho, (2,2), (0,1))
        np.testing.assert_array_almost_equal(rho_id, rho)
        
        # Swap
        rho1 = np.array([[1,0],[0,0]])
        rho2 = np.array([[0,0],[0,1]])
        rho = np.kron(rho1, rho2)
        rho_swap = subsystem_permutation(rho, (2,2), (1,0))
        rho_swap_ref = np.kron(rho2, rho1)
        np.testing.assert_array_almost_equal(rho_swap, rho_swap_ref)
        
        # Reverse
        dim = 2
        n = 3
        vrho_arr = [np.random.random((dim, dim)) for _ in range(n)]
        rho_arr = [ v @ v.T / np.trace(v @ v.T) for v in vrho_arr]
        
        rho = 1
        for rhoi in rho_arr:
            rho = np.kron(rho, rhoi)
        
        rho_reverse_ref = 1
        for rhoi in rho_arr:
            rho_reverse_ref = np.kron(rhoi,rho_reverse_ref)
        
        rho_reverse = subsystem_permutation(rho, (dim,)*n, tuple(range(n-1, -1, -1)))
        np.testing.assert_array_almost_equal(rho_reverse, rho_reverse_ref)

        # Swapping only two of equal dimension
        rho1 = np.array([[1,0],[0,0]])
        rho2 = np.array([[0,0],[0,1]])
        rho3 = np.identity(3) / 3
        rho = np.kron(np.kron(rho1, rho2), rho3)
        rho_swap = subsystem_permutation(rho, (2,2,3), (1,0,2))
        rho_swap_ref = np.kron(np.kron(rho2, rho1), rho3)
        np.testing.assert_array_almost_equal(rho_swap, rho_swap_ref)

        # Incompatible sizes of system and permutation
        rho1 = np.array([[1,0],[0,0]])
        rho2 = np.array([[0,0],[0,1]])
        rho = np.kron(rho1, rho2)
        with self.assertRaises(ValueError):
            rho_swap = subsystem_permutation(rho, (2,2), (0,1,2))

        # Incompatible dimensions should work
        rho1 = np.array([[1,0],[0,0]])
        rho2 = np.array([[0,0],[0,1]])
        rho3 = np.identity(3) / 3
        rho = np.kron(np.kron(rho1, rho2), rho3)
        rho_swap = subsystem_permutation(rho, (2,2,3), (0,2,1))
        np.testing.assert_array_almost_equal(rho_swap, np.kron(np.kron(rho1, rho3), rho2))
    
    def test_subsystem_permutation_201(self):
        from pyqch.state_transformations import subsystem_permutation
        n = 3
        dim = 2
        perm = (2, 0, 1)
        
        rho_arr = [np.zeros((dim, dim)) for _ in range(n)]

        rho_in = 1
        for i_site, rho_site in enumerate(rho_arr):
            rho_site[0,0] = n - i_site
            rho_site[1,1] = i_site

            rho_in = np.kron(rho_in, rho_site)
        
        rho_perm = subsystem_permutation(rho_in, (dim,)*n, perm)

        rho_perm_arr_ref = [0] * n
        for i, i_p in enumerate(perm):
            rho_perm_arr_ref[i_p] = rho_arr[i]
        rho_perm_ref = 1
        for rho_site_perm in rho_perm_arr_ref:
            rho_perm_ref = np.kron(rho_perm_ref, rho_site_perm)

        np.testing.assert_array_almost_equal(rho_perm, rho_perm_ref)

    def test_partial_trace(self):
        from pyqch.state_transformations import partial_trace

        # Two sites
        rho1 = np.array([[1,0],[0,0]])
        rho2 = np.array([[0,0],[0,1]])
        rho = np.kron(rho1, rho2)
        rho_ptr = partial_trace(rho, (2,2), 1)
        np.testing.assert_array_almost_equal(rho_ptr, rho1)
        
        # 2 out of 3 in disjoint positions
        dim = 2
        n = 3
        rho_arr = []
        rho = 1
        for site in range(n):
            rho_site = np.zeros((dim, dim))
            rho_site[:2,:2] =  np.diag([np.cos(site / n * 2 * np.pi) ** 2,
                                          np.sin(site / n * 2 * np.pi) ** 2])
            rho_arr.append(rho_site)
            rho = np.kron(rho, rho_site)
        
        rho_ptr = partial_trace(rho, (dim,)*n, (0, 2))
        np.testing.assert_array_almost_equal(rho_ptr, rho_arr[1])

        # With keep_sites instead
        rho_ptr = partial_trace(rho, (dim,)*n, 1, keep_sites=True)
        np.testing.assert_array_almost_equal(rho_ptr, rho_arr[1])

        # 2 out of 3 in disjoint positions and unormalized
        dim = 2
        n = 3
        alpha = 3.14
        vrho_arr = [np.random.random((dim, dim)) for _ in range(n)]
        rho_arr = [ v @ v.T /np.trace(v @ v.T) for v in vrho_arr]
        rho_arr[0] = alpha * rho_arr[0]

        rho = 1
        for rhoi in rho_arr:
            rho = np.kron(rho, rhoi)
        
        rho_ptr = partial_trace(rho, (dim,)*n, (0, 1))
        np.testing.assert_array_almost_equal(rho_ptr, alpha * rho_arr[2])

        # 2 out of 3 in neighouring positions
        dim = 2
        n = 3
        vrho_arr = [np.random.random((dim, dim)) for _ in range(n)]
        rho_arr = [ v @ v.T /np.trace(v @ v.T) for v in vrho_arr]
        
        rho = 1
        for rhoi in rho_arr:
            rho = np.kron(rho, rhoi)
        
        rho_ptr = partial_trace(rho, (dim,)*n, (0, 1))
        np.testing.assert_array_almost_equal(rho_ptr, rho_arr[2])

        # With keep_sites instead
        rho_ptr = partial_trace(rho, (dim,)*n, 2, keep_sites=True)
        np.testing.assert_array_almost_equal(rho_ptr, rho_arr[2])

        # A maximally entangled state
        dim = 2
        vrho = np.identity(dim).reshape(dim**2)
        rho = np.outer(vrho, vrho) / dim
        rho_ptr = partial_trace(rho, (dim,dim), 1)
        np.testing.assert_array_almost_equal(rho_ptr, np.identity(dim)/dim)

        # With keep_sites instead
        rho_ptr = partial_trace(rho, (dim, dim), 1, keep_sites=True)
        np.testing.assert_array_almost_equal(rho_ptr, np.identity(dim)/dim)


from pyqch.state_transformations import local_channel
import pyqch.channel_families as cf
import pyqch.channel_operations as co

class TestStateTransformations_local_channel(unittest.TestCase):
       
    def test_local_channel_twosites(self):
        # Two sites
        t_depol = cf.depolarizing(2, 0.5)
        rho1 = np.array([[1,0],[0,0]])
        rho2 = np.array([[0,0],[0,1]])
        rho = np.kron(rho1, rho2)
        rho_ptr = local_channel(rho, (2,2), 1, cf.depolarizing(2, 0.5))
        rho_ref = np.kron(rho1, (t_depol @ rho2.reshape(2**2)).reshape((2,2)))
        np.testing.assert_array_almost_equal(rho_ptr, rho_ref)
    
    def test_local_channel_2to3neigh(self):
        # 2 out of 3 in neighbouring positions
        dim = 2
        t_dephas = cf.dephasing(dim**2, 0.5)
        n = 3
        rho_arr = []

        rho = 1
        for site in range(n):
            rho_site = np.zeros((dim, dim))
            rho_site[:2,:2] =  np.diag([np.cos(site / n * 2 * np.pi) ** 2,
                                          np.sin(site / n * 2 * np.pi) ** 2])
            rho_arr.append(rho_site)
            rho = np.kron(rho, rho_site)
        
        rho_ref = (t_dephas @ np.kron(rho_arr[0], rho_arr[1]).reshape(dim**4)).reshape((dim**2, dim**2))
        for rho_site in rho_arr[2:]:
            rho_ref = np.kron(rho_ref, rho_site)
        
        rho_ptr = local_channel(rho, (dim,)*n, (0, 1), t_dephas)
        np.testing.assert_array_almost_equal(rho_ptr, rho_ref)

    def test_local_channel_2to3disj(self):
        # 2 out of 3 in disjoint positions
        dim = 2
        n = 3
        t_replace = cf.depolarizing(dim**2, 1.)
        vrho_arr = [np.random.random((dim, dim)) for _ in range(n)]
        rho_arr = [ v @ v.T /np.trace(v @ v.T) for v in vrho_arr]
        
        rho = 1
        for rho_site in rho_arr:
            rho = np.kron(rho, rho_site)
        rho_ref = 1
        for site, rho_site in enumerate(rho_arr):
            if site in (0, 2):
                rho_site = np.identity(dim)/dim
            rho_ref = np.kron(rho_ref, rho_site)
        
        rho_rep = local_channel(rho, (dim,)*n, (0, 2), t_replace)
        np.testing.assert_array_almost_equal(rho_rep, rho_ref)

    def test_local_channel_2to3neighreorder(self):
        # 2 out of 3 in neighouring positions changeing order
        dim = 2
        n = 3
        tdepol0=cf.depolarizing(dim, .5)
        tdepol1 = cf.depolarizing(dim, .3)
        tdepol = co.tensor([tdepol1, tdepol0])
        vrho_arr = [np.random.random((dim, dim)) for _ in range(n)]
        rho_arr = [ v @ v.T /np.trace(v @ v.T) for v in vrho_arr]
        rho = 1
        for rhoi in rho_arr:
            rho = np.kron(rho, rhoi)
        
        rho_ref = np.kron((tdepol0 @ rho_arr[0].reshape(dim**2)).reshape((dim, dim)),
                          (tdepol1 @ rho_arr[1].reshape(dim**2)).reshape((dim, dim)))
        for rho_site in rho_arr[2:]:
            rho_ref = np.kron(rho_ref, rho_site)
        
        rho_ptr = local_channel(rho, (dim,)*n, (1, 0), tdepol)
        np.testing.assert_array_almost_equal(rho_ptr, rho_ref)

    def test_local_channel_maxentang(self):
        # A maximally entangled state
        dim = 2
        t_replace = cf.depolarizing(dim, 1.0)
        vrho = np.identity(dim).reshape(dim**2)
        rho = np.outer(vrho, vrho) / dim
        rho_ptr = local_channel(rho, (dim,dim), 1, t_replace)
        np.testing.assert_array_almost_equal(rho_ptr, np.identity(dim**2)/dim**2)

    def test_local_channel_dimchange(self):
        # A channel that changes dimension
        dim1 = 3
        dim2 = 4
        t_expand = np.zeros((dim2, dim2, dim1, dim1))
        for i in range(dim1):
            for j in range(dim1):
                t_expand[i,j,i,j] = 1
        t_expand = t_expand.reshape((dim2**2, dim1**2))
         
        rho_in = np.identity(dim1*dim1) / dim1**2

        rho_exp = np.zeros((dim2, dim2))
        rho_exp[:dim1, :dim1] = np.identity(dim1)/dim1
        rho_ref = np.kron(rho_exp, np.identity(dim1)/dim1)
        rho_ptr = local_channel(rho_in, (dim1,dim1), 0, t_expand)
        np.testing.assert_array_almost_equal(rho_ptr, rho_ref)

    def test_local_channel_dimchange_disjoint_preserveorder(self):
        # A channel that changes dimension in disjoint positions preserving original order
        n = 3
        active_sites = (0, 2)
        n_active = len(active_sites)
        dim1 = 2
        dim2 = dim1**n_active + 1  # Has to be larger than dim1**n_active
                    
        t_expand = np.zeros((dim2, dim2, dim1**n_active, dim1**n_active))
        for i in range(dim1**n_active):
            for j in range(dim1**n_active):
                t_expand[i,j,i,j] = 1
        t_expand = t_expand.reshape((dim2**2, dim1**(2*n_active)))

        vrho_arr = [np.random.random((dim1, dim1)) for _ in range(n)]
        rho_arr = [ v @ v.T /np.trace(v @ v.T) for v in vrho_arr]
        
        rho_in = 1
        for rho_site in rho_arr:
            rho_in = np.kron(rho_in, rho_site)

        rho_active = 1
        for site in active_sites:
            rho_active = np.kron(rho_active, rho_arr[site])
        rho_active_ref = (t_expand @ rho_active.reshape(dim1**(2*n_active))).reshape((dim2, dim2))
        
        rho_ref = 1
        for site in range(n):
            if site == active_sites[0]:
                rho_ref = np.kron(rho_ref, rho_active_ref)
            elif site not in active_sites:
                rho_ref = np.kron(rho_ref, rho_arr[site])
        
        rho_exp = local_channel(rho_in, (dim1,)*n, active_sites, t_expand)
        np.testing.assert_array_almost_equal(rho_exp, rho_ref)

    def test_local_channel_dimchange_disjoint_reverseorder(self):
        # A channel that changes dimension in disjoint positions preserving original order
        n = 3
        active_sites = (2, 0)
        n_active = len(active_sites)
        dim1 = 2
        dim2 = dim1**n_active + 1  # Has to be larger than dim1**n_active
                    
        t_expand = np.zeros((dim2, dim2, dim1**n_active, dim1**n_active))
        for i in range(dim1**n_active):
            for j in range(dim1**n_active):
                t_expand[i,j,i,j] = 1
        t_expand = t_expand.reshape((dim2**2, dim1**(2*n_active)))

        rho_arr = [np.zeros((dim1, dim1)) for _ in range(n)]

        rho_in = 1
        for i_site, rho_site in enumerate(rho_arr):
            rho_site[0,0] = n - i_site
            rho_site[1,1] = i_site

            rho_in = np.kron(rho_in, rho_site)

        rho_active = 1
        for site in active_sites:
            rho_active = np.kron(rho_active, rho_arr[site])
        rho_active_ref = (t_expand @ rho_active.reshape(dim1**(2*n_active))).reshape((dim2, dim2))
        
        rho_ref = 1
        for site in range(n):
            if site == active_sites[0]:
                rho_ref = np.kron(rho_ref, rho_active_ref)
            elif site not in active_sites:
                rho_ref = np.kron(rho_ref, rho_arr[site])
        
        rho_exp = local_channel(rho_in, (dim1,)*n, active_sites, t_expand)
        np.testing.assert_array_almost_equal(rho_exp, rho_ref)


    def test_local_channel_dimchange_middlepostition(self):
        # A channel that changes dimension in disjoint positions preserving original order
        n = 3
        active_site = 1
        dim1 = 2
        dim2 = dim1**2  # Has to be larger than dim1
                    
        t_expand = np.zeros((dim2, dim2, dim1, dim1))
        for i in range(dim1):
            for j in range(dim1):
                t_expand[i,j,i,j] = 1
        t_expand = t_expand.reshape((dim2**2, dim1**2))

        vrho_arr = [np.random.random((dim1, dim1)) for _ in range(n)]
        rho_arr = [ v @ v.T /np.trace(v @ v.T) for v in vrho_arr]

        rho_in = 1
        for rho_site in rho_arr:
            rho_in = np.kron(rho_in, rho_site)

        rho_active = rho_arr[active_site]
        rho_active_ref = (t_expand @ rho_active.reshape(dim1**2)).reshape((dim2, dim2))
        
        rho_ref = 1
        for site in range(n):
            if site == active_site:
                rho_ref = np.kron(rho_ref, rho_active_ref)
            else:
                rho_ref = np.kron(rho_ref, rho_arr[site])
        
        rho_exp = local_channel(rho_in, (dim1,)*n, active_site, t_expand)
        np.testing.assert_array_almost_equal(rho_exp, rho_ref)

    def test_local_channel_tensor_replace_should_replace(self):
        dim = 2
        n = 4
        replacer = cf.depolarizing(dim, 1.0)

        rho = np.zeros((dim**n, dim**n))
        rho[1,1] = 1
        for i in range(n):
            rho = local_channel(rho, (dim,)*n, i, replacer)
        np.testing.assert_array_almost_equal(rho, np.identity(dim**n) / (dim**n))


class TestStateTransformations_twirling(unittest.TestCase):


    def setUp(self):
        from pyqch import random_generators as rg

        def gen_states(dim, number):
            for _ in range(number):
                yield rg.state(dim)

        self.gen_states = gen_states

    def test_twirling_with_identity(self):
        from pyqch import state_transformations as st

        dim = 3
        number = 5
        r = [np.identity(dim)]

        for arbpsi in self.gen_states(dim, number):
            twirl_arbpsi = st.twirling(arbpsi, r)
            np.testing.assert_array_almost_equal(twirl_arbpsi, arbpsi)
    
    def test_twirling_with_Paulis(self):
        from pyqch import state_transformations as st

        dim = 2
        number = 5
        r = [np.identity(dim),
             np.array([[0,1],[1,0]]),
             np.array([[0,-1j],[1j,0]]),
             np.array([[1,0],[0,-1]])]

        for arbpsi in self.gen_states(dim, number):
            twirl_arbpsi = st.twirling(arbpsi, r)

            np.testing.assert_array_almost_equal(twirl_arbpsi, np.identity(dim)/dim * np.trace(arbpsi))

    def test_twirling_with_Haar(self):
        from pyqch import state_transformations as st

        dim = 5
        number = 5
        decomposition = [1, np.array([dim]), np.array([1])]

        for arbpsi in self.gen_states(dim, number):
            twirl_arbpsi = st.twirling(arbpsi, decomposition=decomposition)

            np.testing.assert_array_almost_equal(twirl_arbpsi, np.identity(dim)/dim * np.trace(arbpsi))

    def test_twirling_swap(self):
        from pyqch import state_transformations as st

        # Compare the decomposition picture with the explicit
        dim = 3
        number = 5

        swap = np.identity(dim**2).reshape((dim, dim, dim, dim))
        swap = np.einsum("ijkl->jikl", swap).reshape((dim**2, dim**2))

        r = [np.identity(dim**2), swap]
        vals, W = np.linalg.eigh(swap)
        dim_plus = int(np.sum(vals > 0))
        dim_minus = int(np.sum(vals < 0))
        decomposition = [W, np.array([1, 1]), np.array([dim_minus, dim_plus])]

        for arbpsi in self.gen_states(dim**2, number):
            twirl1_arbpsi = st.twirling(arbpsi, r=r)

            twirl2_arbpsi = st.twirling(arbpsi, decomposition=decomposition)

            np.testing.assert_array_almost_equal(twirl1_arbpsi, twirl2_arbpsi)


class TestStateTransformations_lowRankRepresentation(unittest.TestCase):


    def test_shape(self):
        import pyqch.random_generators as rg
        from pyqch.state_transformations import low_rank_representation

        params = [(5, 3, 3),
                  (5, 4, 3),
                  (5, 2, 3)]

        for dim, in_rank, ou_rank in params:
            rho = rg.state(dim, in_rank, random_state=13082024)
            lr_rho = low_rank_representation(rho, output_rank=ou_rank)

            self.assertEqual(lr_rho.shape, (ou_rank, dim))

    def test_expected_shape(self):
        import pyqch.random_generators as rg
        from pyqch.state_transformations import low_rank_representation

        # The expected rank is the rank used in the generator
        params = [(5, 3),
                  (5, 4),
                  (5, 2)]

        for dim, in_rank in params:
            rho = rg.state(dim, in_rank, random_state=13082024)
            lr_rho = low_rank_representation(rho)

            self.assertEqual(lr_rho.shape, (in_rank, dim))

    def test_normalization(self):
        import pyqch.random_generators as rg
        from pyqch.state_transformations import low_rank_representation

        # The expected rank is the rank used in the generator
        params = [(5, 3),
                  (5, 4),
                  (5, 2)]

        for dim, in_rank in params:
            rho = rg.state(dim, in_rank, random_state=13082024)
            lr_rho = low_rank_representation(rho)

            self.assertAlmostEqual(np.trace(lr_rho.T @ lr_rho.conj()), 1.0)

    def test_reverse_is_state(self):
        import pyqch.random_generators as rg
        from pyqch.predicates import is_density_matrix
        from pyqch.state_transformations import low_rank_representation

        params = [(5, 3),
                  (5, 4),
                  (5, 2)]

        for dim, in_rank in params:
            rho = rg.state(dim, in_rank, random_state=13082024)
            lr_rho = low_rank_representation(rho)

            lr_rho_rev = low_rank_representation(lr_rho, reverse=True)

            self.assertTrue(is_density_matrix(lr_rho_rev))
            

    def test_do_undo(self):
        import pyqch.random_generators as rg
        from pyqch.state_transformations import low_rank_representation

        params = [(5, 3),
                  (5, 4),
                  (5, 2)]

        for dim, in_rank in params:
            rho = rg.state(dim, in_rank, random_state=13082024)
            lr_rho = low_rank_representation(rho)

            lr_rho_rev = low_rank_representation(lr_rho, reverse=True)

            np.testing.assert_array_almost_equal(lr_rho_rev, rho)
