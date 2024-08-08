import unittest
import numpy as np

class TestChannelFamily(unittest.TestCase):

    def _channel_function(self):
        raise NotImplementedError('Implement _channel_function for general channel integrity verification.')
    
    def _dimin_dimout(self, *args):
        raise NotImplementedError('Implement _dimin_dimout for general channel integrity verification.')

    def _arbitrary_argument_generator(self):
        raise NotImplementedError('Implement _arbitrary_argument_generator for general channel integrity verification.')
    
    def _transition_matrix_from_kraus_representation(self, kraus_operators):
        krank, dim2, dim1 = kraus_operators.shape
        return np.einsum('sji,spq->jpiq', kraus_operators, kraus_operators.conj()).reshape((dim2**2, dim1**2))

    def _skip_if_abstract_class(self):
        if type(self) is TestChannelFamily:
            self.skipTest("TestChannelFamily is used as a basis for other channel family tests.")

    def test_shape(self):
        self._skip_if_abstract_class()

        for params in self._arbitrary_argument_generator():
            tm = self._channel_function()(*params)
            dimin, dimout = self._dimin_dimout(*params)

            self.assertEqual(tm.shape, (dimout**2, dimin**2))

    def test_is_channel(self):
        self._skip_if_abstract_class()

        from pyqch.predicates import is_channel

        for params in self._arbitrary_argument_generator():
            tm = self._channel_function()(*params)
            self.assertTrue(is_channel(tm))

    def test_kraus_representation(self):
        self._skip_if_abstract_class()

        for params in self._arbitrary_argument_generator():
            tm = self._channel_function()(*params)

            try:
                kraus_ops = self._channel_function()(*params, kraus_representation=True)
                
                np.testing.assert_array_almost_equal(tm,
                                                    self._transition_matrix_from_kraus_representation(kraus_operators=kraus_ops))
            except:
                with self.assertRaisesRegex(ValueError, 'This map does not admit the usual Kraus representation.'):
                    kraus_ops = self._channel_function()(*params, kraus_representation=True)


class TestChannelFamilies_depolarizing(TestChannelFamily):

    def _channel_function(self):
        from  pyqch.channel_families import depolarizing
        return depolarizing
    
    def _dimin_dimout(self, *args):
        return (args[0], args[0])

    def _arbitrary_argument_generator(self):
        
        dim = 5
        p_arr = np.linspace(0, 1, 10, endpoint=True)

        r1 = np.identity(dim) / dim
        
        r2 = np.zeros((dim, dim))
        r2[0,0] = 1

        ref_states = [r1, r2]

        for p in p_arr:
            for r in ref_states:
                yield dim, p, r

    
    def test_depolarizing(self):
        from  src.pyqch.channel_families import depolarizing

        dim = 3
        p = .67
        r = np.identity(dim) / dim

        rho_in = np.diag(np.arange(1, 1+dim, dtype=float))
        rho_in /= np.trace(rho_in)

        dmat = depolarizing(dim, p, r)

        rho_out = (dmat @ rho_in.reshape(dim**2)).reshape((dim, dim))

        np.testing.assert_almost_equal(rho_out, (1-p) * rho_in + p * r)
    
class TestChannelFamilies_probabilistic_damping(TestChannelFamily):

    def _channel_function(self):
        from  pyqch.channel_families import probabilistic_damping
        return probabilistic_damping
    
    def _dimin_dimout(self, *args):
        return (args[0], args[0])

    def _arbitrary_argument_generator(self):
        dim = 5
        g_num = 5

        for g in np.linspace(0, 1, g_num, endpoint=True):
            yield dim, g
    
    def test_probabilistic_damping(self):
        from  src.pyqch.channel_families import probabilistic_damping

        dim = 3
        g = .33

        rho_in = np.diag(np.arange(1, 1+dim, dtype=float))
        rho_in /= np.trace(rho_in)

        rho_out_ref = np.diag(np.arange(2, 2+dim, dtype=float))
        rho_out_ref[dim-1, dim-1] = 0
        rho_out_ref[0,0] += 1
        rho_out_ref /= np.trace(rho_out_ref)

        rho_out_ref = (1-g) * rho_in + g * rho_out_ref

        dmat = probabilistic_damping(dim, g)
        rho_out = (dmat @ rho_in.reshape(dim**2)).reshape((dim, dim))

        np.testing.assert_almost_equal(rho_out, rho_out_ref)

class TestChannelFamilies_povm(TestChannelFamily):
    def test_shape(self):
        raise NotImplementedError('This test requires specific treatment in some cases.')

    def test_is_channel(self):
        raise NotImplementedError('This test requires specific treatment in some cases.')
    
    def test_kraus_representation(self):
        raise NotImplementedError('This test requires specific treatment in some cases.')

    def _channel_function(self):
        from  pyqch.channel_families import povm
        return povm
    
    def _dimin_dimout(self, *args):
        dim = args[0]
        pos = args[1]
        mode = args[2]

        m = pos.shape[0]

        if mode == 'q-q':
            return (dim**2, dim**2)
        elif mode == 'q-c':
            return (m**2, dim**2)
        elif mode == 'q-qc':
            return ((m*dim)**2, dim**2)
        else:
            raise ValueError(f'Not allowed mode {mode}.')

    def _arbitrary_argument_generator(self):
        dim = 8
        pos = np.zeros((dim, dim, dim))
        for x in range(dim):
            pos[x,x,x] = 1

        modes = ['q-q', 'q-c', 'q-qc']

        for mode in modes:
            yield dim, pos, mode
    
    def test_povm_computational_basis(self):
        # POs for the computational basis
        dim = 8
        pos = np.zeros((dim, dim, dim))
        for x in range(dim):
            pos[x,x,x] = 1

        # POVM mode consistency
        self.__povm_mode_consistency(dim, pos)

    def __povm_mode_consistency(self, dim, pos):
        from  src.pyqch.channel_families import povm
        # Checks whether different modes of POVM are consistent
        m = pos.shape[0]

        # Create input state
        rho_in = np.diag(np.arange(1, 1+dim, dtype=float))
        rho_in /= np.trace(rho_in)

        # Create POVM matrices
        qqc_mat = povm(dim, pos, 'q-qc')
        qc_mat = povm(dim, pos, 'q-c')
        qq_mat = povm(dim, pos, 'q-q')

        r_qc = (qqc_mat @ rho_in.reshape((dim**2,))).reshape((m, dim, m ,dim))
        r_q = (qq_mat @ rho_in.reshape((dim**2,))).reshape((dim, dim))
        r_c = (qc_mat @ rho_in.reshape((dim**2,))).reshape((m, m))

        r_qc_q = np.einsum("ijil->jl", r_qc)
        r_qc_c = np.einsum("ijlj->il", r_qc)

        np.testing.assert_almost_equal(r_q,r_qc_q)
        np.testing.assert_almost_equal(r_c,r_qc_c)


class TestChannelFamilies_dephasing(TestChannelFamily):

    def _channel_function(self):
        from  pyqch.channel_families import dephasing
        return dephasing
    
    def _dimin_dimout(self, *args):
        return (args[0], args[0])

    def _arbitrary_argument_generator(self):
        from scipy.stats import unitary_group
        dim = 3
        g = .33
        u = unitary_group(dim, seed=123).rvs()

        yield dim, g, None
        yield dim, g, u

        # Generate a random positive definite matrix
        g = np.random.random((dim, dim))
        g = g @ g.T

        # Normalize so that setting the diagonal elements to one is
        # equivalent to adding a positive diagonal matrix
        g /= np.max(np.diag(g))
        for i in range(dim):
            g[i, i] = 1

        yield dim, g, None
        yield dim, g, u
            
    def test_dephasing_simple_computational(self):
        from  src.pyqch.channel_families import dephasing
        dim = 3
        g = .33

        # Create input state
        rho_in = np.full((dim, dim), 1/dim)

        # Create reference state
        rho_out_ref = (1-g) * rho_in + g * np.diag(np.diag(rho_in))

        # Test the action of the channel
        dephas_mat = dephasing(dim, g)
        rho_out = (dephas_mat @ rho_in.reshape(dim**2)).reshape((dim, dim))

        np.testing.assert_almost_equal(rho_out, rho_out_ref)
    
    def test_dephasing_simple_arbbasis(self):
        from  src.pyqch.channel_families import dephasing
        from scipy.stats import unitary_group

        dim = 3
        g = .33
        u = unitary_group(dim, seed=123).rvs()

        # Create input state
        rho_in = np.full((dim, dim), 1/dim)

        # Create reference state
        rho_out_ref = (1-g) * rho_in + g * (u.conj().T @ np.diag(np.diag( u @ rho_in @ u.T.conj())) @ u)

        # Test the action of the channel
        dephas_mat = dephasing(dim, g, u=u)
        rho_out = (dephas_mat @ rho_in.reshape(dim**2)).reshape((dim, dim))

        np.testing.assert_almost_equal(rho_out, rho_out_ref)

    def test_dephasing_general_computational(self):
        from  src.pyqch.channel_families import dephasing
        dim = 3

        # Generate a random positive definite matrix
        g = np.random.random((dim, dim))
        g = g @ g.T

        # Normalize so that setting the diagonal elements to one is
        # equivalent to adding a positive diagonal matrix
        g /= np.max(np.diag(g))
        for i in range(dim):
            g[i, i] = 1

        # Create input state
        rho_in = np.full((dim, dim), 1/dim)

        # Create reference state
        rho_out_ref = g * rho_in

        # Test the action of the channel
        dephas_mat = dephasing(dim, g)
        rho_out = (dephas_mat @ rho_in.reshape(dim**2)).reshape((dim, dim))

        np.testing.assert_almost_equal(rho_out, rho_out_ref)
    
    def test_dephasing_general_arbbasis(self):
        from  src.pyqch.channel_families import dephasing
        from scipy.stats import unitary_group
        dim = 3
        u = unitary_group(dim, seed=123).rvs()
    
        # Generate a random positive definite matrix
        g = np.random.random((dim, dim))
        g = g @ g.T

        # Normalize so that setting the diagonal elements to one is
        # equivalent to adding a positive diagonal matrix
        g /= np.max(np.diag(g))
        for i in range(dim):
            g[i, i] = 1

        # Create input state
        rho_in = np.full((dim, dim), 1/dim)

        # Create reference state
        rho_out_ref = u.T.conj() @ (g * (u @ rho_in @ u.conj().T)) @ u

        # Test the action of the channel
        dephas_mat = dephasing(dim, g, u=u)
        rho_out = (dephas_mat @ rho_in.reshape(dim**2)).reshape((dim, dim))

        np.testing.assert_almost_equal(rho_out, rho_out_ref)


class TestChannelFamilies_initializer(TestChannelFamily):

    def test_shape(self):
        raise NotImplementedError('This test requires specific treatment in some cases.')

    def test_is_channel(self):
        raise NotImplementedError('This test requires specific treatment in some cases.')
    
    def test_kraus_representation(self):
        raise NotImplementedError('This test requires specific treatment in some cases.')

    def _channel_function(self):
        from  pyqch.channel_families import initializer
        return initializer
    
    def _dimin_dimout(self, *args):
        dim = args[0]
        states = args[1]
        mode = args[2]

        m = states.shape[0]

        if mode == 'q-qc':
            return ((m * dim) ** 2, m)
        elif mode == 'q-q':
            return (dim ** 2, m ** 2)
        elif mode == 'c-qc':
            return ((m * dim) ** 2, m)
        elif mode == 'c-q':
            return (dim ** 2, m)
        else:
            raise ValueError(f'Not allowed mode {mode}.')

    def _arbitrary_argument_generator(self):

        dim = 3

        # Mixed form
        state_list_1 = []
        state_list_1.append(np.full((dim, dim), 1/dim))
        state_list_1.append(np.diag(2/dim/(dim+1)*np.arange(1, 1+dim, dtype=float)))
        states_1 = np.array(state_list_1)

        # Pure form
        state_list_2 = []
        state_list_2.append(np.full(dim, 1/np.sqrt(dim)))
        k2 = np.zeros(dim)
        k2[0] = 1 / np.sqrt(dim)
        k2[1] = 1 / np.sqrt(dim)
        state_list_2.append(k2)
        states_2 = np.array(state_list_2)

        modes = ['q-qc', 'q-q', 'c-qc', 'c-q']

        for states in [states_1, states_2]:
            for mode in modes:
                yield dim, states, mode
    
    def test_initializer(self):
        from  src.pyqch.channel_families import initializer

        dim = 3
        state_list = []
        state_list.append(np.full((dim, dim), 1/dim))
        state_list.append(np.diag(2/dim/(dim+1)*np.arange(1, 1+dim, dtype=float)))
        states = np.array(state_list)

        # Define an input probability distribution
        p_in = np.arange(1, 1+states.shape[0]) * 2 /states.shape[0]/(states.shape[0]+1)

        init_mat = initializer(dim, states, mode='c-q')

        rho_out = (init_mat @ p_in).reshape((dim, dim))

        rho_out_ref = np.sum([p_in[k] * state_list[k] for k in range(len(state_list))], axis=0)
        np.testing.assert_almost_equal(rho_out, rho_out_ref)

    def test_initializer_pure_states(self):
        from  src.pyqch.channel_families import initializer

        dim = 3
        state_list = []
        state_list.append(np.full(dim, 1/np.sqrt(dim)))
        k2 = np.zeros(dim)
        k2[0] = 1 / np.sqrt(dim)
        k2[1] = 1 / np.sqrt(dim)
        state_list.append(k2)
        states = np.array(state_list)

        # Define an input probability distribution
        p_in = np.arange(1, 1+states.shape[0]) * 2 /states.shape[0]/(states.shape[0]+1)

        init_mat = initializer(dim, states, mode='c-q')

        rho_out = (init_mat @ p_in).reshape((dim, dim))

        rho_out_ref = np.sum([p_in[k] * np.outer(state_list[k], state_list[k].conj())
                              for k in range(len(state_list))], axis=0)
        np.testing.assert_almost_equal(rho_out, rho_out_ref)
    

class TestChannelFamilies_probabilistic_unitaries(TestChannelFamily):

    def _channel_function(self):
        from  pyqch.channel_families import probabilistic_unitaries
        return probabilistic_unitaries
    
    def _dimin_dimout(self, *args):
        return (args[0], args[0])

    def _arbitrary_argument_generator(self):
        from scipy.stats import unitary_group
        seed = 20240808
        rng = np.random.default_rng(seed)
        dim = 3

        num_cases = 5
        unitary_num = 4

        for _ in range(num_cases):
            p_arr = np.random.random(unitary_num)
            p_arr /= np.sum(p_arr)

            u_arr = np.array([unitary_group.rvs(dim, random_state=rng) for _ in range(unitary_num)])
            yield dim, p_arr, u_arr    
    
    def test_probabilistic_unitaries(self):
        from  src.pyqch.channel_families import probabilistic_unitaries

        dim = 3

        u_list = []

        u_list.append(np.identity(dim))
        u_list.append(np.diag(np.exp(-1j * np.linspace(0, np.pi, dim))))
        u_list.append(np.diag(np.exp(-1j * np.linspace(0, .3*np.pi, dim))))

        us = np.array(u_list)
        probs = np.arange(1, 1+us.shape[0]) * 2/(1+us.shape[0])/us.shape[0]

        rho_in = np.full((dim, dim), 1/dim)

        pu_mat = probabilistic_unitaries(dim, probs, us)

        rho_out = (pu_mat @ rho_in.reshape(dim**2)).reshape((dim, dim))

        rho_out_ref = np.sum([probs[k] * us[k] @ rho_in @ us[k].T.conj() for k in range(len(u_list))], axis=0)
        np.testing.assert_almost_equal(rho_out, rho_out_ref)


class TestChannelFamilies_embed_classical(TestChannelFamily):

    def _channel_function(self):
        from  pyqch.channel_families import embed_classical
        return embed_classical
    
    def _dimin_dimout(self, *args):
        return args[1].shape

    def _arbitrary_argument_generator(self):
        dim = 3
        stoch = np.outer(np.arange(1, dim+1, dtype=float), np.arange(1, dim+1, dtype=float))
        stoch /= np.sum(stoch, axis=1)

        yield dim, stoch
    
    def test_embed_classical(self):
        from  src.pyqch.channel_families import embed_classical

        dim = 3
        stoch = np.outer(np.arange(1, dim+1, dtype=float), np.arange(1, dim+1, dtype=float))
        stoch /= np.sum(stoch, axis=1)

        p_in = np.full(dim, 1/dim)
        rho_in = np.full((dim, dim), 1/dim)

        rho_p_out = np.diag(stoch @ p_in)
        rho_out = (embed_classical(dim, stoch) @ rho_in.reshape(dim**2)).reshape((dim, dim))

        np.testing.assert_almost_equal(rho_p_out, rho_out)


class TestChannelFamilies_classical_permutation(TestChannelFamily):

    def test_shape(self):
        raise NotImplementedError('This test requires specific treatment in some cases.')

    def test_is_channel(self):
        raise NotImplementedError('This test requires specific treatment in some cases.')
    
    def test_kraus_representation(self):
        raise NotImplementedError('This test requires specific treatment in some cases.')

    def _channel_function(self):
        from  pyqch.channel_families import classical_permutation
        return classical_permutation
    
    def _dimin_dimout(self, *args):
        return (args[0], args[0])

    def _arbitrary_argument_generator(self):
        dim = 3
        shift = 1
        perm = np.roll(np.arange(dim), -shift)
        yield dim, perm
            
    def test_classical_permutation(self):
        from  src.pyqch.channel_families import classical_permutation

        dim = 3
        shift = 1
        perm = np.roll(np.arange(dim), -shift)

        p_in = np.arange(dim)

        p_out_ref = np.roll(p_in, shift)

        p_out = classical_permutation(dim, perm) @ p_in

        np.testing.assert_almost_equal(p_out, p_out_ref)
    

class TestChannelFamilies_transposition(TestChannelFamily):

    def test_is_channel(self):
        self._skip_if_abstract_class()

        from pyqch.predicates import is_channel

        for params in self._arbitrary_argument_generator():
            tm = self._channel_function()(*params)
            self.assertFalse(is_channel(tm))

    def _channel_function(self):
        from  pyqch.channel_families import transposition
        return transposition
    
    def _dimin_dimout(self, *args):
        return (args[0], args[0])

    def _arbitrary_argument_generator(self):
        from scipy.stats import unitary_group
        seed = 20240808
        dim = 3

        u_arr = [None, unitary_group.rvs(dim, random_state=seed)]

        for u in u_arr:
            yield dim, u
            
    def test_transposition(self):
        from pyqch.channel_families import transposition

        dim = 3
        rho_in = np.random.default_rng(123).random((dim, dim))

        trans = transposition(dim)
        rho_out = (trans @ rho_in.reshape((dim**2))).reshape((dim, dim))
        np.testing.assert_array_almost_equal(rho_out, rho_in.T)


class TestChannelFamilies_amplitude_damping(TestChannelFamily):

    def _channel_function(self):
        from  pyqch.channel_families import amplitude_damping
        return amplitude_damping
    
    def _dimin_dimout(self, *args):
        return (args[0], args[0])

    def _arbitrary_argument_generator(self):

        rng = np.random.default_rng(123)

        dim = 5
        lamb = .3

        for xtype in ['int', 'array']:
            if xtype == 'int':
                x = rng.integers(0, dim)
            else:
                x = rng.random(dim)
                x /= np.linalg.norm(x)
            for ytype in ['int', 'array']:
                if ytype == 'int':
                    y = rng.integers(0, dim)
                else:
                    y = rng.random(dim)
                    y /= np.linalg.norm(y)
            yield dim, lamb, x, y

    def test_amplitude_damping_qubit(self):
        from src.pyqch.channel_families import amplitude_damping
        
        # Expected behavior for single qubit
        lamb = .3
        tad_qubit = amplitude_damping(2, lamb, x=1, y=0)
        
        Kl = np.diag([1, np.sqrt(1-lamb)])
        Ll = np.array([[0, np.sqrt(lamb)],
                       [0, 0]])

        tad_qubit_ref = (np.kron(Kl, Kl)
                         + np.kron(Ll, Ll))
        
        np.testing.assert_array_almost_equal(tad_qubit, tad_qubit_ref)

    def test_amplitude_damping_computational(self):
        from src.pyqch.channel_families import amplitude_damping
                
        # Different input formats agree for states in computational basis
        dim = 5
        lamb = .3
        
        xint = 1
        xarr = np.zeros(dim)
        xarr[xint] = 1

        yint = 0
        yarr = np.zeros(dim)
        yarr[yint] = 1

        tad_comp = amplitude_damping(dim, lamb, x=xint, y=yint)

        tad_comp2 = amplitude_damping(dim, lamb, x=xarr, y=yarr)
        
        np.testing.assert_array_almost_equal(tad_comp, tad_comp2)
