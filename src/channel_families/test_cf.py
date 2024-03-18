import unittest
import numpy as np

class TestChannelFamilies(unittest.TestCase):

    def test_depolarizing(self):
        from src.channel_families.cf_depolarizing import cf_depolarizing

        dim = 3
        p = .67
        r = np.identity(dim) / dim

        rho_in = np.diag(np.arange(1, 1+dim, dtype=float))
        rho_in /= np.trace(rho_in)

        dmat = cf_depolarizing(dim, p, r)

        rho_out = (dmat @ rho_in.reshape(dim**2)).reshape((dim, dim))

        self.assertTrue(np.allclose(rho_out, (1-p) * rho_in + p * r))
    

    def test_amplitude_damping(self):
        from src.channel_families.cf_amplitude_damping import cf_amplitude_damping

        dim = 3
        g = .33

        rho_in = np.diag(np.arange(1, 1+dim, dtype=float))
        rho_in /= np.trace(rho_in)

        rho_out_ref = np.diag(np.arange(2, 2+dim, dtype=float))
        rho_out_ref[dim-1, dim-1] = 0
        rho_out_ref[0,0] += 1
        rho_out_ref /= np.trace(rho_out_ref)

        rho_out_ref = (1-g) * rho_in + g * rho_out_ref

        dmat = cf_amplitude_damping(dim, g)
        rho_out = (dmat @ rho_in.reshape(dim**2)).reshape((dim, dim))

        self.assertTrue(np.allclose(rho_out, rho_out_ref))

    def test_povm_computational_basis(self):
        # POs for the coputational basis
        dim = 8
        pos = np.zeros((dim, dim, dim))
        for x in range(dim):
            pos[x,x,x] = 1

        # POVM mode consistency
        self.__povm_mode_consistency(dim, pos)

    def __povm_mode_consistency(self, dim, pos):
        from src.channel_families.cf_povm import cf_povm
        # Checks whether different modes of POVM are consistent
        m = pos.shape[0]

        # Create input state
        rho_in = np.diag(np.arange(1, 1+dim, dtype=float))
        rho_in /= np.trace(rho_in)

        # Create POVM matrices
        qqc_mat = cf_povm(dim, pos, 'q-qc')
        qc_mat = cf_povm(dim, pos, 'q-c')
        qq_mat = cf_povm(dim, pos, 'q-q')

        r_qc = (qqc_mat @ rho_in.reshape((dim**2,))).reshape((m, dim, m ,dim))
        r_q = (qq_mat @ rho_in.reshape((dim**2,))).reshape((dim, dim))
        r_c = (qc_mat @ rho_in.reshape((dim**2,))).reshape((m, m))

        r_qc_q = np.einsum("ijil->jl", r_qc)
        r_qc_c = np.einsum("ijlj->il", r_qc)

        self.assertTrue(np.allclose(r_q,r_qc_q), msg="qc->q Inconsistency.")
        self.assertTrue(np.allclose(r_c,r_qc_c), msg="qc->c Inconsistency.")

    def test_cf_dephasing(self):
        from src.channel_families.cf_dephasing import cf_dephasing
        dim = 3
        g = .33

        # Create input state
        rho_in = np.full((dim, dim), 1/dim)

        # Create reference state
        rho_out_ref = g * rho_in + (1-g) * np.diag(np.diag(rho_in))

        # Test the action of the channel
        dephas_mat = cf_dephasing(dim, g)
        rho_out = (dephas_mat @ rho_in.reshape(dim**2)).reshape((dim, dim))

        self.assertTrue(np.allclose(rho_out, rho_out_ref))

    def test_cf_initializer(self):
        from src.channel_families.cf_initializer import cf_initializer

        dim = 3
        state_list = []
        state_list.append(np.full((dim, dim), 1/dim))
        state_list.append(np.diag(2/dim/(dim+1)*np.arange(1, 1+dim, dtype=float)))
        states = np.array(state_list)

        # Define an input probability distribution
        p_in = np.arange(1, 1+states.shape[0]) * 2 /states.shape[0]/(states.shape[0]+1)

        init_mat = cf_initializer(dim, states, mode='c-q')

        rho_out = (init_mat @ p_in).reshape((dim, dim))

        rho_out_ref = np.sum([p_in[k] * state_list[k] for k in range(len(state_list))], axis=0)
        self.assertTrue(np.allclose(rho_out, rho_out_ref))
    
    def test_cf_probabilistic_unitaries(self):
        from src.channel_families.cf_probabilistic_unitaries import cf_probabilistic_unitaries

        dim = 3

        u_list = []

        u_list.append(np.identity(dim))
        u_list.append(np.diag(np.exp(-1j * np.linspace(0, np.pi, dim))))
        u_list.append(np.diag(np.exp(-1j * np.linspace(0, .3*np.pi, dim))))

        us = np.array(u_list)
        probs = np.arange(1, 1+us.shape[0]) * 2/(1+us.shape[0])/us.shape[0]

        rho_in = np.full((dim, dim), 1/dim)

        pu_mat = cf_probabilistic_unitaries(dim, probs, us)

        rho_out = (pu_mat @ rho_in.reshape(dim**2)).reshape((dim, dim))

        rho_out_ref = np.sum([probs[k] * us[k] @ rho_in @ us[k].T.conj() for k in range(len(u_list))], axis=0)
        self.assertTrue(np.allclose(rho_out, rho_out_ref))

    def test_embed_classical(self):
        from src.channel_families.cf_embed_classical import cf_embed_classical

        dim = 3
        stoch = np.outer(np.arange(1, dim+1, dtype=float), np.arange(1, dim+1, dtype=float))
        stoch /= np.sum(stoch, axis=1)

        p_in = np.full(dim, 1/dim)
        rho_in = np.full((dim, dim), 1/dim)

        rho_p_out = np.diag(stoch @ p_in)
        rho_out = (cf_embed_classical(dim, stoch) @ rho_in.reshape(dim**2)).reshape((dim, dim))

        self.assertTrue(np.allclose(rho_p_out, rho_out))

    def test_classical_permutation(self):
        from src.channel_families.cf_classical_permutation import cf_classical_permutaiton

        dim = 3
        shift = 1
        perm = np.roll(np.arange(dim), -shift)

        p_in = np.arange(dim)

        p_out_ref = np.roll(p_in, shift)

        p_out = cf_classical_permutaiton(dim, perm) @ p_in

        self.assertTrue(np.allclose(p_out, p_out_ref))


if __name__ == "__main__":
    from cf_classical_permutation import cf_classical_permutaiton

    dim = 3
    shift = 1
    perm = np.roll(np.arange(dim), -shift)

    p_in = np.arange(dim)

    p_out_ref = np.roll(p_in, shift)

    p_out = cf_classical_permutaiton(dim, perm) @ p_in

    print(p_out_ref)
    print(p_out)
    print(p_in)
    print(perm)