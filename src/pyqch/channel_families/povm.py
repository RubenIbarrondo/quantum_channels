import numpy as np

def povm(dim: int, pos: np.ndarray, mode='q-q'):
    # Verify inputs
    # Cast pos to np array with proper shape

    # mode is either 'q-q', 'q-c', or 'q-qc'
    m = pos.shape[0]
    a = np.identity(m)

    if mode == 'q-q':
        mat = np.einsum('kij,klm->iljm', pos, pos.conj())
        mat = mat.reshape((dim**2, dim**2))
    elif mode == 'q-c':
        mat = np.einsum('ki,ksl,kj,kso->ijlo', a, pos, a.conj(), pos.conj())
        mat = mat.reshape((m**2, dim**2))
    elif mode == 'q-qc':
        mat = np.einsum('mi,mjk,ml,mno->ijlnko', a,pos, a, pos.conj())
        mat = mat.reshape(((m*dim)**2, dim**2))
    return mat