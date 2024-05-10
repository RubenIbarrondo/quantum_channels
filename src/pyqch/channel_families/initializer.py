import numpy as np

def initializer(dim: int, states: np.ndarray, mode='c-q'):
    m = states.shape[0]
    a = np.identity(m)

    if mode == 'q-qc':
        mat = np.einsum('ij,ik,kl,kpq->ipjqkl', a, a, a, states)
        mat = mat.reshape(((m*dim)**2, m**2))
    elif mode == 'q-q':
        mat = np.einsum('kl,kpq->pqkl', a, states)
        mat = mat.reshape((dim**2, m**2))

    elif mode == 'c-qc':
        mat = np.einsum('ij,ik,kpq->ipjqk', a, a, states)
        mat = mat.reshape(((m*dim)**2, m))
    elif mode == 'c-q':
        mat = np.einsum('kpq->pqk', states)
        mat = mat.reshape((dim**2, m))
    
    return mat