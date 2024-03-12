import numpy as np

def co_choi_state(t):
    d2 = int(np.sqrt(t.shape[0]))
    d1 = int(np.sqrt(t.shape[1]))

    t = t.reshape((d2, d2, d1, d1))

    choi = 1/d1 * t.transpose((0, 2, 1, 3)).reshape((d1*d2, d1*d2))
    return choi