import numpy as np

def bcu_gm_el(dim: int):
    if dim == 2:
        return __get_pauli_unitary()
    elif dim == 3:
        return __get_gellmannd3_unitary()
    else:
        ugm = np.zeros((dim,)*4, dtype=complex)
        
        ugm[np.arange(dim), np.arange(dim), 0, 0] = 1 / np.sqrt(dim)
        
        for l in range(1, dim):
            for k in range(l):
                # x's
                ugm[k, l, k, l] = 1/np.sqrt(2)
                ugm[l, k, k, l] = 1/np.sqrt(2)

                # y's
                ugm[k, l, l, k] = -1j/np.sqrt(2)
                ugm[l, k, l, k] = 1j/np.sqrt(2)

                # z's
                ugm[l, l, l, l] = - l / np.sqrt(l*(l+1))
                ugm[np.arange(l), np.arange(l), l, l] = 1 / np.sqrt(l*(l+1))
                
        ugm = ugm.reshape((dim**2, dim**2))
        return ugm
    

def __get_pauli_unitary():
    upauli = np.zeros((4, 4), dtype=complex)
    upauli[:, 0] = np.array([1,0,0,1])
    upauli[:, 1] = np.array([0,1,1,0])
    upauli[:, 2] = np.array([0,-1j,1j,0])
    upauli[:, 3] = np.array([1,0,0,-1])
    return upauli / np.sqrt(2)


def __get_gellmannd3_unitary():
    gm3 = np.zeros((9, 9), dtype=complex)
    gm3[:, 0] = np.identity(3).reshape(9) * np.sqrt(2/3)
    
    gm3[[1, 3], 1] = 1
    gm3[[1, 3], 2] = [-1j, 1j]
    gm3[:, 3] = np.diag([1,-1,0]).reshape(9)

    gm3[[2, 6], 4] = 1
    gm3[[2, 6], 5] = [-1j, 1j]

    gm3[[5, 7], 6] = 1
    gm3[[5, 7], 7] = [-1j, 1j]

    gm3[:, 8] = np.diag([1, 1, -2]).reshape(9)/np.sqrt(3)

    return gm3 / np.sqrt(2)