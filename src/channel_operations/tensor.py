import numpy as np

def tensor(t_arr, n=None):
    # n is only used if t_arr is not a list
    if isinstance(t_arr, list) and len(t_arr) == 2:
        t = t_arr[0]
        g = t_arr[1]
        d1 = int(np.sqrt(t.shape[0]))
        d2 = int(np.sqrt(g.shape[0]))
        
        tres = t.reshape((d1, d1, d1, d1))
        gres = g.reshape((d2, d2, d2 ,d2))
            
        tg = np.einsum("ijkl,mnop->imjnkolp", tres, gres)

        return tg.reshape(((d1*d2)**2, (d1*d2)**2))
    elif isinstance(t_arr, list) and len(t_arr) > 2:
        return co_tensor([t_arr[0], co_tensor(t_arr[1:])])
    elif n is not None:
        if n==1:
            return t_arr
        else:
            return co_tensor([t_arr]*n)
    else:
        raise ValueError()
