import numpy as np

def co_tensor(t_arr, n=None):
    # n is only used if ts is not a list
    if isinstance(t_arr, list) and len(t_arr) == 2:
        t = t_arr[0]
        g = t_arr[1]
        td1 = int(np.sqrt(t.shape[1]))
        td2 = int(np.sqrt(t.shape[0]))
        gd1 = int(np.sqrt(g.shape[1]))
        gd2 = int(np.sqrt(g.shape[0]))
        
        tres = t.reshape((td2, td2, td1, td1))
        gres = g.reshape((gd2, gd2, gd1, gd1))
            
        tg = np.einsum("ijkl,mnop->imjnkolp", tres, gres)

        return tg.reshape(((td2*gd2)**2, (td1*gd1)**2))
    
    elif isinstance(t_arr, list) and len(t_arr) > 2:
        return co_tensor([t_arr[0], co_tensor(t_arr[1:])])
    elif n is not None:
        if n==1:
            return t_arr
        else:
            return co_tensor([t_arr]*n)
    else:
        raise ValueError()