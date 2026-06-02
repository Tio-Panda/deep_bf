import numpy as np

def sharifzadeh_normalize(data, eps=1e-8):
    if data.ndim == 2:
        axes = (0, 1) # (nc, ns)
    elif data.ndim == 3:
        axes = (1, 2) # (na, nc, ns) -> por cada 'a'
    elif data.ndim == 4 and data.shape[-1] == 2:
        axes = (1, 2) # (na, nc, ns, 2) -> por cada (a, canal IQ)

    max_abs = np.max(np.abs(data), axis=axes, keepdims=True)
    data_norm = data / max_abs
    sigma = np.std(data_norm, axis=axes, keepdims=True) + eps
    return data_norm / sigma
