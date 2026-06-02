import numpy as np

def padding_input(data, new_ns, mode):
    ns = data.shape[1]
    pad_width = ((0,0), (0, new_ns - ns)) if mode == "RF" else ((0,0), (0, new_ns - ns), (0, 0))
    if ns <= new_ns:
        data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
    else:
        data = data[:, :new_ns]
    
    return data
