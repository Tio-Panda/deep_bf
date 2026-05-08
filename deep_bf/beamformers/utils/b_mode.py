import numpy as np
from scipy.signal import hilbert
from ...constants.bf import PWDataType

def get_bmode(data, mode, vmin=-60, vmax=0, eps=1e-10):
    if mode == PWDataType.RF:
        env = np.abs(hilbert(data, axis=0))
    elif mode == PWDataType.IQ_COMPLEX or PWDataType.IQ_COMPLEX_DEMOD:
        env = np.abs(data)
    else:
        env = np.linalg.norm(data, axis=-1)  # sqrt(I^2 + Q^2)

    b_mode = 20 * np.log10(env + eps)
    b_mode -= np.amax(b_mode)
    b_mode = np.clip(b_mode, vmin, vmax)

    return b_mode
