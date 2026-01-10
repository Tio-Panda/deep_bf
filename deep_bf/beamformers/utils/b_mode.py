import numpy as np
from scipy.signal import hilbert

def get_rf_bmode(data, vmin=-60, vmax=0, eps=1e-10):
    env = np.abs(hilbert(data, axis=0))
    b_mode = 20 * np.log10(env + eps)
    b_mode -= np.amax(b_mode)
    b_mode = np.clip(b_mode, vmin, vmax)

    return b_mode
