import numpy as np
from scipy import signal

def rf2iq(rf, fs, fc, decimation=4):
    f_mod = fc % fs
    if f_mod > fs / 2:
        f_effective = fs - f_mod
        spectral_inversion = True
    else:
        f_effective = f_mod
        spectral_inversion = False

    n_samples = rf.shape[-1]
    t = np.arange(n_samples) / fs
    mixer = np.exp(-1j * 2 * np.pi * f_effective * t)
    demodulated = rf * mixer
    iq = signal.decimate(demodulated, decimation, axis=-1, ftype="fir")

    return iq
