import numpy as np
from scipy import signal

def rf2iq(rf, fs, fc, decimation=4):
    f_mod = fc % fs
    f_effective = fs - f_mod if f_mod > fs/2 else f_mod
    # if f_mod > fs / 2:
    #     f_effective = fs - f_mod
    #     spectral_inversion = True
    # else:
    #     f_effective = f_mod
    #     spectral_inversion = False

    n_samples = rf.shape[-1]
    t = np.arange(n_samples) / fs
    mixer = np.exp(-1j * 2 * np.pi * f_effective * t)
    demodulated = rf * mixer
    iq = signal.decimate(demodulated, decimation, axis=-1, ftype="fir")

    i = iq.real.astype(np.float16)
    q = iq.imag.astype(np.float16)

    return np.stack((i, q), axis=-2)
