import torch
import torch.fft

from .apod import get_window

def get_freqs(fs, f0, BW=0.7, for_dmas=True):
    if for_dmas:
        f_center = 2 * f0
    else:
        f_center = f0

    f_low = f_center * (1 - BW/2)
    f_high = f_center * (1 + BW/2)

    return fs, f_low, f_high

def pass_band_filter(N, freqs, window, device="cuda", dtype=torch.float32):
    fs, f_low, f_high = freqs
    freqs = torch.fft.fftfreq(N, d=1/fs, device=device, dtype=dtype)
    freqs = torch.abs(freqs)

    f_center = (f_high + f_low) / 2
    f_bandwidth = f_high - f_low

    distance = torch.abs(freqs - f_center)
    aperture = f_bandwidth / 2

    H = get_window(distance, aperture, window)
    mask = (freqs >= f_low) & (freqs <= f_high)
    H *= mask.float()

    return H
