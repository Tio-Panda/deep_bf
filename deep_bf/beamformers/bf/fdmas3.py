from __future__ import annotations

import torch
import torch.nn as nn

from ..apod.apod import get_window

class FDMAS3D(nn.Module):
    def __init__(
        self,
        fs: float,
        f0: float,
        batch_size: int = 128,
        BW: float = 0.7,
        bp_window: str = "tukey50",
        eps: float = 1e-10,
        min_band_bins: int = 4,
    ):
        super().__init__()
        self.fs = fs
        self.f0 = f0
        self.batch_size = batch_size
        self.BW = BW
        self.bp_window = bp_window
        self.eps = eps
        self.min_band_bins = min_band_bins
        self._filter_cache: dict[tuple, torch.Tensor] = {}

    def _validate(self):
        if self.fs <= 0:
            raise ValueError(f"fs must be > 0, got {self.fs}")
        if self.f0 <= 0:
            raise ValueError(f"f0 must be > 0, got {self.f0}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.BW <= 0:
            raise ValueError(f"BW must be > 0, got {self.BW}")
        if self.min_band_bins < 1:
            raise ValueError(
                f"min_band_bins must be >= 1, got {self.min_band_bins}"
            )

    def _get_bp_filter(
        self, nz: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (nz, device, dtype, self.fs, self.f0, self.BW, self.bp_window)
        if key in self._filter_cache:
            return self._filter_cache[key]

        nyq = 0.5 * self.fs
        f_center = 2.0 * self.f0
        half_bw = 0.5 * self.BW * f_center

        f_low = max(0.0, f_center - half_bw)
        f_high = min(nyq, f_center + half_bw)
        if f_high <= f_low:
            raise ValueError(
                "Invalid F-DMAS pass band after Nyquist clipping: "
                f"f_low={f_low}, f_high={f_high}, nyquist={nyq}."
            )

        freqs = torch.fft.rfftfreq(nz, d=1.0 / self.fs, device=device, dtype=dtype)
        mask = (freqs >= f_low) & (freqs <= f_high)
        idx = torch.where(mask)[0]
        if idx.numel() < self.min_band_bins:
            raise ValueError(
                "Insufficient FFT bins in F-DMAS pass band: "
                f"{idx.numel()} < {self.min_band_bins}. "
                f"Try increasing nz or adjusting fs/f0/BW."
            )

        H = torch.zeros_like(freqs, dtype=dtype)
        band_freqs = freqs[idx]
        aperture = 0.5 * (f_high - f_low)
        center = 0.5 * (f_high + f_low)
        distance = torch.abs(band_freqs - center)
        H[idx] = get_window(distance, aperture, kind=self.bp_window).to(dtype)

        self._filter_cache[key] = H
        return H

    def forward(self, tofc_data: torch.Tensor) -> torch.Tensor:
        if tofc_data.dim() != 4:
            raise ValueError(
                f"Expected RF 3D tensor [B, nc, nz, nx], got {tuple(tofc_data.shape)}"
            )

        self._validate()

        b, nc, nz, nx = tofc_data.shape
        weighted = tofc_data

        sum_s_hat = torch.zeros(b, nz, nx, device=weighted.device, dtype=weighted.dtype)
        sum_abs_s = torch.zeros(b, nz, nx, device=weighted.device, dtype=weighted.dtype)

        for s in range(0, nc, self.batch_size):
            e = min(s + self.batch_size, nc)
            chunk = weighted[:, s:e, ...]

            s_hat = torch.sign(chunk) * torch.sqrt(torch.abs(chunk) + self.eps)
            sum_s_hat += torch.sum(s_hat, dim=1)
            sum_abs_s += torch.sum(torch.abs(chunk), dim=1)

        y_dmas_base = 0.5 * (sum_s_hat**2 - sum_abs_s)

        H = self._get_bp_filter(nz, y_dmas_base.device, y_dmas_base.dtype)
        y_fft = torch.fft.rfft(y_dmas_base, dim=1)
        y_fft = y_fft * H[None, :, None]
        y_fdmas = torch.fft.irfft(y_fft, n=nz, dim=1).real

        return y_fdmas
