from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FDMAS3D(nn.Module):
    def __init__(
        self,
        fs: float,
        f0: float,
        batch_size: int = 128,
        BW: float = 0.7,
        numtaps: int = 129,
        kaiser_beta: float = 8.6,
        eps: float = 1e-10,
        min_band_bins: int = 4,
    ):
        super().__init__()
        self.fs = fs
        self.f0 = f0
        self.batch_size = batch_size
        self.BW = BW
        self.numtaps = numtaps
        self.kaiser_beta = kaiser_beta
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
        if self.numtaps <= 1:
            raise ValueError(f"numtaps must be > 1, got {self.numtaps}")
        if self.numtaps % 2 == 0:
            raise ValueError(
                f"numtaps must be odd for same-length FIR filtering, got {self.numtaps}"
            )
        if self.kaiser_beta < 0:
            raise ValueError(f"kaiser_beta must be >= 0, got {self.kaiser_beta}")
        if self.min_band_bins < 1:
            raise ValueError(
                f"min_band_bins must be >= 1, got {self.min_band_bins}"
            )

    def _resolve_band(
        self, nz: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[float, float]:
        nyq = 0.5 * self.fs
        f_center = 2.0 * self.f0
        half_bw = 0.5 * self.BW * f_center

        f_low = max(0.0, f_center - half_bw)
        f_high = min(nyq, f_center + half_bw)
        if f_high <= f_low:
            raise ValueError(
                "Invalid F-DMAS FIR pass band after Nyquist clipping: "
                f"f_low={f_low}, f_high={f_high}, nyquist={nyq}."
            )

        freqs = torch.fft.rfftfreq(nz, d=1.0 / self.fs, device=device, dtype=dtype)
        n_bins = int(((freqs >= f_low) & (freqs <= f_high)).sum().item())
        if n_bins < self.min_band_bins:
            raise ValueError(
                "Insufficient FFT bins in F-DMAS FIR pass band: "
                f"{n_bins} < {self.min_band_bins}. "
                f"Try increasing nz or adjusting fs/f0/BW."
            )

        return f_low, f_high

    def _kaiser_window(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        center = (self.numtaps - 1) / 2.0
        n = torch.arange(self.numtaps, device=device, dtype=dtype)
        x = (n - center) / center

        beta = torch.tensor(self.kaiser_beta, device=device, dtype=dtype)
        numerator = torch.i0(beta * torch.sqrt(torch.clamp(1.0 - x**2, min=0.0)))
        denominator = torch.i0(beta)
        return numerator / denominator

    def _lowpass_kernel(
        self, cutoff: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        center = (self.numtaps - 1) / 2.0
        n = torch.arange(self.numtaps, device=device, dtype=dtype) - center
        normalized_cutoff = 2.0 * cutoff / self.fs
        return normalized_cutoff * torch.sinc(normalized_cutoff * n)

    def _normalize_passband_gain(
        self,
        kernel: torch.Tensor,
        f_low: float,
        f_high: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        n_fft = max(4096, 8 * self.numtaps)
        freqs = torch.fft.rfftfreq(
            n_fft, d=1.0 / self.fs, device=kernel.device, dtype=dtype
        )
        response = torch.abs(torch.fft.rfft(kernel, n=n_fft))
        mask = (freqs >= f_low) & (freqs <= f_high)
        if not torch.any(mask):
            return kernel

        max_gain = torch.max(response[mask])
        if max_gain.item() <= self.eps:
            return kernel
        return kernel / max_gain

    def _get_bp_filter(
        self, nz: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (
            nz,
            device,
            dtype,
            self.fs,
            self.f0,
            self.BW,
            self.numtaps,
            self.kaiser_beta,
        )
        if key in self._filter_cache:
            return self._filter_cache[key]

        f_low, f_high = self._resolve_band(nz, device, dtype)
        window = self._kaiser_window(device, dtype)

        high = self._lowpass_kernel(f_high, device, dtype)
        low = self._lowpass_kernel(f_low, device, dtype)
        kernel = (high - low) * window
        kernel = self._normalize_passband_gain(kernel, f_low, f_high, dtype)
        kernel = kernel.view(1, 1, -1)

        self._filter_cache[key] = kernel
        return kernel

    def _apply_filter(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        b, nz, nx = x.shape
        x = x.permute(0, 2, 1).reshape(b * nx, 1, nz)
        x = F.conv1d(x, kernel, padding=self.numtaps // 2)
        return x.reshape(b, nx, nz).permute(0, 2, 1)

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

        kernel = self._get_bp_filter(nz, y_dmas_base.device, y_dmas_base.dtype)
        return self._apply_filter(y_dmas_base, kernel)
