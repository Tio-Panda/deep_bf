from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FDMAS3D(nn.Module):
    def __init__(
        self,
        fs: float,
        f0: float,
        batch_size: int = 128,
        passband_low: float = 8e6,
        passband_high: float = 13e6,
        stopband_low: float = 5e6,
        stopband_high: float = 15e6,
        ripple: float = 1e-3,
        numtaps: int | None = None,
        kaiser_beta: float | None = None,
        eps: float = 1e-10,
        min_band_bins: int = 4,
    ):
        super().__init__()
        self.fs = fs
        self.f0 = f0
        self.batch_size = batch_size
        self.passband_low = passband_low
        self.passband_high = passband_high
        self.stopband_low = stopband_low
        self.stopband_high = stopband_high
        self.ripple = ripple
        self.numtaps = numtaps
        self.kaiser_beta = kaiser_beta
        self.eps = eps
        self.min_band_bins = min_band_bins
        self._filter_cache: dict[tuple, torch.Tensor] = {}

    def _attenuation_db(self) -> float:
        return -20.0 * math.log10(self.ripple)

    def _effective_kaiser_beta(self) -> float:
        if self.kaiser_beta is not None:
            return self.kaiser_beta

        attenuation = self._attenuation_db()
        if attenuation > 50.0:
            return 0.1102 * (attenuation - 8.7)
        if attenuation >= 21.0:
            return 0.5842 * (attenuation - 21.0) ** 0.4 + 0.07886 * (
                attenuation - 21.0
            )
        return 0.0

    def _transition_width(self) -> float:
        low_transition = self.passband_low - self.stopband_low
        high_transition = self.stopband_high - self.passband_high
        return min(low_transition, high_transition)

    def _effective_numtaps(self) -> int:
        if self.numtaps is not None:
            return self.numtaps

        attenuation = self._attenuation_db()
        transition = self._transition_width()
        normalized_transition = transition / self.fs
        order = math.ceil(
            (attenuation - 7.95)
            / (2.0 * math.pi * 2.285 * normalized_transition)
        )
        return max(order + 1, 2)

    def _fir1_cutoffs(self) -> tuple[float, float]:
        return (
            0.5 * (self.stopband_low + self.passband_low),
            0.5 * (self.passband_high + self.stopband_high),
        )

    def _validate(self):
        if self.fs <= 0:
            raise ValueError(f"fs must be > 0, got {self.fs}")
        if self.f0 <= 0:
            raise ValueError(f"f0 must be > 0, got {self.f0}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.ripple <= 0 or self.ripple >= 1:
            raise ValueError(f"ripple must be in (0, 1), got {self.ripple}")
        if not (
            0.0
            <= self.stopband_low
            < self.passband_low
            < self.passband_high
            < self.stopband_high
        ):
            raise ValueError(
                "Expected 0 <= stopband_low < passband_low < passband_high < stopband_high, "
                f"got stopband_low={self.stopband_low}, passband_low={self.passband_low}, "
                f"passband_high={self.passband_high}, stopband_high={self.stopband_high}."
            )

        nyq = 0.5 * self.fs
        if self.stopband_high >= nyq:
            raise ValueError(
                "stopband_high must be below Nyquist. "
                f"stopband_high={self.stopband_high}, nyquist={nyq}. "
                "If this is PICMUS F-DMAS, make sure RF upsampling is active so fs is doubled."
            )

        if self.min_band_bins < 1:
            raise ValueError(
                f"min_band_bins must be >= 1, got {self.min_band_bins}"
            )

        taps = self._effective_numtaps()
        if taps <= 1:
            raise ValueError(f"numtaps must be > 1, got {taps}")

        beta = self._effective_kaiser_beta()
        if beta < 0:
            raise ValueError(f"kaiser_beta must be >= 0, got {beta}")

    def _validate_band_bins(
        self, nz: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        freqs = torch.fft.rfftfreq(nz, d=1.0 / self.fs, device=device, dtype=dtype)
        n_bins = int(
            ((freqs >= self.passband_low) & (freqs <= self.passband_high)).sum().item()
        )
        if n_bins < self.min_band_bins:
            raise ValueError(
                "Insufficient FFT bins in F-DMAS FIR pass band: "
                f"{n_bins} < {self.min_band_bins}. "
                "Try increasing nz or adjusting passband/fs."
            )

    def _kaiser_window(
        self, numtaps: int, beta_value: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        center = (numtaps - 1) / 2.0
        n = torch.arange(numtaps, device=device, dtype=dtype)
        x = (n - center) / center

        beta = torch.tensor(beta_value, device=device, dtype=dtype)
        numerator = torch.i0(beta * torch.sqrt(torch.clamp(1.0 - x**2, min=0.0)))
        denominator = torch.i0(beta)
        return numerator / denominator

    def _lowpass_kernel(
        self, cutoff: float, numtaps: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        center = (numtaps - 1) / 2.0
        n = torch.arange(numtaps, device=device, dtype=dtype) - center
        normalized_cutoff = 2.0 * cutoff / self.fs
        return normalized_cutoff * torch.sinc(normalized_cutoff * n)

    def _get_bp_filter(
        self, nz: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        numtaps = self._effective_numtaps()
        beta = self._effective_kaiser_beta()
        low_cutoff, high_cutoff = self._fir1_cutoffs()
        key = (
            nz,
            device,
            dtype,
            self.fs,
            self.passband_low,
            self.passband_high,
            self.stopband_low,
            self.stopband_high,
            self.ripple,
            numtaps,
            beta,
            low_cutoff,
            high_cutoff,
        )
        if key in self._filter_cache:
            return self._filter_cache[key]

        self._validate_band_bins(nz, device, dtype)
        window = self._kaiser_window(numtaps, beta, device, dtype)

        high = self._lowpass_kernel(high_cutoff, numtaps, device, dtype)
        low = self._lowpass_kernel(low_cutoff, numtaps, device, dtype)
        kernel = (high - low) * window
        kernel = kernel.view(1, 1, -1)

        self._filter_cache[key] = kernel
        return kernel

    def _apply_filter(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        b, nz, nx = x.shape
        x = x.permute(0, 2, 1).reshape(b * nx, 1, nz)
        taps = kernel.shape[-1]
        pad_left = (taps - 1) // 2
        pad_right = taps // 2
        x = F.pad(x, (pad_left, pad_right))
        x = F.conv1d(x, kernel)
        return x.reshape(b, nx, nz).permute(0, 2, 1)

    def forward(
        self, tofc_data: torch.Tensor, apod: torch.Tensor | None = None
    ) -> torch.Tensor:
        if tofc_data.dim() != 4:
            raise ValueError(
                f"Expected RF 3D tensor [B, nc, nz, nx], got {tuple(tofc_data.shape)}"
            )

        self._validate()

        b, nc, nz, nx = tofc_data.shape
        if apod is None:
            weighted = tofc_data
        else:
            weighted = tofc_data * apod.unsqueeze(0)

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
