from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..apod.apod import get_window


class FDMASBase(nn.Module, ABC):
    def __init__(
        self,
        fs: float,
        f0: float,
        batch_size: int = 32,
        BW: float = 0.7,
        bp_window: str = "tukey50",
        eps: float = 1e-10,
        fallback_mode: str = "auto",
        allow_asymmetric: bool = True,
        nyquist_margin: float = 0.995,
        min_band_bins: int = 4,
        strict_if_no_valid_band: bool = False,
    ):
        super().__init__()
        self.fs = fs
        self.f0 = f0
        self.batch_size = batch_size
        self.BW = BW
        self.bp_window = bp_window
        self.eps = eps
        self.fallback_mode = fallback_mode
        self.allow_asymmetric = allow_asymmetric
        self.nyquist_margin = nyquist_margin
        self.min_band_bins = min_band_bins
        self.strict_if_no_valid_band = strict_if_no_valid_band
        self._filter_cache: dict[tuple, torch.Tensor] = {}
        self.last_filter_info: dict[str, float | str | int | None] = {
            "mode": "uninitialized",
            "f_center_requested": None,
            "f_low": None,
            "f_high": None,
            "nyquist": None,
            "n_bins": None,
        }

    @abstractmethod
    def _malloc_output(self, b, nz, nx, device, dtype) -> torch.Tensor:
        pass

    def _validate_filter_params(self):
        valid_modes = {"auto", "force_2f0", "force_f0"}
        if self.fallback_mode not in valid_modes:
            raise ValueError(
                f"Invalid fallback_mode='{self.fallback_mode}'. "
                f"Expected one of {sorted(valid_modes)}."
            )
        if self.fs <= 0:
            raise ValueError(f"fs must be > 0, got {self.fs}")
        if self.f0 <= 0:
            raise ValueError(f"f0 must be > 0, got {self.f0}")
        if self.BW <= 0:
            raise ValueError(f"BW must be > 0, got {self.BW}")
        if self.nyquist_margin <= 0 or self.nyquist_margin > 1:
            raise ValueError(
                f"nyquist_margin must be in (0, 1], got {self.nyquist_margin}"
            )
        if self.min_band_bins < 1:
            raise ValueError(f"min_band_bins must be >= 1, got {self.min_band_bins}")

    def _nominal_band(self, f_center: float) -> tuple[float, float]:
        half_bw = 0.5 * self.BW * f_center
        return f_center - half_bw, f_center + half_bw

    def _clip_or_reject_band(
        self, f_low_nom: float, f_high_nom: float, nyq: float
    ) -> tuple[float | None, float | None, bool, bool]:
        if self.allow_asymmetric:
            f_low = max(0.0, f_low_nom)
            f_high = min(nyq, f_high_nom)
            clipped = (f_low != f_low_nom) or (f_high != f_high_nom)
            return f_low, f_high, clipped, True

        if f_low_nom < 0.0 or f_high_nom > nyq:
            return None, None, False, False
        return f_low_nom, f_high_nom, False, True

    def _resolve_band(
        self, nz: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, dict[str, float | str | int | None], bool]:
        nyq = 0.5 * self.fs
        freqs = torch.fft.rfftfreq(nz, d=1.0 / self.fs, device=device, dtype=dtype)

        if self.fallback_mode == "force_2f0":
            candidates = [("2f0", 2.0 * self.f0)]
        elif self.fallback_mode == "force_f0":
            candidates = [("f0", self.f0)]
        else:
            candidates = [("2f0", 2.0 * self.f0), ("f0", self.f0)]

        for label, f_center in candidates:
            if self.fallback_mode == "auto" and f_center >= nyq * self.nyquist_margin:
                continue

            f_low_nom, f_high_nom = self._nominal_band(f_center)
            f_low, f_high, clipped, ok_range = self._clip_or_reject_band(
                f_low_nom, f_high_nom, nyq
            )
            if not ok_range or f_low is None or f_high is None or f_high <= f_low:
                continue

            band_mask = (freqs >= f_low) & (freqs <= f_high)
            n_bins = int(band_mask.sum().item())
            if n_bins < self.min_band_bins:
                continue

            mode = label + ("_asymmetric" if clipped else "")
            info = {
                "mode": mode,
                "f_center_requested": f_center,
                "f_low": f_low,
                "f_high": f_high,
                "nyquist": nyq,
                "n_bins": n_bins,
            }
            return freqs, info, True

        info = {
            "mode": "identity_fallback",
            "f_center_requested": None,
            "f_low": None,
            "f_high": None,
            "nyquist": nyq,
            "n_bins": None,
        }
        return freqs, info, False

    def _build_one_sided_filter(
        self,
        freqs: torch.Tensor,
        f_low: float,
        f_high: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        H = torch.zeros_like(freqs, dtype=dtype)
        mask = (freqs >= f_low) & (freqs <= f_high)
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            return H

        band_freqs = freqs[idx]
        f_center = 0.5 * (f_low + f_high)
        half_bw = 0.5 * (f_high - f_low)
        distance = torch.abs(band_freqs - f_center)
        window_vals = get_window(distance, half_bw, kind=self.bp_window).to(dtype)
        H[idx] = window_vals
        return H

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
            self.bp_window,
            self.fallback_mode,
            self.allow_asymmetric,
            self.nyquist_margin,
            self.min_band_bins,
        )
        if key in self._filter_cache:
            return self._filter_cache[key]

        self._validate_filter_params()
        freqs, info, has_valid_band = self._resolve_band(nz, device, dtype)
        self.last_filter_info = info

        if not has_valid_band:
            if self.strict_if_no_valid_band:
                raise ValueError(
                    "No valid pass-band found for current fs/f0/BW configuration. "
                    f"Details: {info}"
                )
            H = torch.ones_like(freqs, dtype=dtype)
        else:
            f_low = float(info["f_low"])
            f_high = float(info["f_high"])
            H = self._build_one_sided_filter(freqs, f_low, f_high, dtype)

        self._filter_cache[key] = H
        return H

    def _apply_filter(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.rfft(x, dim=1)
        if x.dim() == 3:
            x_fft = x_fft * H[None, :, None]
        else:
            x_fft = x_fft * H[None, :, None, None]
        return torch.fft.irfft(x_fft, n=x.shape[1], dim=1).real

    def forward(self, tofc_data):
        if tofc_data.dim() not in (4, 5):
            raise ValueError(
                f"Expected tofc_data to have 4 or 5 dims, got {tofc_data.dim()}"
            )

        b, nc, nz, nx = tofc_data.shape[:4]
        sum_s_hat = self._malloc_output(b, nz, nx, tofc_data.device, tofc_data.dtype)
        sum_abs_s = self._malloc_output(b, nz, nx, tofc_data.device, tofc_data.dtype)

        for s in range(0, nc, self.batch_size):
            e = min(s + self.batch_size, nc)
            chunk = tofc_data[:, s:e, ...]

            s_hat = torch.sign(chunk) * torch.sqrt(torch.abs(chunk))
            sum_s_hat += torch.sum(s_hat, dim=1)
            sum_abs_s += torch.sum(torch.abs(chunk), dim=1)

        fdmas = 0.5 * (sum_s_hat**2 - sum_abs_s)
        H = self._get_bp_filter(nz, fdmas.device, fdmas.dtype)
        fdmas = self._apply_filter(fdmas, H)

        return fdmas


class FDMAS3D(FDMASBase):
    def _malloc_output(self, b, nz, nx, device, dtype):
        return torch.zeros(b, nz, nx, device=device, dtype=dtype)


class FDMAS4D(FDMASBase):
    def _malloc_output(self, b, nz, nx, device, dtype):
        return torch.zeros(b, nz, nx, 2, device=device, dtype=dtype)
