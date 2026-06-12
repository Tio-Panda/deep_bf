from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class RFUpsampler1D(nn.Module):
    def __init__(
        self,
        factor: int = 2,
        numtaps: int = 65,
        window: str = "hamming",
    ):
        super().__init__()
        self.factor = factor
        self.numtaps = numtaps
        self.window = window
        self._kernel_cache: dict[tuple, torch.Tensor] = {}

    def _validate(self):
        if not isinstance(self.factor, int) or isinstance(self.factor, bool):
            raise ValueError(f"factor must be an int, got {self.factor}")
        if self.factor < 1:
            raise ValueError(f"factor must be >= 1, got {self.factor}")
        if self.numtaps <= 1:
            raise ValueError(f"numtaps must be > 1, got {self.numtaps}")
        if self.numtaps % 2 == 0:
            raise ValueError(
                f"numtaps must be odd for same-length FIR filtering, got {self.numtaps}"
            )

    def _get_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.window == "boxcar":
            return torch.ones(self.numtaps, device=device, dtype=dtype)
        if self.window == "hanning":
            return torch.hann_window(
                self.numtaps, periodic=False, device=device, dtype=dtype
            )
        if self.window == "hamming":
            return torch.hamming_window(
                self.numtaps, periodic=False, device=device, dtype=dtype
            )

        raise ValueError(
            f"Unsupported window '{self.window}'. Expected 'boxcar', 'hanning' or 'hamming'."
        )

    def _get_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype, self.factor, self.numtaps, self.window)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        center = (self.numtaps - 1) / 2.0
        n = torch.arange(self.numtaps, device=device, dtype=dtype) - center
        normalized_cutoff = 1.0 / self.factor

        kernel = normalized_cutoff * torch.sinc(normalized_cutoff * n)
        kernel = kernel * self._get_window(device, dtype)
        kernel = kernel * self.factor
        kernel = kernel.view(1, 1, -1)

        self._kernel_cache[key] = kernel
        return kernel

    def forward(self, rf: torch.Tensor) -> torch.Tensor:
        self._validate()
        if rf.dim() != 3:
            raise ValueError(f"Expected RF tensor [B, nc, ns], got {tuple(rf.shape)}")
        if self.factor == 1:
            return rf

        b, nc, ns = rf.shape
        x = rf.reshape(b * nc, 1, ns)
        upsampled = x.new_zeros(b * nc, 1, ns * self.factor)
        upsampled[:, :, :: self.factor] = x

        kernel = self._get_kernel(upsampled.device, upsampled.dtype)
        upsampled = F.conv1d(upsampled, kernel, padding=self.numtaps // 2)
        return upsampled.reshape(b, nc, ns * self.factor)
