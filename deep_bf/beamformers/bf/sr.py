from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SRBase(nn.Module, ABC):
    def __init__(
        self,
        batch_size: int = 32,
        num_iters: int = 20,
        lam: float = 1e-3,
        step: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_iters = num_iters
        self.lam = lam
        self.step = step
        self.eps = eps

    @abstractmethod
    def _malloc_output(self, b, nz, nx, device, dtype) -> torch.Tensor:
        pass

    @abstractmethod
    def _apod_tofc_data(self, tofc_data, apod) -> torch.Tensor:
        pass

    @abstractmethod
    def _soft_threshold(self, x: torch.Tensor, thr: float) -> torch.Tensor:
        pass

    def _das_mean(self, tofc_data: torch.Tensor) -> torch.Tensor:
        b, nc, nz, nx = tofc_data.shape[:4]
        output = self._malloc_output(b, nz, nx, tofc_data.device, tofc_data.dtype)

        for s in range(0, nc, self.batch_size):
            e = min(s + self.batch_size, nc)
            chunk = tofc_data[:, s:e, ...]
            output += torch.sum(chunk, dim=1)

        return output / nc

    def forward(self, tofc_data, apod):
        if tofc_data.dim() not in (4, 5):
            raise ValueError(
                f"Expected tofc_data to have 4 or 5 dims, got {tofc_data.dim()}"
            )

        y = self._das_mean(self._apod_tofc_data(tofc_data, apod))
        x = y.clone()
        thr = self.lam * self.step

        for _ in range(self.num_iters):
            grad = x - y
            z = x - self.step * grad
            x = self._soft_threshold(z, thr)

        return x


class SR3D(SRBase):
    def __init__(
        self,
        batch_size: int = 32,
        num_iters: int = 20,
        lam: float = 1e-3,
        step: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__(batch_size, num_iters, lam, step, eps)

    def _malloc_output(self, b, nz, nx, device, dtype):
        return torch.zeros(b, nz, nx, device=device, dtype=dtype)

    def _apod_tofc_data(self, tofc_data, apod):
        return tofc_data * apod.unsqueeze(0)

    def _soft_threshold(self, x: torch.Tensor, thr: float) -> torch.Tensor:
        return torch.sign(x) * torch.relu(torch.abs(x) - thr)


class SR4D(SRBase):
    def __init__(
        self,
        batch_size: int = 32,
        num_iters: int = 20,
        lam: float = 1e-3,
        step: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__(batch_size, num_iters, lam, step, eps)

    def _malloc_output(self, b, nz, nx, device, dtype):
        return torch.zeros(b, nz, nx, 2, device=device, dtype=dtype)

    def _apod_tofc_data(self, tofc_data, apod):
        return tofc_data * apod.unsqueeze(0).unsqueeze(-1)

    def _soft_threshold(self, x: torch.Tensor, thr: float) -> torch.Tensor:
        mag = torch.sqrt(torch.sum(x**2, dim=-1) + self.eps)
        scale = torch.relu(mag - thr) / (mag + self.eps)
        return x * scale.unsqueeze(-1)
