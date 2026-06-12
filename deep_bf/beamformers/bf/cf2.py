from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class CFBase(nn.Module, ABC):
    def __init__(self, batch_size: int = 32, eps: float = 1e-8):
        super().__init__()
        self.batch_size = batch_size
        self.eps = eps

    @abstractmethod
    def _malloc_das_output(self, b: int, nz: int, nx: int, device, dtype) -> torch.Tensor:
        pass

    @abstractmethod
    def _compute_coherent_power(self, das: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _accumulate_incoherent_power(self, chunk: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _compute_output(self, cf: torch.Tensor, das: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, tofc_data: torch.Tensor) -> torch.Tensor:
        b, nc, nz, nx = tofc_data.shape[:4]

        das = self._malloc_das_output(b, nz, nx, tofc_data.device, tofc_data.dtype)
        incoherent_power = torch.zeros(
            b,
            nz,
            nx,
            device=tofc_data.device,
            dtype=tofc_data.real.dtype,
        )

        for s in range(0, nc, self.batch_size):
            e = min(s + self.batch_size, nc)
            chunk = tofc_data[:, s:e, ...]

            das += torch.sum(chunk, dim=1)
            incoherent_power += self._accumulate_incoherent_power(chunk)

        coherent_power = self._compute_coherent_power(das)
        cf = coherent_power / (nc * incoherent_power + self.eps)

        return self._compute_output(cf, das)


class CF3D(CFBase):
    def _malloc_das_output(self, b, nz, nx, device, dtype):
        return torch.zeros(b, nz, nx, device=device, dtype=dtype)

    def _compute_coherent_power(self, das):
        return torch.abs(das) ** 2

    def _accumulate_incoherent_power(self, chunk):
        return torch.sum(torch.abs(chunk) ** 2, dim=1)

    def _compute_output(self, cf, das):
        return cf * das


class CF4D(CFBase):
    def _malloc_das_output(self, b, nz, nx, device, dtype):
        return torch.zeros(b, nz, nx, 2, device=device, dtype=dtype)

    def _compute_coherent_power(self, das):
        return torch.sum(das**2, dim=-1)

    def _accumulate_incoherent_power(self, chunk):
        return torch.sum(torch.sum(chunk**2, dim=-1), dim=1)

    def _compute_output(self, cf, das):
        return cf.unsqueeze(-1) * das
