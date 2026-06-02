from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class IMAPBase(nn.Module, ABC):
    def __init__(self, batch_size: int = 32, num_iters: int = 2, eps: float = 1e-8):
        super().__init__()
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.eps = eps

    @abstractmethod
    def _apod_tofc_data(self, tofc_data: torch.Tensor, apod: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _signal_power(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _apply_weight(self, w: torch.Tensor, x_das: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, tofc_data: torch.Tensor, apod: torch.Tensor) -> torch.Tensor:
        b, nc, nz, nx = tofc_data.shape[:4]
        tofc_data = self._apod_tofc_data(tofc_data, apod)

        x_das = torch.mean(tofc_data, dim=1)
        # x_das = torch.sum(tofc_data, dim=1)
        x_hat = x_das

        for _ in range(self.num_iters):
            sigma_x2 = self._signal_power(x_hat)
            noise_acc = torch.zeros(b, nz, nx, device=tofc_data.device, dtype=sigma_x2.dtype)
            x_expanded = x_hat.unsqueeze(1)

            for s in range(0, nc, self.batch_size):
                e = min(s + self.batch_size, nc)
                chunk = tofc_data[:, s:e, ...]
                diff = chunk - x_expanded
                noise_acc += torch.sum(self._signal_power(diff), dim=1)

            sigma_n2 = noise_acc / nc
            w = (nc * sigma_x2) / (sigma_n2 + nc * sigma_x2 + self.eps)
            x_hat = self._apply_weight(w, x_das)

        return x_hat


class IMAP3D(IMAPBase):
    def _apod_tofc_data(self, tofc_data: torch.Tensor, apod: torch.Tensor) -> torch.Tensor:
        return tofc_data * apod.unsqueeze(0)

    def _signal_power(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x) ** 2

    def _apply_weight(self, w: torch.Tensor, x_das: torch.Tensor) -> torch.Tensor:
        return w * x_das


class IMAP4D(IMAPBase):
    def _apod_tofc_data(self, tofc_data: torch.Tensor, apod: torch.Tensor) -> torch.Tensor:
        return tofc_data * apod.unsqueeze(0).unsqueeze(-1)

    def _signal_power(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x**2, dim=-1)

    def _apply_weight(self, w: torch.Tensor, x_das: torch.Tensor) -> torch.Tensor:
        return w.unsqueeze(-1) * x_das
