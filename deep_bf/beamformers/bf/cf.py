from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
import torch
import torch.nn as nn

class CFBase(nn.Module, ABC):
    def __init__(self, batch_size=128, eps=1e-8):
        super().__init__()
        self.batch_size = batch_size
        self.eps = eps

    @abstractmethod
    def _compute_coherent_and_incoherent_power(self, das, tofc_data) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    @abstractmethod
    def _compute_output(self, cf, das) -> torch.Tensor:
        pass

    def forward(self, tofc_data):
        nc = tofc_data.shape[1]
        das = torch.sum(tofc_data, dim=1) # RF/IQComplex: [B, nz, nx] | IQ: [B, nz, nx, 2]

        coherent_power, incoherent_power = self._compute_coherent_and_incoherent_power(das, tofc_data)
        cf = coherent_power / (nc * incoherent_power + self.eps)

        output = self._compute_output(cf, das)
        
        return output


class CF3D(CFBase):
    def __init__(self, batch_size, eps):
        super().__init__(batch_size, eps)
    def _compute_coherent_and_incoherent_power(self, das, tofc_data):
        coherent_power = torch.abs(das) ** 2
        incoherent_power = torch.sum(tofc_data ** 2, dim=1)
        return coherent_power, incoherent_power
    def _compute_output(self, cf, das):
        return cf * das # [B, nz, nx] * [B, nz, nx]

class CF4D(CFBase):
    def __init__(self, batch_size, eps):
        super().__init__(batch_size, eps)
    def _compute_coherent_and_incoherent_power(self, das, tofc_data):
        coherent_power = torch.sum(torch.abs(das) ** 2, dim=-1)
        incoherent_power = torch.sum(torch.sum(tofc_data ** 2, dim=-1), dim=1)
        return coherent_power, incoherent_power
    def _compute_output(self, cf, das):
        return cf.unsqueeze(-1) * das # [B, nz, nx, 1] * [B, nz, nx, 2]
