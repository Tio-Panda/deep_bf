from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

# TODO: DAS se calcula distinto para IQ que RF con el jblabla asi que son 3 clases o ver como lo hacemos

class DASBase(nn.Module, ABC):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    @abstractmethod
    def _malloc_output(self, b, nz, nx, device, dtype) -> torch.Tensor:
        pass

    @abstractmethod
    def _apod_tofc_data(self, tofc_data, apod) -> torch.Tensor:
        pass

    def forward(self, tofc_data, apod):
        b, nc, nz, nx = tofc_data.shape[:4]
        tofc_data = self._apod_tofc_data(tofc_data, apod)
        output = self._malloc_output(b, nz, nx, tofc_data.device, tofc_data.dtype)

        for s in range(0, nc, self.batch_size):
            e = min(s + self.batch_size, nc)
            chunk = tofc_data[:, s:e, ...]
            output += torch.sum(chunk, dim=1) # RF/IQComplex: [B, nz, nx] | IQ: [B, nz, nx, 2]
        
        return output

class DAS3D(DASBase):
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def _malloc_output(self, b, nz, nx, device, dtype):
        return torch.zeros(b, nz, nx, device=device, dtype=dtype)

    def _apod_tofc_data(self, tofc_data, apod):
        return tofc_data * apod.unsqueeze(0)  # RF/IQComplex: [B, nc, nz, nx]

class DAS4D(DASBase):
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def _malloc_output(self, b, nz, nx, device, dtype):
        return torch.zeros(b, nz, nx, 2, device=device, dtype=dtype)

    def _apod_tofc_data(self, tofc_data, apod):
        return tofc_data * apod.unsqueeze(0).unsqueeze(-1)  # IQ: [B, nc, nz, nx, 2]
