import torch
import torch.nn as nn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...beamformers.resampler.resampler_module import ResamplerByIdsAndAngles

class DAS(nn.Module):
    def __init__(self, nz, nx, resampler: "ResamplerByIdsAndAngles"):
        super().__init__()
        self.nz = nz
        self.nx = nx
        self.resampler = resampler

    def forward(self, data, ids, angles):
        B, C2, _, _ = data.shape

        sampled_data = self.resampler(data, ids, angles) # [B, C2, nc, nz, nx]
        das = sampled_data.sum(dim=2) # [B, C2, nz, nx]

        C = C2 // 2
        das = das.reshape(B, C, 2, self.nz, self.nx) # [B, C, 2, H, W]
        envelope = torch.linalg.vector_norm(das, dim=2)  # [B, C, H, W]
        return envelope
