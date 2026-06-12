import torch
import torch.nn as nn

from ....beamformers.resampler.resampler_builder import resampler_builder
from ....config_registery import ResamplerConfig
from ....constants.bf import ResamplerType, PWDataType

MODEL_RESAMPLER = resampler_builder(
    ResamplerConfig(-1, ResamplerType.GRID_SAMPLE, {}),
    PWDataType.RF
)

class ToFCModels(nn.Module):
    def __init__(self,
        batch_size=1, filter_chunk_size=2, resampler=MODEL_RESAMPLER):
        super().__init__()
        self.batch_size = batch_size
        self.filter_chunk_size = filter_chunk_size
        self.resampler = resampler

    def forward(self, batch_data, batch_samples_idx):
        b, c, nc, ns = batch_data.shape
        _, _, nz, nx = batch_samples_idx.shape

        flat_data = batch_data.reshape(b * c, nc, ns)
        flat_idx = (
            batch_samples_idx.to(device=batch_data.device, dtype=batch_data.dtype)
            .unsqueeze(1)
            .expand(b, c, nc, nz, nx)
            .reshape(b * c, nc, nz, nx)
        )
        flat_tofc = torch.empty(
            b * c, nc, nz, nx, device=batch_data.device, dtype=batch_data.dtype
        )
       
        # TODO: Un chunk por nc tambien podria ser una buena idea si no son tantas llamadas.
        chunk_size = self.batch_size * self.filter_chunk_size
        for s in range(0, b * c, chunk_size):
            e = min(s + chunk_size, b * c)
            flat_tofc[s:e] = self.resampler(flat_data[s:e], flat_idx[s:e])

        return flat_tofc.reshape(b, c, nc, nz, nx)
