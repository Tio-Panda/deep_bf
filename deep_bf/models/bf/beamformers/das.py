import torch
import torch.nn as nn

class DAS(nn.Module):
    def __init__(self, batch_size=1, filter_chunk_size=None):
        super().__init__()
        if batch_size <= 0:
            raise ValueError(f"batch_size debe ser > 0, recibido {batch_size}")
        if filter_chunk_size is not None and filter_chunk_size <= 0:
            raise ValueError(
                f"filter_chunk_size debe ser None o > 0, recibido {filter_chunk_size}"
            )
        self.batch_size = batch_size
        self.filter_chunk_size = filter_chunk_size

    def forward(self, tofc_data):
        if tofc_data.dim() != 5:
            raise ValueError(
                f"tofc_data debe tener shape [B, C, nc, nz, nx], recibido {tuple(tofc_data.shape)}"
            )

        b, c, _, nz, nx = tofc_data.shape
        output = torch.zeros(b, c, nz, nx, device=tofc_data.device, dtype=tofc_data.dtype)
        filter_step = self.filter_chunk_size or c

        for b_start in range(0, b, self.batch_size):
            b_end = min(b_start + self.batch_size, b)
            for c_start in range(0, c, filter_step):
                c_end = min(c_start + filter_step, c)
                chunk = tofc_data[b_start:b_end, c_start:c_end]
                output[b_start:b_end, c_start:c_end] = torch.sum(chunk, dim=2)
        
        return output
