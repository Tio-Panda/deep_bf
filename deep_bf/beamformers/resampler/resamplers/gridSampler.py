from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

class GridSampleBase(nn.Module, ABC):
    def __init__(self, mode="bilinear", padding_mode="zeros", align_corners=False):
        super().__init__()

        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    @abstractmethod
    def _reshape_data(self, data, b, nc, ns) -> torch.Tensor:
        pass

    @abstractmethod
    def _reshape_output(self, data, b, nc, nz, nx) -> torch.Tensor:
        pass

    def forward(self, data, samples_idx):
        B, nc, nz, nx = samples_idx.shape
        ns = data.shape[2]

        x = self._reshape_data(data, B, nc, ns)
        samples_idx = samples_idx.reshape(B * nc, nz, nx)

        norm_factor = 2.0 / (ns - 1) if self.align_corners else 2.0 / (ns)
        grid = torch.empty(B * nc, nz, nx, 2, device=x.device, dtype=x.dtype)

        if self.align_corners:
            grid[..., 0] = samples_idx * norm_factor - 1.0
        else:
            grid[..., 0] = (samples_idx + 0.5) * norm_factor - 1.0

        grid[..., 1] = 0.0

        sampled_data = F.grid_sample(
            x,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

        return self._reshape_output(sampled_data, B, nc, nz, nx)


class GridSample3D(GridSampleBase):
    def __init__(self, mode="bilinear", padding_mode="zeros", align_corners=False):
        super().__init__(mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    def _reshape_data(self, data, b, nc, ns) -> torch.Tensor:
        return data.reshape(b * nc, 1, 1, ns) # RF-IQComplex: [B*nc, 1, 1, ns]

    def _reshape_output(self, output, b, nc, nz, nx) -> torch.Tensor:
        return output.reshape(b, nc, nz, nx)

class GridSample4D(GridSampleBase):
    def __init__(self, mode="bilinear", padding_mode="zeros", align_corners=False):
        super().__init__(mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    def _reshape_data(self, data, b, nc, ns) -> torch.Tensor:
        return data.permute(0, 1, 3, 2).reshape(b * nc, 2, 1, ns) # IQ: [B*nc, 2, 1, ns]

    def _reshape_output(self, output, b, nc, nz, nx) :
        return output.reshape(b, nc, 2, nz, nx).permute(0, 1, 3, 4, 2)
