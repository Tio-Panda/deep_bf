import torch
import torch.nn as nn

from typing import List

from ..utils.delays import compute_meshgrid, compute_d_rx, compute_samples_idx_by_angles
from ..resampler.resampler_builder import resampler_builder
from ...config_registery import ResamplerConfig
# from ...constants.bf import ResamplerType, PWDataType
from ...wrapper.timer import Timer      

class ToFCClassic(nn.Module):
    def __init__(self, pw, resampler_configs:List[ResamplerConfig], nz, nx, nyquist=False, batch_size=5):
        super().__init__()
        self.pw = pw
        self.nz = nz
        self.nx = nx
        self.nc = pw.nc
        self.ns = pw.ns
        self.B = batch_size

        self.resamplers = {}
        for rC in resampler_configs:
            self.resamplers[rC.type] = resampler_builder(rC, pw.type)

        self.choosed_resampler = self.resamplers[resampler_configs[0].type]

        self.timer = Timer()

        with self.timer.measure("pre-align"):
            self.Z, self.X = compute_meshgrid(pw, nz, nx, nyquist=nyquist)
            self.d_rx = compute_d_rx(pw, self.Z, self.X)

        if nyquist:
            self.nz, self.nx = self.Z.shape

        self.samples_idx = torch.ones(
            batch_size, pw.nc, nz, nx, device="cuda", dtype=torch.float32
        )

    def set_resampler(self, resampler_type):
        self.choosed_resampler = self.resamplers[resampler_type]

    def get_ZX(self):
        return self.Z, self.X

    def forward(self, angles_idxs):
        with self.timer.measure("tofc"):
            data = self.pw.data[angles_idxs].to("cuda")
            self.samples_idx = compute_samples_idx_by_angles(self.pw, self.Z, self.X, self.d_rx, angles_idxs, device=data.device, dtype=data.dtype) # [B, nc, nz, nx]
            B = self.samples_idx.shape[0]

            tofc_data = self.choosed_resampler(data, self.samples_idx)

        tofc_time = self.timer.times["tofc"] / B
        times = [self.timer.times["pre-align"] + tofc_time] * B

        return tofc_data, times

class ToFC(nn.Module):
    def __init__(self, resampler, batch_size=1):
        super().__init__()
        self.resampler = resampler
        self.batch_size = batch_size

    def forward(self, batch_data, batch_samples_idx):
        b, nc, _ = batch_data.shape
        _, _, nz, nx = batch_samples_idx.shape

        batch_tofc_data = torch.empty(
            b, nc, nz, nx, device=batch_data.device, dtype=batch_data.dtype
        )

        for s in range(0, b, self.batch_size):
            e = min(s + self.batch_size, b)
            batch_tofc_data[s:e] = self.resampler(batch_data[s:e], batch_samples_idx[s:e])

        return batch_tofc_data


