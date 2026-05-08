import torch
import numpy as np
from ...beamformers.utils.delays import compute_samples_idx_by_angles


class GlobalSamplesIdxFastBench:
    def __init__(self, pw, Z, X, d_rx, batch_size=5):
        self.pw = pw
        self.Z = Z
        self.X = X
        self.d_rx = d_rx
        self.batch_size = batch_size

        nc = pw.nc
        nz, nx = self.Z.shape
        self.samples_idx = torch.ones(
            batch_size, nc, nz, nx, device="cuda", dtype=torch.float32
        )

    def __getitem__(self, key):
        return np.arange(self.batch_size)

    def set_samples_idx_by_pw(self, angles, verbose=False):
        if verbose:
            print(f"Loading {len(angles)} samples_idx for {self.pw.name}")
        samples_idx = compute_samples_idx_by_angles(
            self.pw, self.Z, self.X, self.d_rx, angles
        )  # [B, nc, nz, nx]
        self.samples_idx = samples_idx
        if verbose:
            print("Samples_idx loaded")
