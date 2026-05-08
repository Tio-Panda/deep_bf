from __future__ import annotations

import torch
import torch.nn as nn

from ...apod.apod import get_window

class AngularCPWCShortPulse(nn.Module):
    def __init__(self, pw, kind="boxcar"):
        super().__init__()
        self.pw = pw
        self.kind = kind

    def forward(self, bf_data):
        na, nz, nx = bf_data.shape
        device = bf_data.device
        dtype = bf_data.dtype

        x = torch.linspace(
            self.pw.xlims[0], self.pw.xlims[1], nx, device=device, dtype=dtype
        )
        z = torch.linspace(
            self.pw.zlims[0], self.pw.zlims[1], nz, device=device, dtype=dtype
        )
        Z, X = torch.meshgrid(z, x, indexing="ij")

        theta = torch.from_numpy(self.pw.angles[:na]).to(device=device, dtype=dtype)

        aperture = torch.tensor(
            self.pw.aperture_width / 2.0, device=device, dtype=dtype
        )
        x_center = torch.tensor(
            (self.pw.xlims[0] + self.pw.xlims[1]) / 2.0, device=device, dtype=dtype
        )

        u = X.unsqueeze(0) - Z.unsqueeze(0) * torch.tan(theta).view(-1, 1, 1)
        distance = torch.abs(u - x_center)
        w = get_window(distance, aperture, kind=self.kind)

        return torch.sum(w * bf_data, dim=0)


class AngularCPWCParaxial(nn.Module):
    def __init__(self, pw, kind="boxcar"):
        super().__init__()
        self.pw = pw
        self.kind = kind

    def forward(self, bf_data):
        na, nz, nx = bf_data.shape
        device = bf_data.device
        dtype = bf_data.dtype

        x = torch.linspace(
            self.pw.xlims[0], self.pw.xlims[1], nx, device=device, dtype=dtype
        )
        z = torch.linspace(
            self.pw.zlims[0], self.pw.zlims[1], nz, device=device, dtype=dtype
        )
        Z, X = torch.meshgrid(z, x, indexing="ij")

        theta = torch.from_numpy(self.pw.angles[:na]).to(device=device, dtype=dtype)

        aperture = torch.tensor(
            self.pw.aperture_width / 2.0, device=device, dtype=dtype
        )
        x_center = torch.tensor(
            (self.pw.xlims[0] + self.pw.xlims[1]) / 2.0, device=device, dtype=dtype
        )

        u = X.unsqueeze(0) - Z.unsqueeze(0) * theta.view(-1, 1, 1)
        distance = torch.abs(u - x_center)
        w = get_window(distance, aperture, kind=self.kind)

        return torch.sum(w * bf_data, dim=0)
