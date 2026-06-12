import torch
import torch.nn as nn
import torch.nn.functional as F

class GridSample(nn.Module):
    def __init__(self, mode="bilinear", padding_mode="zeros", align_corners=False):
        super().__init__()

        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, rfs, samples_idx, b, nc, ns, nz, nx):
        is_iq = rfs.dim() == 4

        if is_iq:
            x = rfs.permute(0, 1, 3, 2).reshape(b * nc, 2, 1, ns) # IQ: [B*nc, 2, 1, ns]
        else:
            x = rfs.reshape(b * nc, 1, 1, ns) # RF: [B*nc, 1, 1, ns]

        if samples_idx.dim() == 3:
            samples_idx = samples_idx.reshape(1, nc, nz, nx).expand(b, nc, nz, nx)
        samples_idx = samples_idx.reshape(b * nc, nz, nx)

        norm_factor = 2.0 / (ns - 1) if self.align_corners else 2.0 / (ns)
        grid = torch.empty(b * nc, nz, nx, 2, device=x.device, dtype=x.dtype)

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

        if is_iq:
            return sampled_data.reshape(b, nc, 2, nz, nx).permute(0, 1, 3, 4, 2)

        return sampled_data.reshape(b, nc, nz, nx)

class LinearInterpolation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, rfs, samples_idx, b, nc, ns, nz, nx):
        is_iq = rfs.dim() == 4

        if is_iq:
            x = rfs.permute(0, 1, 3, 2).reshape(b * nc, 2, ns)  # IQ: [B*nc, 2, ns]
            c = 2
        else:
            x = rfs.reshape(b * nc, 1, ns)  # RF: [B*nc, 1, ns]
            c = 1

        if samples_idx.dim() == 3:
            samples_idx = samples_idx.reshape(1, nc, nz, nx).expand(b, nc, nz, nx)
        samples_idx = samples_idx.reshape(b * nc, nz, nx)

        idx0 = torch.floor(samples_idx).to(torch.long)  # [B*nc, nz, nx]
        idx0 = torch.minimum(idx0, torch.full_like(idx0, ns - 2))

        idx1 = idx0 + 1
        frac = (samples_idx - idx0.to(samples_idx.dtype)).unsqueeze(1)  # [B*nc, 1, nz, nx]

        idx0f = idx0.reshape(b * nc, 1, -1).expand(-1, c, -1)  # [B*nc, C, nz*nx]
        idx1f = idx1.reshape(b * nc, 1, -1).expand(-1, c, -1)
        s0 = torch.gather(x, 2, idx0f).reshape(b * nc, c, nz, nx)
        s1 = torch.gather(x, 2, idx1f).reshape(b * nc, c, nz, nx)
        sampled = s0 + frac * (s1 - s0)  # [B*nc, C, nz, nx]

        if is_iq:
            return sampled.reshape(b, nc, 2, nz, nx).permute(0, 1, 3, 4, 2)  # [B, nc, nz, nx, 2]
        return sampled.reshape(b, nc, nz, nx)  # [B, nc, nz, nx]

class IntegerDelayFractionalFIR(nn.Module):
    def __init__(
        self,
        n_taps=9,
        n_phases=64,
        window="hamming",
        z_chunk=128,
        eps=1e-12,
    ):
        super().__init__()
        if n_taps < 3 or (n_taps % 2 == 0):
            raise ValueError("n_taps must be odd and >= 3")
        if n_phases < 2:
            raise ValueError("n_phases must be >= 2")

        self.n_taps = int(n_taps)
        self.n_phases = int(n_phases)
        self.z_chunk = int(z_chunk)
        self.eps = float(eps)

        offsets = torch.arange(self.n_taps, dtype=torch.long) - (self.n_taps // 2)
        coeffs = self._build_polyphase_table(self.n_taps, self.n_phases, window, self.eps)
        self.register_buffer("offsets", offsets, persistent=False)  # [T]
        self.register_buffer("coeffs", coeffs, persistent=False)    # [P, T]

    def _build_window(self, n_taps, window):
        if window == "hamming":
            return torch.hamming_window(n_taps, periodic=False, dtype=torch.float32)
        if window == "hann":
            return torch.hann_window(n_taps, periodic=False, dtype=torch.float32)
        if window == "boxcar":
            return torch.ones(n_taps, dtype=torch.float32)
        raise ValueError(f"Unsupported window: {window}")

    def _build_polyphase_table(self, n_taps, n_phases, window, eps):
        # h_mu[k] = sinc(k - mu) * w[k], mu in [0,1)
        k = torch.arange(n_taps, dtype=torch.float32) - (n_taps // 2) # [T]
        mu = torch.arange(n_phases, dtype=torch.float32) / n_phases # [P]
        x = k.unsqueeze(0) - mu.unsqueeze(1) # [P, T]
        h = torch.sinc(x)
        w = self._build_window(n_taps, window).unsqueeze(0) # [1, T]
        h = h * w
        h = h / (h.sum(dim=1, keepdim=True) + eps) # ganancia DC ~1
        return h

    def forward(self, rfs, samples_idx, b, nc, ns, nz, nx):
        is_iq = (rfs.dim() == 4)
        if is_iq:
            x = rfs.permute(0, 1, 3, 2).reshape(b * nc, 2, ns) # [B*nc, 2, ns]
            c = 2
        else:
            x = rfs.reshape(b * nc, 1, ns) # [B*nc, 1, ns]
            c = 1

        if ns < self.n_taps:
            raise ValueError(f"ns ({ns}) must be >= n_taps ({self.n_taps})")

        if samples_idx.dim() == 3:
            samples_idx = samples_idx.reshape(1, nc, nz, nx).expand(b, nc, nz, nx)
        samples_idx = samples_idx.reshape(b * nc, nz, nx)

        half = self.n_taps // 2
        s_min = float(half)
        s_max = float((ns - 1) - half) - 1e-6
        s = samples_idx.clamp(min=s_min, max=s_max)

        idx_int = torch.floor(s).to(torch.long)                               # [B*nc, nz, nx]
        frac = s - idx_int.to(s.dtype)                                        # [B*nc, nz, nx]

        phase = torch.round(frac * (self.n_phases - 1)).to(torch.long)        # [B*nc, nz, nx]
        phase = phase.clamp(0, self.n_phases - 1)
        offsets = self.offsets.to(device=x.device)
        coeffs = self.coeffs.to(device=x.device, dtype=x.dtype)
        out = torch.empty(b * nc, c, nz, nx, device=x.device, dtype=x.dtype)

        for z0 in range(0, nz, self.z_chunk):
            z1 = min(z0 + self.z_chunk, nz)
            zc = z1 - z0
            idx_int_chunk = idx_int[:, z0:z1, :]           # [B*nc, zc, nx]
            phase_chunk = phase[:, z0:z1, :]               # [B*nc, zc, nx]
            # [B*nc, zc, nx, T]
            idx_taps = idx_int_chunk.unsqueeze(-1) + offsets.view(1, 1, 1, -1)
            # gather sobre eje temporal ns
            gather_idx = idx_taps.reshape(b * nc, 1, -1).expand(-1, c, -1)    # [B*nc, C, zc*nx*T]
            taps = torch.gather(x, dim=2, index=gather_idx)
            taps = taps.reshape(b * nc, c, zc, nx, self.n_taps)                # [B*nc, C, zc, nx, T]
            # coeficientes por fase
            h = coeffs[phase_chunk]                                             # [B*nc, zc, nx, T]
            y = torch.sum(taps * h.unsqueeze(1), dim=-1)                        # [B*nc, C, zc, nx]
            out[:, :, z0:z1, :] = y
        # Salida final unificada
        if is_iq:
            return out.reshape(b, nc, 2, nz, nx).permute(0, 1, 3, 4, 2)        # [B, nc, nz, nx, 2]
        return out.reshape(b, nc, nz, nx)                                       # [B, nc, nz, nx]
