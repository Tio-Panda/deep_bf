import torch
import torch.nn as nn
import torch.nn.functional as F

class MVB(nn.Module):
    def __init__(self, batch_size_angle=1, nc=128, z_chunk=512, L=16, diagonal_loading=5e-2, dtype=torch.float32, device="cuda"):
        super().__init__()
        self.batch_size = batch_size_angle
        self.L = L
        self.dl = diagonal_loading
        self.dtype = dtype
        self.device = device

        self.z_chunk = z_chunk
        M = nc - L + 1

        subarrays_idx = torch.arange(L, device=device).unsqueeze(0) + \
                        torch.arange(M, device=device).unsqueeze(1)


        self.register_buffer("d_idx", torch.arange(L, device=device))

        self.register_buffer("subarrays_idx", subarrays_idx.view(-1))
        self.register_buffer("a", torch.ones(L, dtype=dtype, device=device))
        self.register_buffer("rhs", torch.ones(L, 1, dtype=dtype, device=device))
        self.register_buffer("eye", torch.eye(L, dtype=dtype, device=device))

        self.grid_template = None

    def _get_aligned_signals(self, a_batch, norm, n_channels, n_samples, nz, nx, _rf, _t0, _d_tx, d_rx, fs, c0):
        samples = fs * (((_d_tx[:, None, :, :] + d_rx[None, ...]) / c0) + _t0.view(-1, 1, 1, 1))
        x = _rf.view(a_batch * n_channels, 1, 1, n_samples)
        samples = samples.view(a_batch * n_channels, nz, nx)
        
        if self.grid_template is None or self.grid_template.shape[:3] != (a_batch * n_channels, nz, nx):
            self.grid_template = torch.empty(a_batch * n_channels, nz, nx, 2, device=self.device, dtype=self.dtype)
        
        self.grid_template[..., 1] = 0
        self.grid_template[..., 0] = samples * norm - 1.0
        return F.grid_sample(x, self.grid_template, mode="bilinear", padding_mode="border", align_corners=True).view(a_batch, n_channels, nz, nx)

    def forward(self, rf, t0, d_tx, d_rx, fs, f0, c0, apod):
        torch.set_float32_matmul_precision('high')
        
        with torch.no_grad():
            n_angles, n_channels, n_samples = rf.shape
            _, nz, nx = d_tx.shape
            M = n_channels - self.L + 1
            y_mv = torch.zeros(n_angles, nz, nx, dtype=self.dtype, device=self.device)
            
            for _start in range(0, n_angles, self.batch_size):
                _end = min(_start + self.batch_size, n_angles)
                a_batch = _end - _start
                
                aligned_signals = self._get_aligned_signals(a_batch, 2.0/(n_samples-1), n_channels, n_samples, nz, nx, rf[_start:_end], t0[_start:_end], d_tx[_start:_end], d_rx, fs, c0)
                
                for chunk_idx, chunk in enumerate(torch.split(aligned_signals, self.z_chunk, dim=2)):
                    B, C, CZ, CX = chunk.shape
                    # x_flat: [B*P, Nc]
                    x_flat = chunk.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C) # [B*P, nc]
                    
                    x = F.avg_pool1d(x_flat.unsqueeze(1), kernel_size=M, stride=1).squeeze(1)
                    
                    subarrays = torch.index_select(x_flat, 1, self.subarrays_idx).view(-1, M, self.L) # [B*P, M, L]
                    
                    R = torch.bmm(subarrays.transpose(-2, -1).conj(), subarrays) / M # [B*P, L, L]
                    R = 0.5 * (R + R.transpose(-2, -1).conj()) # +

                    trace_R = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1)
                    delta = trace_R[..., None] / self.L # +
                    R[..., self.d_idx, self.d_idx] += self.dl * delta

                    del subarrays
                    
                    L_mat, _ = torch.linalg.cholesky_ex(R, check_errors=True)
                    rhs = self.rhs.expand(R.shape[0], -1, -1)
                    u = torch.cholesky_solve(rhs, L_mat).squeeze(-1) 
                    
                    denom = torch.sum(u * self.a, dim=-1, keepdim=True)
                    # w = u / (denom.abs() + 1e-10)
                    w = u / (denom + 1e-10)
                    
                    y_chunk = torch.sum(w.conj() * x, dim=-1)
                    
                    start_z = chunk_idx * self.z_chunk
                    y_mv[_start:_end, start_z : start_z + CZ, :] = y_chunk.view(B, CZ, CX)
            
            return y_mv
