import torch
import torch.nn as nn
import torch.nn.functional as F

class DAS(nn.Module):
    def __init__(self, gsi, chunk_size=16, device="cuda", dtype=torch.float32):
        super().__init__()
        self.gsi = gsi
        self.device = device
        self.dtype = dtype
        self.chunk_size = chunk_size

    def forward(self, sample):
        rfs, ids, _, _ = sample
        rfs = rfs.to(device=self.device)

        unique_ids, inverse_idxs = torch.unique(ids, return_inverse=True) 
        U = unique_ids.size(0)

        samples_idx = self.gsi.samples_idx[unique_ids]
        samples_idx = samples_idx.to(device=self.device, non_blocking=True)

        B, K, nc, ns = rfs.shape
        _, _, nz, nx = samples_idx.shape
        
        norm_factor = 2.0 / (ns - 1)
        output = torch.zeros(B, K, nz, nx, device=self.device, dtype=self.dtype)
        
        for u in range(U):
            mask = (inverse_idxs == u)
            n_rf = mask.sum().item()

            filtered_rfs = rfs[mask]
            selected_samples_idx = samples_idx[u]

            for c_start in range(0, nc, self.chunk_size):
                c_end = min(c_start + self.chunk_size, nc)
                CHK = c_end - c_start

                _rf = filtered_rfs[:, :, c_start:c_end].reshape(-1, 1, 1, ns)

                _samples_idx = selected_samples_idx[c_start:c_end]
                _samples_idx = _samples_idx.unsqueeze(0).expand(n_rf * K, CHK, nz, nx).reshape(-1, nz, nx)


                grid = torch.zeros(_samples_idx.shape + (2,), device=self.device, dtype=self.dtype)
                grid[..., 0] = _samples_idx * norm_factor - 1.0

                sampled = F.grid_sample(
                    _rf,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True
                )

                sampled = sampled.view(n_rf, K, CHK, nz, nx)
                output[mask] += sampled.sum(dim=2)
        
        return output.permute(0, 2, 3 ,1)
