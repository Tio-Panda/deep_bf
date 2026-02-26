import torch
import torch.nn as nn
import torch.nn.functional as F

class DAS(nn.Module):
    def __init__(self, gsi, chunk_size=64, device="cuda", dtype=torch.float32):
        super().__init__()
        # self.gsi = gsi
        self.device = device
        self.dtype = dtype
        self.chunk_size = chunk_size

        # self.samples_idx = gsi.samples_idx.to(device=device)
        self.register_buffer("samples_idx", gsi.samples_idx.to(device=device, non_blocking=True))

    def forward(self, rfs, ids):
        unique_ids, inverse_idxs = torch.unique(ids, return_inverse=True) 
        U = unique_ids.size(0)

        # samples_idx = self.gsi.samples_idx[unique_ids]
        # samples_idx = samples_idx.to(device=self.device, non_blocking=True)

        B, K, nc, ns = rfs.shape
        _, _, nz, nx = self.samples_idx.shape
        
        norm_factor = 2.0 / (ns - 1)
        output = torch.zeros(B, K, nz, nx, device=self.device, dtype=self.dtype)

        perm = torch.argsort(inverse_idxs)
        inv_sorted = inverse_idxs.index_select(0, perm)
        rfs_sorted = rfs.index_select(0, perm)

        if B > 1:
            change = inv_sorted[1:] != inv_sorted[:-1]
            starts = torch.cat([inv_sorted.new_zeros(1), change.nonzero(as_tuple=False).squeeze(1) + 1])
        else:
            starts = inv_sorted.new_zeros(1)

        ends = torch.cat([starts[1:], inv_sorted.new_full((1,), B)])
        keys = inv_sorted.index_select(0, starts)

        starts = starts.to("cpu").tolist()
        ends = ends.to("cpu").tolist()
        keys = keys.to("cpu").tolist()
        
        # for u in range(U):
        for g, (s, e) in enumerate(zip(starts, ends)):
            G = e - s
            if G == 0:
                continue

            gid = unique_ids[keys[g]].to(device=self.device)
            selected_samples_idx = self.samples_idx.index_select(0, gid.view(1)).squeeze(0) 
            block = rfs_sorted[s:e]

            for c_start in range(0, nc, self.chunk_size):
                c_end = min(c_start + self.chunk_size, nc)
                CHK = c_end - c_start

                rf = block[:, :, c_start:c_end].reshape(-1, 1, 1, ns)

                sidx = selected_samples_idx[c_start:c_end]
                sidx = sidx.unsqueeze(0).expand(G * K, CHK, nz, nx).reshape(-1, nz, nx)

                grid_x = sidx * norm_factor - 1.0
                grid = torch.stack([grid_x, torch.zeros_like(grid_x)], dim=-1)

                sampled = F.grid_sample(
                    rf,
                    grid,
                    mode="bilinear",
                    # padding_mode="zeros",
                    padding_mode="border",
                    align_corners=True
                )

                sampled = sampled.view(G, K, CHK, nz, nx).sum(dim=2)
                output.index_add_(0, perm[s:e], sampled)
                # output[mask] += sampled.sum(dim=2)
        
        return output
