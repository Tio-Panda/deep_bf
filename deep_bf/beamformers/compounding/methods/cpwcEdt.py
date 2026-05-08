from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt

class CPWCEDT(nn.Module):
    def __init__(self, alpha=1e-3, eps=1e-10):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, bf_data):
        k = bf_data.shape[0]
        device = bf_data.device
        dtype = bf_data.dtype

        abs_data = torch.abs(bf_data)
        min_vals = abs_data.amin(dim=(1, 2), keepdim=True)
        max_vals = abs_data.amax(dim=(1, 2), keepdim=True)
        norm_data = (abs_data - min_vals) / (max_vals - min_vals + self.eps)

        binary = norm_data >= self.alpha
        binary_np = binary.detach().cpu().numpy()

        masks = np.empty(binary_np.shape, dtype=np.float32)
        for i in range(k):
            b = binary_np[i]
            d = distance_transform_edt(b)
            d_min = float(d.min())
            d_max = float(d.max())
            masks[i] = (d - d_min) / (d_max - d_min + self.eps)

        a = torch.from_numpy(masks).to(device=device, dtype=dtype)
        return torch.sum(a * bf_data, dim=0) / k
