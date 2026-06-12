from __future__ import annotations

import torch
import torch.nn as nn

from ....constants.bf import CompooundingType

class CPWC(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.type = type

    def forward(self, bf_data):
        if self.type == CompooundingType.CPWC_SUM:
            return torch.sum(bf_data, dim=0)
        else:
            return torch.mean(bf_data, dim=0)


