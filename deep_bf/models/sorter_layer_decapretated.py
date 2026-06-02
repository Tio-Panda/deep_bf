import torch.nn as nn

class ClassicSorter(nn.Module):
    def forward(self, data, x2=None, x3=None):
        # IQ: [B, C, nc, ns, 2] -> [B, 2C, nc, ns]
        if data.dim() == 5 and data.shape[-1] == 2:
            B, C, nc, ns, _ = data.shape
            return data.permute(0, 1, 4, 2, 3).reshape(B, C * 2, nc, ns)

        # RF [B, C, nc, ns]
        if data.dim() == 4:
            return data
        raise ValueError(f"Input shape no soportado: {tuple(data.shape)}")
