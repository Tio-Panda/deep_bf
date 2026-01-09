import torch
import torch.nn as nn
import torch.nn.functional as F

class GridSampleDAS(nn.Module):
    def __init__(self, nz, nx, c0, fs, decimation=1, angle_batch_size=5):
        """
        Args:
            nz: Número de puntos en profundidad
            nx: Número de puntos en azimuth
            c0: Velocidad del sonido [m/s]
            fs: Frecuencia de muestreo [Hz]
            decimation: Factor de decimación (default: 1)
            angle_batch_size: Número de ángulos a procesar simultáneamente (default: 10)
        """
        super().__init__()
        self.nz, self.nx = nz, nx
        self.c0 = c0
        self.fs = fs
        self.decimation = decimation
        self.angle_batch_size = angle_batch_size

    def forward(self, rf_tensor, d_tx, d_rx, t0):
        """
        Args:
            rf_tensor: RF data con shape [n_angles, n_elements, n_samples]
            d_tx: Transmit distances con shape [n_angles, nz, nx]
            d_rx: Receive distances con shape [n_elements, nz, nx]
            t0: Time offset con shape [n_angles]

        Returns:
            das: Beamformed data con shape [n_angles, nz, nx]
        """
        assert rf_tensor.ndim == 3, f"Expected 3D input [n_angles, n_elements, n_samples], got {rf_tensor.ndim}D"

        n_angles, n_elements, n_samp = rf_tensor.shape
        norm_factor = 2.0 / (n_samp - 1)
        das_list = []

        for i in range(0, n_angles, self.angle_batch_size):
            end_idx = min(i + self.angle_batch_size, n_angles)
            batch_size = end_idx - i

            rf_batch = rf_tensor[i:end_idx]  # [batch_size, n_elements, n_samples]
            d_tx_batch = d_tx[i:end_idx]  # [batch_size, nz, nx]
            t0_batch = t0[i:end_idx]  # [batch_size]

            total_delay = (d_tx_batch.unsqueeze(1) + d_rx.unsqueeze(0)) / self.c0 - t0_batch.view(-1, 1, 1, 1)
            delays_batch = total_delay * (self.fs / self.decimation)  # [batch_size, n_elements, nz, nx]

            x_folded = rf_batch.reshape(batch_size * n_elements, 1, 1, n_samp)
            delays_folded = delays_batch.reshape(batch_size * n_elements, self.nz, self.nx)

            grid = torch.zeros(batch_size * n_elements, self.nz, self.nx, 2,
                             dtype=torch.float16, device=rf_tensor.device)
            grid[..., 0] = delays_folded * norm_factor - 1.0

            sampled = F.grid_sample(
                x_folded,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )

            sampled = sampled.view(batch_size, n_elements, 1, self.nz, self.nx)
            das_batch = torch.sum(sampled, dim=1).squeeze(1)

            das_list.append(das_batch)

        das = torch.cat(das_list, dim=0)

        return das
