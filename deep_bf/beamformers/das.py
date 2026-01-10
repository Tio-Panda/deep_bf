import torch
import torch.nn as nn
import torch.nn.functional as F

class RFGridSampleDAS(nn.Module):
    def __init__(self, nz, nx, pw, angles_idx, batch_size=5, device="cuda", dtype=torch.float32):
        """
        Args:
            nz: Número de puntos en profundidad
            nx: Número de puntos en azimuth
            pw: RFData object
            angles_idx: Índices de los ángulos a procesar
            batch_size: Número de ángulos a procesar simultáneamente (default: 10)
        """
        super().__init__()
        self.nz, self.nx = nz, nx
        self.c0 = pw.c0
        self.fs = pw.fs
        self.t0 = torch.from_numpy(pw.t0).to(device=device, dtype=dtype)
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        self.rf = torch.from_numpy(pw.data[angles_idx]).to(device=device, dtype=dtype) # [n_angles, n_elements, n_samples]

    def forward(self, d_tx, d_rx, apod):
        """
        Args:
            d_tx: Transmit distances con shape [n_angles, nz, nx]
            d_rx: Receive distances con shape [n_elements, nz, nx]
            apod: Apodization con shape [n_elements, nz, nx]

        Returns:
            das: Beamformed data con shape [n_angles, nz, nx]
        """

        n_angles, n_elements, n_samp = self.rf.shape
        norm_factor = 2.0 / (n_samp - 1)

        das = torch.zeros(n_angles, self.nz, self.nx, dtype=self.dtype, device=self.device)

        for i in range(0, n_elements, self.batch_size):
            end_idx = min(i + self.batch_size, n_elements)
            batch_size_elem = end_idx - i

            rf_batch = self.rf[:, i:end_idx, :]  # [n_angles, batch_size_elem, n_samples]
            d_rx_batch = d_rx[i:end_idx, :, :]  # [batch_size_elem, nz, nx]
            apod_batch = apod[i:end_idx, :, :]  # [batch_size_elem, nz, nx]

            total_delay = (d_tx.unsqueeze(1) + d_rx_batch.unsqueeze(0)) / self.c0 - self.t0.view(-1, 1, 1, 1)
            delays_batch = total_delay * self.fs # [n_angles, batch_size_elem, nz, nx]

            x_folded = rf_batch.reshape(n_angles * batch_size_elem, 1, 1, n_samp)
            delays_folded = delays_batch.reshape(n_angles * batch_size_elem, self.nz, self.nx)

            grid = torch.zeros(n_angles * batch_size_elem, self.nz, self.nx, 2,
                             dtype=self.dtype, device=self.device)
            grid[..., 0] = delays_folded * norm_factor - 1.0

            sampled = F.grid_sample(
                x_folded,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )

            sampled = sampled.view(n_angles, batch_size_elem, self.nz, self.nx)
            sampled = sampled * apod_batch.unsqueeze(0)  # [n_angles, batch_size_elem, nz, nx] * [1, batch_size_elem, nz, nx]

            das += torch.sum(sampled, dim=1)  # [n_angles, nz, nx]

        return das

class RFInterDAS(nn.Module):
    def __init__(self, nz, nx, pw, angles_idx, batch_size=1, device="cuda", dtype=torch.float32):
        super().__init__()
        self.nz, self.nx = nz, nx
        self.c0 = pw.c0
        self.fs = pw.fs
        self.t0 = torch.from_numpy(pw.t0).to(device=device, dtype=dtype)
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        self.rf = torch.from_numpy(pw.data[angles_idx]).to(device=device, dtype=dtype) # [n_angles, n_elements, n_samples]

    def forward(self, d_tx, d_rx, apod):
        """
        Beamforming DAS usando interpolación lineal manual (sustituye a grid_sample).
        d_tx: [n_angles, nz, nx]
        d_rx: [n_elements, nz, nx]
        apod: [n_elements, nz, nx]
        """
        n_angles, n_elements, n_samples = self.rf.shape
        
        das_all = torch.zeros(n_angles, self.nz, self.nx, dtype=self.dtype, device=self.device)

        for i in range(0, n_angles, self.batch_size):
            end_idx = min(i + self.batch_size, n_angles)
            batch_ang = end_idx - i

            rf_b = self.rf[i:end_idx] # [batch, n_elements, n_samples]
            d_tx_b = d_tx[i:end_idx] # [batch, nz, nx]
            t0_b = self.t0[i:end_idx] # [batch]

            tau = t0_b.view(-1, 1, 1, 1) + (d_tx_b.unsqueeze(1) + d_rx.unsqueeze(0)) / self.c0
            s_idx = tau * self.fs

            s_idx = torch.clamp(s_idx, 0.0, float(n_samples - 1.001))

            idx_low = s_idx.long()
            idx_high = idx_low + 1
            idx_frac = s_idx - idx_low.float()

            # flat_idx: [batch, n_elements, nz*nx]
            flat_idx_low = idx_low.view(batch_ang, n_elements, -1)
            flat_idx_high = idx_high.view(batch_ang, n_elements, -1)

            s_low = torch.gather(rf_b, 2, flat_idx_low).view(batch_ang, n_elements, self.nz, self.nx)
            s_high = torch.gather(rf_b, 2, flat_idx_high).view(batch_ang, n_elements, self.nz, self.nx)

            sampled = s_low + idx_frac * (s_high - s_low)

            weighted = sampled * apod.unsqueeze(0) # [batch, n_elements, nz, nx]
            
            das_all[i:end_idx] = torch.sum(weighted, dim=1)

        return das_all