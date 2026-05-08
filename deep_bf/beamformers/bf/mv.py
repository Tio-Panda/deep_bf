from __future__ import annotations

import torch
import torch.nn as nn


class MVB(nn.Module):
    def __init__(
        self,
        nz,
        nx,
        sampler: nn.Module,
        batch_size=1,
        nc=128,
        z_chunk=512,
        L=16,
        diagonal_loading=1e-3,
        eps=1e-10,
    ):
        super().__init__()
        self.nz = nz
        self.nx = nx
        self.sampler = sampler

        self.batch_size = batch_size
        self.L = L
        self.dl = diagonal_loading
        self.z_chunk = z_chunk
        self.M = nc - L + 1
        self.eps = eps

        subarrays_idx = torch.arange(L).unsqueeze(0) + torch.arange(self.M).unsqueeze(1)
        self.register_buffer("subarrays_idx", subarrays_idx.view(-1))
        self.register_buffer("d_idx", torch.arange(L))
        self.register_buffer("a", torch.ones(L))
        self.register_buffer("rhs", torch.ones(L, 1))
        self.register_buffer("eye", torch.eye(L))

    def forward(self, rfs, ids, apod=None):
        torch.set_float32_matmul_precision("high")

        B, nc, _ = rfs.shape
        output = torch.zeros(B, self.nz, self.nx, device=rfs.device, dtype=rfs.dtype)

        sampled_data = self.sampler(rfs, ids)  # [B, nc, nz, nx]

        for s in range(0, B, self.batch_size):
            e = min(s + self.batch_size, B)

            _sampled_data = sampled_data[s:e]

            for i, chunk in enumerate(torch.split(_sampled_data, self.z_chunk, dim=2)):
                b, _, nz_chunk, nx = chunk.shape
                x = chunk.reshape(b, nc, -1).permute(0, 2, 1).reshape(-1, nc)

                subarrays = torch.index_select(x, 1, self.subarrays_idx).view(
                    -1, self.M, self.L
                )

                R = torch.bmm(subarrays.transpose(-2, -1).conj(), subarrays) / self.M
                R = 0.5 * (R + R.transpose(-2, -1).conj())

                trace_R = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1)
                delta = trace_R[..., None] / self.L
                R[..., self.d_idx, self.d_idx] += self.dl * delta + self.eps

                L_mat, _ = torch.linalg.cholesky_ex(R, check_errors=True)
                rhs = self.rhs.expand(R.shape[0], -1, -1)
                u = torch.cholesky_solve(rhs, L_mat).squeeze(-1)

                denom = torch.sum(u * self.a, dim=-1, keepdim=True)
                w = u / (denom + self.eps)

                output_chunk = torch.sum(
                    w.conj().unsqueeze(1) * subarrays, dim=-1, keepdim=True
                ).mean(dim=1)

                s_chunk = i * self.z_chunk
                output[s:e, s_chunk : s_chunk + nz_chunk, :] = output_chunk.view(
                    b, nz_chunk, nx
                )

        return output


class MVBSimple(nn.Module):
    def __init__(
        self,
        nz,
        nx,
        batch_size=1,
        nc=128,
        z_chunk=512,
        L=16,
        diagonal_loading=1e-3,
        eps=1e-10,
    ):
        super().__init__()
        self.nz = nz
        self.nx = nx

        self.batch_size = batch_size
        self.L = L
        self.dl = diagonal_loading
        self.z_chunk = z_chunk
        self.M = nc - L + 1
        self.eps = eps

        subarrays_idx = torch.arange(L).unsqueeze(0) + torch.arange(self.M).unsqueeze(1)
        self.register_buffer("subarrays_idx", subarrays_idx.view(-1))
        self.register_buffer("d_idx", torch.arange(L))
        self.register_buffer("a", torch.ones(L))
        self.register_buffer("rhs", torch.ones(L, 1))
        self.register_buffer("eye", torch.eye(L))

    def forward(self, sampled_data, apod=None):
        torch.set_float32_matmul_precision("high")

        B, nc, _, nx = sampled_data.shape
        output = torch.zeros(
            B, self.nz, self.nx, device=sampled_data.device, dtype=sampled_data.dtype
        )

        for s in range(0, B, self.batch_size):
            e = min(s + self.batch_size, B)

            _sampled_data = sampled_data[s:e]

            for i, chunk in enumerate(torch.split(_sampled_data, self.z_chunk, dim=2)):
                b, _, nz_chunk, _ = chunk.shape
                x = chunk.reshape(b, nc, -1).permute(0, 2, 1).reshape(-1, nc)

                subarrays = torch.index_select(x, 1, self.subarrays_idx).view(
                    -1, self.M, self.L
                )

                R = torch.bmm(subarrays.transpose(-2, -1).conj(), subarrays) / self.M
                R = 0.5 * (R + R.transpose(-2, -1).conj())

                trace_R = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1)
                delta = trace_R[..., None] / self.L
                R[..., self.d_idx, self.d_idx] += self.dl * delta + self.eps

                L_mat, _ = torch.linalg.cholesky_ex(R, check_errors=True)
                rhs = self.rhs.expand(R.shape[0], -1, -1)
                u = torch.cholesky_solve(rhs, L_mat).squeeze(-1)

                denom = torch.sum(u * self.a, dim=-1, keepdim=True)
                w = u / (denom + self.eps)

                output_chunk = torch.sum(
                    w.conj().unsqueeze(1) * subarrays, dim=-1, keepdim=True
                ).mean(dim=1)

                s_chunk = i * self.z_chunk
                output[s:e, s_chunk : s_chunk + nz_chunk, :] = output_chunk.view(
                    b, nz_chunk, nx
                )

        return output
