from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class MVBase(nn.Module, ABC):
    def __init__(
        self,
        batch_size: int = 1,
        z_chunk: int = 512,
        L: int = 16,
        diagonal_loading: float | None = None,
        temporal_radius: int = 0,
        eps: float = 1e-10,
    ):
        super().__init__()
        if L <= 0:
            raise ValueError(f"Expected L > 0, got L={L}")
        if temporal_radius < 0:
            raise ValueError(
                f"Expected temporal_radius >= 0, got temporal_radius={temporal_radius}"
            )

        self.batch_size = batch_size
        self.z_chunk = z_chunk
        self.L = L
        self.dl = 1.0 / (100.0 * L) if diagonal_loading is None else diagonal_loading
        self.temporal_radius = temporal_radius
        self.eps = eps

        self._cache_nc: int | None = None
        self._cache_device: torch.device | None = None
        self._cache_dtype: torch.dtype | None = None
        self._cache_M: int | None = None
        self.register_buffer(
            "subarrays_idx", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "d_idx", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer("a", torch.empty(0), persistent=False)
        self.register_buffer("rhs", torch.empty(0, 1), persistent=False)

    @abstractmethod
    def _to_solver_domain(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _from_solver_domain(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def _build_runtime_buffers(self, nc: int, device: torch.device, dtype: torch.dtype):
        if nc < self.L:
            raise ValueError(f"Expected nc >= L, got nc={nc}, L={self.L}")

        if self.L > (nc + 1) // 2:
            raise ValueError(
                f"Expected L <= (nc + 1) / 2 for spatial smoothing, got nc={nc}, L={self.L}"
            )

        if (
            self._cache_nc == nc
            and self._cache_device == device
            and self._cache_dtype == dtype
        ):
            return

        M = nc - self.L + 1
        subarrays_idx = torch.arange(self.L, device=device).unsqueeze(0) + torch.arange(
            M, device=device
        ).unsqueeze(1)

        self.subarrays_idx = subarrays_idx.reshape(-1)
        self.d_idx = torch.arange(self.L, device=device)
        self.a = torch.ones(self.L, device=device, dtype=dtype)
        self.rhs = torch.ones(self.L, 1, device=device, dtype=dtype)

        self._cache_nc = nc
        self._cache_device = device
        self._cache_dtype = dtype
        self._cache_M = M

    def _estimate_covariance(
        self, subarrays: torch.Tensor, b: int, nz: int, nx: int, M: int
    ) -> torch.Tensor:
        if self.temporal_radius == 0:
            return torch.bmm(subarrays.transpose(-2, -1), subarrays.conj()) / M

        subarrays_grid = subarrays.view(b, nz, nx, M, self.L)
        R_spatial = torch.einsum(
            "bznmi,bznmj->bznij", subarrays_grid, subarrays_grid.conj()
        ) / M

        R_accum = torch.zeros_like(R_spatial)
        counts = torch.zeros(nz, device=R_spatial.device, dtype=R_spatial.real.dtype)

        for offset in range(-self.temporal_radius, self.temporal_radius + 1):
            target_start = max(0, -offset)
            target_end = min(nz, nz - offset)
            if target_start >= target_end:
                continue

            source_start = target_start + offset
            source_end = target_end + offset
            R_accum[:, target_start:target_end] += R_spatial[
                :, source_start:source_end
            ]
            counts[target_start:target_end] += 1

        R = R_accum / counts.view(1, nz, 1, 1, 1)
        return R.reshape(-1, self.L, self.L)

    def forward(self, tofc_data):
        if tofc_data.dim() not in (4, 5):
            raise ValueError(
                f"Expected tofc_data to have 4 or 5 dims, got {tofc_data.dim()}"
            )

        torch.set_float32_matmul_precision("high")

        b, nc, nz, nx = tofc_data.shape[:4]
        solver_data = self._to_solver_domain(tofc_data)

        self._build_runtime_buffers(nc, solver_data.device, solver_data.dtype)
        M = self._cache_M
        if M is None:
            raise RuntimeError("Internal error: M cache was not initialized")

        output = torch.zeros(
            b,
            nz,
            nx,
            device=solver_data.device,
            dtype=solver_data.dtype,
        )

        for s in range(0, b, self.batch_size):
            e = min(s + self.batch_size, b)
            _sampled_data = solver_data[s:e]

            for i, chunk in enumerate(torch.split(_sampled_data, self.z_chunk, dim=2)):
                b_chunk, _, nz_chunk, _ = chunk.shape
                x = chunk.reshape(b_chunk, nc, -1).permute(0, 2, 1).reshape(-1, nc)

                subarrays = torch.index_select(x, 1, self.subarrays_idx).view(
                    -1, M, self.L
                )

                R = self._estimate_covariance(subarrays, b_chunk, nz_chunk, nx, M)
                R = 0.5 * (R + R.transpose(-2, -1).conj())

                trace_R = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1).real
                R[..., self.d_idx, self.d_idx] += self.dl * trace_R[..., None] + self.eps

                rhs = self.rhs.expand(R.shape[0], -1, -1)
                u = torch.linalg.solve(R, rhs).squeeze(-1)

                denom = torch.sum(self.a.conj() * u, dim=-1, keepdim=True)
                w = u / denom

                output_chunk = torch.sum(
                    w.conj().unsqueeze(1) * subarrays, dim=-1, keepdim=True
                ).mean(dim=1)

                s_chunk = i * self.z_chunk
                output[s:e, s_chunk : s_chunk + nz_chunk, :] = output_chunk.view(
                    b_chunk, nz_chunk, nx
                )

        return self._from_solver_domain(output)


class MV3D(MVBase):
    def _to_solver_domain(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _from_solver_domain(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MV4D(MVBase):
    def _to_solver_domain(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 2:
            raise ValueError(
                f"Expected IQ split channels with last dim=2, got {x.shape}"
            )
        return torch.view_as_complex(x.contiguous())

    def _from_solver_domain(self, x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_real(x)
