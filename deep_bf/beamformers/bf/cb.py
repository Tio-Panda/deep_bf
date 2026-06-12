from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

from ..utils.delays import compute_d_rx, compute_meshgrid


class CB3D(nn.Module):
    """Compressing Beamforming for RAW RF using one effective plane wave.

    This implementation models the low-memory approximation agreed for the
    current pipeline: the input RF is already angularly compounded by the
    caller and is reconstructed with the center-angle geometry.
    """

    def __init__(
        self,
        pw,
        nz: int,
        nx: int,
        batch_size: int = 32,
        num_iters: int = 50,
        measurement_ratio: float = 0.2,
        compression: str = "cmix",
        distribution: str = "rademacher",
        dt: int = 10,
        gamma: float = 1e-4,
        epsilon: float | None = None,
        epsilon_scale: float = 1e-3,
        sigma1: float = 1.0,
        sigma2: float | None = None,
        tau: float = 0.5,
        power_iters: int = 3,
        time_batch_size: int = 256,
        J: int = 2,
        mode: str = "symmetric",
        center_angle_idx: int | None = None,
        angle_tol: float = 1e-4,
        seed: int | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        if measurement_ratio <= 0.0 or measurement_ratio > 1.0:
            raise ValueError(
                f"measurement_ratio must be in (0, 1], got {measurement_ratio}"
            )
        if dt < 1:
            raise ValueError(f"dt must be >= 1, got {dt}")

        self.pw = pw
        self.nz = nz
        self.nx = nx
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.measurement_ratio = measurement_ratio
        self.compression = compression.lower()
        self.distribution = distribution.lower()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_scale = epsilon_scale
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.tau = tau
        self.power_iters = power_iters
        self.time_batch_size = time_batch_size
        self.J = J
        self.mode = mode
        self.center_angle_idx = center_angle_idx
        self.angle_tol = angle_tol
        self.seed = seed
        self.eps = eps

        self.wavelets = [f"db{i}" for i in range(1, 9)]
        self.q = float(len(self.wavelets))
        self.sa_scale = 1.0 / math.sqrt(self.q)
        self.dwt_fwds = nn.ModuleList(
            [DWTForward(J=self.J, wave=w, mode=self.mode) for w in self.wavelets]
        )
        self.dwt_invs = nn.ModuleList(
            [DWTInverse(wave=w, mode=self.mode) for w in self.wavelets]
        )

        self.register_buffer("Z", torch.empty(0), persistent=False)
        self.register_buffer("X", torch.empty(0), persistent=False)
        self.register_buffer("d_rx", torch.empty(0), persistent=False)
        self.register_buffer("selected_idx", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("mixing", torch.empty(0), persistent=False)
        self.register_buffer("time_offsets", torch.empty(0, dtype=torch.long), persistent=False)

        self._geom_device: torch.device | None = None
        self._geom_dtype: torch.dtype | None = None
        self._runtime_nz: int | None = None
        self._runtime_nx: int | None = None
        self._comp_cache: tuple[int, int, torch.device, torch.dtype] | None = None

    def _get_pw_angle_count(self) -> int:
        if hasattr(self.pw, "na"):
            return int(self.pw.na)
        return int(self.pw.n_angles)

    def _get_center_angle_idx(self) -> int:
        if self.center_angle_idx is not None:
            return int(self.center_angle_idx)
        return self._get_pw_angle_count() // 2

    def _get_center_t0(self) -> float:
        return float(self.pw.t0[self._get_center_angle_idx()])

    def _validate_center_angle(self):
        theta = float(self.pw.angles[self._get_center_angle_idx()])
        if abs(theta) > self.angle_tol:
            raise NotImplementedError(
                "CB3D currently implements the H_RF operator only for theta=0. "
                f"Got center angle theta={theta}."
            )

    def _build_geometry(self, device: torch.device, dtype: torch.dtype):
        if self._geom_device == device and self._geom_dtype == dtype:
            return

        self._validate_center_angle()

        nyquist = self.nz == -1
        Z, X = compute_meshgrid(
            self.pw,
            self.nz,
            self.nx,
            nyquist=nyquist,
            device=device,
            dtype=dtype,
        )
        self.Z = Z
        self.X = X
        self.d_rx = compute_d_rx(self.pw, Z, X, device=device, dtype=dtype)
        self._runtime_nz = int(Z.shape[0])
        self._runtime_nx = int(Z.shape[1])
        self._geom_device = device
        self._geom_dtype = dtype

    def _validate_raw_rf(self, raw_rf: torch.Tensor):
        if raw_rf.dim() != 3:
            raise ValueError(f"Expected raw_rf with shape [1, nc, ns], got {raw_rf.shape}")
        if raw_rf.shape[0] != 1:
            raise ValueError(
                "CB3D expects one effective RAW RF acquisition. "
                f"Got {raw_rf.shape[0]} angles/acquisitions."
            )
        if raw_rf.shape[1] != int(self.pw.nc):
            raise ValueError(f"Expected nc={self.pw.nc}, got {raw_rf.shape[1]}")
        if torch.is_complex(raw_rf):
            raise ValueError("CB3D is implemented only for real RF data")

    def _generator(self, device: torch.device) -> torch.Generator | None:
        if self.seed is None:
            return None
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed)
        return gen

    def _build_compression(
        self, n_elements: int, n_samples: int, device: torch.device, dtype: torch.dtype
    ):
        cache = (n_elements, n_samples, device, dtype)
        if self._comp_cache == cache:
            return

        n_measurements = max(1, int(math.ceil(n_elements * self.measurement_ratio)))
        gen = self._generator(device)

        if self.compression in ("none", "identity"):
            self.selected_idx = torch.arange(n_elements, device=device)
            self.mixing = torch.eye(n_elements, device=device, dtype=dtype)
            self.time_offsets = torch.zeros(1, device=device, dtype=torch.long)
        elif self.compression == "uniform":
            self.selected_idx = torch.linspace(
                0,
                n_elements - 1,
                n_measurements,
                device=device,
                dtype=dtype,
            ).round().long().unique()
            self.mixing = torch.empty(0, device=device, dtype=dtype)
            self.time_offsets = torch.zeros(1, device=device, dtype=torch.long)
        elif self.compression == "random":
            self.selected_idx = torch.randperm(
                n_elements, device=device, generator=gen
            )[:n_measurements].sort().values
            self.mixing = torch.empty(0, device=device, dtype=dtype)
            self.time_offsets = torch.zeros(1, device=device, dtype=torch.long)
        elif self.compression == "cmix":
            self.selected_idx = torch.empty(0, device=device, dtype=torch.long)
            self.mixing = self._make_mixing(
                (n_measurements, n_elements), device, dtype, gen
            )
            self.time_offsets = torch.zeros(1, device=device, dtype=torch.long)
        elif self.compression == "ctmix":
            self.selected_idx = torch.empty(0, device=device, dtype=torch.long)
            self.mixing = self._make_mixing(
                (n_measurements, n_elements, self.dt), device, dtype, gen
            )
            self.time_offsets = torch.arange(self.dt, device=device) - (self.dt // 2)
        else:
            raise ValueError(
                "compression must be one of none, uniform, random, cmix, ctmix; "
                f"got {self.compression}"
            )

        self._comp_cache = cache

    def _make_mixing(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        gen: torch.Generator | None,
    ) -> torch.Tensor:
        if self.distribution == "rademacher":
            values = torch.randint(0, 2, shape, device=device, generator=gen, dtype=torch.long)
            return values.to(dtype=dtype).mul_(2.0).sub_(1.0)
        if self.distribution in ("gaussian", "normal"):
            return torch.randn(shape, device=device, dtype=dtype, generator=gen)
        raise ValueError(
            f"distribution must be rademacher or gaussian, got {self.distribution}"
        )

    def _D(self, m: torch.Tensor) -> torch.Tensor:
        self._build_compression(m.shape[1], m.shape[2], m.device, m.dtype)
        if self.compression in ("none", "identity", "uniform", "random"):
            return torch.index_select(m, 1, self.selected_idx)
        if self.compression == "cmix":
            return torch.einsum("me,aet->amt", self.mixing, m)

        out = torch.zeros(
            m.shape[0], self.mixing.shape[0], m.shape[2], device=m.device, dtype=m.dtype
        )
        for j, offset in enumerate(self.time_offsets.tolist()):
            shifted = self._shift_read(m, int(offset))
            out += torch.einsum("me,aet->amt", self.mixing[:, :, j], shifted)
        return out

    def _Dt(self, md: torch.Tensor) -> torch.Tensor:
        n_elements = int(self.pw.nc)
        self._build_compression(n_elements, md.shape[2], md.device, md.dtype)
        if self.compression in ("none", "identity", "uniform", "random"):
            out = torch.zeros(md.shape[0], n_elements, md.shape[2], device=md.device, dtype=md.dtype)
            out.index_copy_(1, self.selected_idx, md)
            return out
        if self.compression == "cmix":
            return torch.einsum("me,amt->aet", self.mixing, md)

        out = torch.zeros(md.shape[0], n_elements, md.shape[2], device=md.device, dtype=md.dtype)
        for j, offset in enumerate(self.time_offsets.tolist()):
            back = torch.einsum("me,amt->aet", self.mixing[:, :, j], md)
            out += self._shift_adjoint(back, int(offset))
        return out

    def _shift_read(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        out = torch.zeros_like(x)
        if offset >= 0:
            if offset < x.shape[-1]:
                out[..., : x.shape[-1] - offset] = x[..., offset:]
        else:
            offset_abs = -offset
            if offset_abs < x.shape[-1]:
                out[..., offset_abs:] = x[..., : x.shape[-1] - offset_abs]
        return out

    def _shift_adjoint(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        out = torch.zeros_like(x)
        if offset >= 0:
            if offset < x.shape[-1]:
                out[..., offset:] = x[..., : x.shape[-1] - offset]
        else:
            offset_abs = -offset
            if offset_abs < x.shape[-1]:
                out[..., : x.shape[-1] - offset_abs] = x[..., offset_abs:]
        return out

    def _element_positions(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(self.pw.probe_geometry[:, 0]).to(device=device, dtype=dtype)

    def _time_grid(self, n_samples: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sample_idx = torch.arange(n_samples, device=device, dtype=dtype)
        return sample_idx / float(self.pw.fs) - self._get_center_t0()

    def _z_axis(self) -> torch.Tensor:
        return self.Z[:, 0]

    def _x_axis(self) -> torch.Tensor:
        return self.X[0, :]

    def _curve_terms(
        self,
        element_pos: torch.Tensor,
        time_values: torch.Tensor,
    ):
        x_axis = self._x_axis()
        z_axis = self._z_axis()
        if z_axis.numel() < 2:
            raise ValueError("CB3D requires at least two z samples")

        z0 = z_axis[0]
        dz = z_axis[1] - z_axis[0]
        c0 = torch.as_tensor(float(self.pw.c0), device=x_axis.device, dtype=x_axis.dtype)

        xt = element_pos.view(-1, 1, 1)
        t = time_values.view(1, -1, 1)
        xn = x_axis.view(1, 1, -1)

        s = c0 * t
        s_safe = s.clamp_min(self.eps)
        d = xn - xt
        z = (s_safe.square() - d.square()) / (2.0 * s_safe)

        q_float = (z - z0) / dz
        q0 = torch.floor(q_float).long()
        alpha = q_float - q0.to(q_float.dtype)

        valid = (
            (s > self.eps)
            & torch.isfinite(z)
            & (q0 >= 0)
            & (q0 + 1 < z_axis.numel())
        )
        q0 = q0.clamp(0, z_axis.numel() - 1)
        q1 = (q0 + 1).clamp(0, z_axis.numel() - 1)

        r = torch.sqrt(d.square() + z.square()).clamp_min(self.eps)
        od = 1.0 / (2.0 * math.pi * r)
        jac = torch.sqrt(1.0 + (d / s_safe).square())
        grad = torch.sqrt((d / (c0 * r)).square() + (1.0 / c0 + z / (c0 * r)).square())
        weight = od * jac * grad * valid.to(x_axis.dtype)

        return q0, q1, alpha, weight

    def _H(
        self,
        x: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected x with shape [nz, nx], got {x.shape}")

        self._build_geometry(x.device, x.dtype)
        n_elements = int(self.pw.nc)
        nx = x.shape[1]
        col_idx = torch.arange(nx, device=x.device).view(1, 1, nx)
        element_pos_all = self._element_positions(x.device, x.dtype)
        time_all = self._time_grid(n_samples, x.device, x.dtype)
        out = torch.zeros(1, n_elements, n_samples, device=x.device, dtype=x.dtype)

        for s in range(0, n_elements, self.batch_size):
            e = min(s + self.batch_size, n_elements)
            elem_pos = element_pos_all[s:e]
            cols = col_idx.expand(e - s, -1, -1)
            for ts in range(0, n_samples, self.time_batch_size):
                te = min(ts + self.time_batch_size, n_samples)
                q0, q1, alpha, weight = self._curve_terms(elem_pos, time_all[ts:te])
                cols_chunk = cols.expand(-1, te - ts, -1)
                g0 = x[q0, cols_chunk]
                g1 = x[q1, cols_chunk]
                interp = (1.0 - alpha) * g0 + alpha * g1
                out[0, s:e, ts:te] = torch.sum(weight * interp, dim=-1)

        return out

    def _Ht(
        self,
        r: torch.Tensor,
    ) -> torch.Tensor:
        if r.dim() != 3:
            raise ValueError(f"Expected residual with shape [1, nc, ns], got {r.shape}")

        self._build_geometry(r.device, r.dtype)
        nz = int(self.Z.shape[0])
        nx = int(self.X.shape[1])
        n_elements = r.shape[1]
        n_samples = r.shape[-1]
        out = torch.zeros(nz * nx, device=r.device, dtype=r.dtype)
        col_idx = torch.arange(nx, device=r.device).view(1, 1, nx)
        element_pos_all = self._element_positions(r.device, r.dtype)
        time_all = self._time_grid(n_samples, r.device, r.dtype)

        for s in range(0, n_elements, self.batch_size):
            e = min(s + self.batch_size, n_elements)
            elem_pos = element_pos_all[s:e]
            cols = col_idx.expand(e - s, -1, -1)
            for ts in range(0, n_samples, self.time_batch_size):
                te = min(ts + self.time_batch_size, n_samples)
                q0, q1, alpha, weight = self._curve_terms(elem_pos, time_all[ts:te])
                cols_chunk = cols.expand(-1, te - ts, -1)
                residual = r[0, s:e, ts:te].unsqueeze(-1)

                idx0 = q0 * nx + cols_chunk
                idx1 = q1 * nx + cols_chunk
                base = residual * weight
                out.scatter_add_(0, idx0.reshape(-1), (base * (1.0 - alpha)).reshape(-1))
                out.scatter_add_(0, idx1.reshape(-1), (base * alpha).reshape(-1))

        return out.view(nz, nx)

    def _Hd(
        self,
        x: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        return self._D(self._H(x, n_samples))

    def _Hdt(
        self,
        rd: torch.Tensor,
    ) -> torch.Tensor:
        return self._Ht(self._Dt(rd))

    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0).unsqueeze(0)

    def _from_nchw(self, x: torch.Tensor) -> torch.Tensor:
        return x[0, 0]

    def _match_nchw_shape(self, x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        x = x[..., : shape[-2], : shape[-1]]
        pad_h = max(0, shape[-2] - x.shape[-2])
        pad_w = max(0, shape[-1] - x.shape[-1])
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def _psi_t(self, x: torch.Tensor):
        x_nchw = self._to_nchw(x)
        coeffs = []
        for dwt in self.dwt_fwds:
            yl, yh = dwt(x_nchw)
            coeffs.append((yl * self.sa_scale, [band * self.sa_scale for band in yh]))
        return coeffs

    def _psi(self, coeffs, target_shape: torch.Size) -> torch.Tensor:
        x = None
        for (yl, yh), idwt in zip(coeffs, self.dwt_invs):
            x_i = idwt((yl, yh)) * self.sa_scale
            x_i = self._match_nchw_shape(x_i, target_shape)
            x = x_i if x is None else x + x_i
        if x is None:
            raise RuntimeError("No wavelet coefficients available")
        return self._from_nchw(x)

    def _zeros_like_coeffs(self, coeffs):
        return [
            (torch.zeros_like(yl), [torch.zeros_like(band) for band in yh])
            for yl, yh in coeffs
        ]

    def _coeffs_add(self, a, b):
        out = []
        for (yl_a, yh_a), (yl_b, yh_b) in zip(a, b):
            out.append((yl_a + yl_b, [ba + bb for ba, bb in zip(yh_a, yh_b)]))
        return out

    def _coeffs_sub(self, a, b):
        out = []
        for (yl_a, yh_a), (yl_b, yh_b) in zip(a, b):
            out.append((yl_a - yl_b, [ba - bb for ba, bb in zip(yh_a, yh_b)]))
        return out

    def _coeffs_linear(self, a, b, alpha: float, beta: float):
        out = []
        for (yl_a, yh_a), (yl_b, yh_b) in zip(a, b):
            out.append(
                (
                    alpha * yl_a + beta * yl_b,
                    [alpha * ba + beta * bb for ba, bb in zip(yh_a, yh_b)],
                )
            )
        return out

    def _soft_coeffs(self, coeffs, gamma: float):
        out = []
        for yl, yh in coeffs:
            yl_s = yl
            yh_s = [self._soft_threshold(band, gamma) for band in yh]
            out.append((yl_s, yh_s))
        return out

    def _soft_threshold(self, x: torch.Tensor, gamma: float) -> torch.Tensor:
        return torch.sign(x) * torch.relu(torch.abs(x) - gamma)

    def _proj_l2_ball(self, x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(x)
        scale = torch.minimum(radius / (norm + self.eps), torch.ones_like(norm))
        return x * scale

    def _estimate_operator_norm(
        self,
        n_samples: int,
    ) -> torch.Tensor:
        nz = int(self.Z.shape[0])
        nx = int(self.X.shape[1])
        x = torch.randn(nz, nx, device=self.Z.device, dtype=self.Z.dtype)
        x = x / (torch.linalg.vector_norm(x) + self.eps)

        for _ in range(max(1, self.power_iters)):
            y = self._Hd(x, n_samples)
            x = self._Hdt(y)
            x = x / (torch.linalg.vector_norm(x) + self.eps)

        y = self._Hd(x, n_samples)
        return torch.linalg.vector_norm(y).clamp_min(self.eps)

    def forward(
        self,
        raw_rf: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_raw_rf(raw_rf)
        self._build_geometry(raw_rf.device, raw_rf.dtype)

        y = self._D(raw_rf)
        epsilon = (
            torch.as_tensor(self.epsilon, device=y.device, dtype=y.dtype)
            if self.epsilon is not None
            else self.epsilon_scale * torch.linalg.vector_norm(y)
        )

        n_samples = raw_rf.shape[-1]
        target_shape = torch.Size((1, 1, self.Z.shape[0], self.X.shape[1]))
        sigma2 = self.sigma2
        if sigma2 is None:
            if self.power_iters > 0:
                L = self._estimate_operator_norm(n_samples)
                sigma2_t = 1.0 / (L * L + self.eps)
                sigma2 = float(sigma2_t.detach().cpu())
            else:
                sigma2 = 1.0

        x = torch.zeros(self.Z.shape[0], self.X.shape[1], device=y.device, dtype=y.dtype)
        r1 = self._psi_t(x)
        v1 = self._zeros_like_coeffs(r1)
        r2 = self._Hd(x, n_samples)
        v2 = torch.zeros_like(y)

        for _ in range(self.num_iters):
            sparse_grad = self._psi(v1, target_shape)
            data_grad = self._Hdt(v2)
            x = x - self.tau * (self.sigma1 * sparse_grad + sigma2 * data_grad)

            psi_t_x = self._psi_t(x)
            two_psi_t_x = self._coeffs_linear(psi_t_x, psi_t_x, 2.0, 0.0)
            r1 = self._coeffs_add(v1, self._coeffs_sub(two_psi_t_x, r1))
            v1 = self._coeffs_sub(r1, self._soft_coeffs(r1, self.gamma))

            hx = self._Hd(x, n_samples)
            r2 = v2 + (2.0 * hx - r2)
            residual = r2 - y
            v2 = residual - self._proj_l2_ball(residual, epsilon)

        return x
