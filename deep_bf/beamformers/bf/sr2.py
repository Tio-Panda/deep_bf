from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from pytorch_wavelets import DWTForward, DWTInverse


class SR2Base(nn.Module, ABC):
    def __init__(
        self,
        batch_size: int = 32,
        num_iters: int = 30,
        lam: float = 1e-4,
        tau: float = 0.2,
        sigma: float = 0.2,
        theta: float = 1.0,
        eps: float = 1e-8,
        J: int = 2,
        mode: str = "symmetric",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.lam = lam
        self.tau = tau
        self.sigma = sigma
        self.theta = theta
        self.eps = eps
        self.J = J
        self.mode = mode

        self.wavelets = [f"db{i}" for i in range(1, 9)]
        self.q = float(len(self.wavelets))
        self.sa_scale = 1.0 / (self.q**0.5)

        self.dwt_fwds = nn.ModuleList(
            [DWTForward(J=self.J, wave=w, mode=self.mode) for w in self.wavelets]
        )
        self.dwt_invs = nn.ModuleList(
            [DWTInverse(wave=w, mode=self.mode) for w in self.wavelets]
        )

    @abstractmethod
    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _from_nchw(self, x_nchw: torch.Tensor) -> torch.Tensor:
        pass

    def _A(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _At(self, r: torch.Tensor) -> torch.Tensor:
        return r

    def _psi_t(self, x_nchw: torch.Tensor):
        coeffs = []
        for dwt in self.dwt_fwds:
            yl, yh = dwt(x_nchw)
            yl = yl * self.sa_scale
            yh = [band * self.sa_scale for band in yh]
            coeffs.append((yl, yh))
        return coeffs

    def _psi(self, coeffs) -> torch.Tensor:
        x = None
        for (yl, yh), idwt in zip(coeffs, self.dwt_invs):
            x_i = idwt((yl, yh)) * self.sa_scale
            if x is None:
                x = x_i
            else:
                x = x + x_i
        return x

    def _zeros_like_coeffs(self, coeffs):
        out = []
        for yl, yh in coeffs:
            out.append((torch.zeros_like(yl), [torch.zeros_like(b) for b in yh]))
        return out

    def _coeffs_add_scaled(self, a, b, alpha: float):
        out = []
        for (yl_a, yh_a), (yl_b, yh_b) in zip(a, b):
            yl = yl_a + alpha * yl_b
            yh = [ba + alpha * bb for ba, bb in zip(yh_a, yh_b)]
            out.append((yl, yh))
        return out

    @abstractmethod
    def _proj_linf_ball(self, coeffs):
        pass

    def _proj_linf_ball_scalar(self, coeffs):
        out = []
        for yl, yh in coeffs:
            yl_p = yl
            yh_p = [torch.clamp(b, min=-self.lam, max=self.lam) for b in yh]
            out.append((yl_p, yh_p))
        return out

    def forward(self, tofc_data: torch.Tensor) -> torch.Tensor:
        m = tofc_data

        y = torch.zeros_like(m[:, 0, ...])
        _, nc = m.shape[:2]
        for s in range(0, nc, self.batch_size):
            e = min(s + self.batch_size, nc)
            y += torch.sum(m[:, s:e, ...], dim=1)
        y = y / nc

        x = y
        x_bar = x

        x_nchw = self._to_nchw(x)
        u = self._zeros_like_coeffs(self._psi_t(x_nchw))

        for _ in range(self.num_iters):
            x_bar_nchw = self._to_nchw(x_bar)
            psi_t_xbar = self._psi_t(x_bar_nchw)
            u_tilde = self._coeffs_add_scaled(u, psi_t_xbar, self.sigma)
            u = self._proj_linf_ball(u_tilde)

            grad_data = self._At(self._A(x) - y)
            grad_sparse = self._from_nchw(self._psi(u))

            x_new = x - self.tau * (grad_data + grad_sparse)
            x_bar = x_new + self.theta * (x_new - x)
            x = x_new

        return x


class SR2_3D(SR2Base):
    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1)

    def _from_nchw(self, x_nchw: torch.Tensor) -> torch.Tensor:
        return x_nchw[:, 0, ...]

    def _proj_linf_ball(self, coeffs):
        return self._proj_linf_ball_scalar(coeffs)


class SR2_4D(SR2Base):
    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2)

    def _from_nchw(self, x_nchw: torch.Tensor) -> torch.Tensor:
        return x_nchw.permute(0, 2, 3, 1)

    def _proj_linf_ball(self, coeffs):
        out = []
        for yl, yh in coeffs:
            yl_p = yl
            yh_p = []
            for b in yh:
                mag = torch.sqrt(torch.sum(b**2, dim=1, keepdim=True) + self.eps)
                scale = torch.clamp(self.lam / mag, max=1.0)
                yh_p.append(b * scale)
            out.append((yl_p, yh_p))
        return out
