"""Exact Gaussian-process surrogate (ARD Matern-5/2) in pure torch.

Hyperparameters (per-dim lengthscales, signal + noise variance) are fit by L-BFGS on the log
marginal likelihood over standardized targets. ``warp_input=True`` adds a learned per-dim
Kumaraswamy input warp (HEBO-style non-stationarity handling) fit jointly with the kernel.
Sized for BO designs (tens to hundreds of points, up to ~30 dims).
"""

import math

import numpy as np
import torch

_JITTER = 1e-4


def _safe_cholesky(k: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
    """Cholesky of ``k`` with escalating diagonal jitter until it succeeds.

    Non-finite entries (which float32 GP fits can produce) are zeroed first, and the last resort
    adds a diagonal that dominates every row, so a factorization always exists.
    """
    k = torch.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
    diag = float(torch.diagonal(k).detach().mean().clamp_min(1e-9))
    for jitter in (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0):
        try:
            return torch.linalg.cholesky(k + max(jitter * diag, jitter) * eye)
        except torch.linalg.LinAlgError:
            continue
    row_sum = float(k.abs().sum(dim=1).max().detach())
    return torch.linalg.cholesky(k + (row_sum + 10.0 * diag + 10.0) * eye)


def _matern52(d2: torch.Tensor) -> torch.Tensor:
    """Matern-5/2 correlation from squared scaled distance: ``(1+r+r^2/3) e^{-r}, r=sqrt(5 d2)``."""
    r = torch.sqrt(d2.clamp_min(1e-12) * 5.0)
    return (1.0 + r + r * r / 3.0) * torch.exp(-r)


def _kumaraswamy(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Kumaraswamy CDF ``1 - (1 - x^a)^b``: a monotone ``[0,1] -> [0,1]`` warp (identity at a=b=1)."""
    xc = x.clamp(1e-6, 1 - 1e-6)
    return 1.0 - (1.0 - xc.pow(a)).clamp_min(1e-12).pow(b)


class GP:
    """Exact ARD Matern-5/2 GP: fit on ``(x, y)``, then ``predict`` or joint-``sample``."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, warp_input: bool = False) -> None:
        self.x = x
        self.device, self.dtype = x.device, x.dtype
        self.warp_input = warp_input
        self.ymean, self.ystd = y.mean(), y.std().clamp_min(1e-6)
        self.y = (y - self.ymean) / self.ystd
        d = x.shape[1]
        kw = {"device": self.device, "dtype": self.dtype}
        self._log_ls = torch.zeros(d, requires_grad=True, **kw)
        self._log_sf = torch.zeros(1, requires_grad=True, **kw)
        self._log_sn = torch.full((1,), -2.0, requires_grad=True, **kw)
        self._params = [self._log_ls, self._log_sf, self._log_sn]
        if warp_input:
            self._warp_a = torch.zeros(d, requires_grad=True, **kw)
            self._warp_b = torch.zeros(d, requires_grad=True, **kw)
            self._params += [self._warp_a, self._warp_b]
        self._eye = torch.eye(x.shape[0], **kw)
        self._fit()

    def _warp(self, x: torch.Tensor) -> torch.Tensor:
        if not self.warp_input:
            return x
        a = torch.nn.functional.softplus(self._warp_a) + 0.5
        b = torch.nn.functional.softplus(self._warp_b) + 0.5
        return _kumaraswamy(x, a, b)

    def _kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ls = torch.nn.functional.softplus(self._log_ls) + 1e-3
        aw, bw = self._warp(a) / ls, self._warp(b) / ls
        d2 = (aw * aw).sum(1)[:, None] + (bw * bw).sum(1)[None, :] - 2.0 * aw @ bw.T
        return torch.nn.functional.softplus(self._log_sf) * _matern52(d2.clamp_min(0.0))

    def _nll(self) -> torch.Tensor:
        n = self.x.shape[0]
        k = self._kernel(self.x, self.x)
        sn = torch.nn.functional.softplus(self._log_sn) + _JITTER
        k = k + sn * self._eye
        chol = _safe_cholesky(k, self._eye)
        alpha = torch.cholesky_solve(self.y[:, None], chol)
        fit = 0.5 * (self.y[:, None] * alpha).sum()
        logdet = torch.log(torch.diagonal(chol)).sum()
        return fit + logdet + 0.5 * n * math.log(2 * math.pi)

    def _fit(self) -> None:
        opt = torch.optim.LBFGS(self._params, lr=0.1, max_iter=15)

        def closure():
            opt.zero_grad()
            loss = self._nll()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            sn = torch.nn.functional.softplus(self._log_sn) + _JITTER
            k = self._kernel(self.x, self.x) + sn * self._eye
            self._chol = _safe_cholesky(k, self._eye)
            self._alpha = torch.cholesky_solve(self.y[:, None], self._chol)

    @torch.no_grad()
    def predict(self, xq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior mean and std at ``xq``, in the original y scale. Non-finite entries (possible
        in float32) are sanitized so downstream acquisitions never see NaN."""
        kqx = torch.nan_to_num(self._kernel(xq, self.x), nan=0.0, posinf=0.0, neginf=0.0)
        mean = kqx @ self._alpha
        v = torch.linalg.solve_triangular(self._chol, kqx.T, upper=False)
        kqq = torch.nn.functional.softplus(self._log_sf)
        var = (kqq - (v * v).sum(0)[:, None]).clamp_min(1e-9)
        mean = torch.nan_to_num(mean.squeeze(1), nan=0.0, posinf=0.0, neginf=0.0)
        std = torch.nan_to_num(var.squeeze(1), nan=1e-9).clamp_min(1e-9).sqrt()
        return (mean * self.ystd + self.ymean, std * self.ystd)

    @torch.no_grad()
    def sample(self, xq: torch.Tensor, n: int) -> torch.Tensor:
        """Draw ``n`` joint posterior samples over ``xq``, shape ``(n, len(xq))``, original y scale.

        Uses the full posterior covariance (mean + Cholesky(cov) z) so draws respect correlations
        between candidates, which batch Thompson sampling requires. Draws come from torch's
        global RNG (callers seed via ``torch.manual_seed``).
        """
        kqx = self._kernel(xq, self.x)
        mean = (kqx @ self._alpha).squeeze(1)
        v = torch.linalg.solve_triangular(self._chol, kqx.T, upper=False)
        kqq = self._kernel(xq, xq)
        cov = kqq - v.T @ v
        cov = 0.5 * (cov + cov.T)
        cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        eye = torch.eye(len(xq), dtype=cov.dtype, device=cov.device)
        base = 1e-6 * float(torch.diagonal(cov).mean().clamp_min(1e-12))
        chol = None
        for scale in (1.0, 10.0, 100.0, 1000.0, 10000.0):
            try:
                chol = torch.linalg.cholesky(cov + base * scale * eye)
                break
            except torch.linalg.LinAlgError:
                continue
        if chol is None:
            chol = torch.sqrt(torch.diagonal(cov).clamp_min(1e-12)).diag()
        z = torch.randn(len(xq), n, dtype=cov.dtype, device=cov.device)
        draws = mean[:, None] + chol @ z
        return draws.T * self.ystd + self.ymean


def output_warp(y: np.ndarray) -> np.ndarray:
    """Signed-log transform ``sign(y-med) log1p(|y-med|)``: compresses skew and penalty cliffs
    while preserving the argmin (HEBO's output warp)."""
    c = np.median(y)
    d = y - c
    return np.sign(d) * np.log1p(np.abs(d))
