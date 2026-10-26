"""The Gaussian ``gam(z ~ te(i, j, k))`` used by RLT survival bands.

This deliberately implements only the complete, square index-grid problem:
natural cubic cardinal margins, integrated-curvature tensor penalties and
Gaussian GCV selection. It does not depend on an R installation.

The two-parameter Newton/descent search and its stopping/boundary rules are
adapted from Simon N. Wood's mgcv 1.9-4 ``magic.c`` (2003-2013), GPL-2.0-or-later.
The spline construction follows mgcv's ``cr`` and tensor constructors.
This package is distributed under GPL-3.0-or-later; see LICENSE.
"""
from __future__ import annotations

from functools import lru_cache
import warnings

import numpy as np
from scipy import linalg
from scipy.interpolate import CubicSpline


_EPS = np.finfo(float).eps
_RANK_TOL = np.sqrt(_EPS)


@lru_cache(maxsize=8)
def _tensor_setup(n, k):
    """R's cardinal cr basis, penalty scaling and absorbed sum constraint."""
    knots = np.linspace(1.0, float(n), k)
    spline = CubicSpline(knots, np.eye(k), bc_type="natural", axis=0)
    B = spline(np.arange(1.0, n + 1.0))
    h = np.diff(knots)
    D, E = 6 * spline.c[0], 2 * spline.c[1]
    P = (D.T @ (D * (h ** 3 / 3)[:, None])
         + D.T @ (E * (h ** 2 / 2)[:, None])
         + E.T @ (D * (h ** 2 / 2)[:, None])
         + E.T @ (E * h[:, None]))
    P = (P + P.T) / 2
    P /= linalg.eigvalsh(P)[-1]
    penalties = [np.kron(P, np.eye(k)), np.kron(np.eye(k), P)]
    # smoothCon scales penalties before absorbing the centering constraint.
    max_row_sum = np.max(np.sum(np.abs(B), axis=1)) ** 2
    scales = [np.linalg.norm(S, 1) / max_row_sum ** 2 for S in penalties]
    penalties = [S / scale for S, scale in zip(penalties, scales)]

    # R's qr(t(C)) is a single Householder reflector. The first basis
    # direction is removed; a separate unpenalized intercept is added.
    c = np.outer(B.sum(axis=0), B.sum(axis=0)).ravel()
    v = c.copy()
    v[0] += linalg.norm(c)
    H = np.eye(k * k) - 2 * np.outer(v, v) / (v @ v)
    transform = np.column_stack((np.ones(k * k), H[:, 1:]))
    gram = np.kron(B.T @ B, B.T @ B)
    gram = transform.T @ gram @ transform
    gram = (gram + gram.T) / 2
    R = linalg.cholesky(gram, lower=False)
    centered, roots = [], []
    penalty_rank = k * (k - 2)
    for S in penalties:
        S = transform.T @ S @ transform
        S[0, :] = 0
        S[:, 0] = 0
        S = (S + S.T) / 2
        vals, vecs = linalg.eigh(S)
        root = vecs[:, -penalty_rank:] * np.sqrt(np.maximum(vals[-penalty_rank:], 0))
        centered.append(S)
        roots.append(root)
    initial = _initial_sp(gram, centered)
    return B, transform, R, tuple(roots), initial, np.asarray(scales)


def _initial_sp(gram, penalties):
    """mgcv initial.sp's diagonal balancing, on the centered design."""
    diagonal = np.diag(gram)
    result = []
    total = np.zeros_like(diagonal)
    penalized = np.zeros(len(diagonal), dtype=bool)
    for S in penalties:
        ds = np.diag(S)
        threshold = np.max(np.abs(S)) * _EPS ** 0.8
        positive = ((np.mean(np.abs(S), axis=0) > threshold)
                    & (np.mean(np.abs(S), axis=1) > threshold)
                    & (np.abs(ds) > threshold))
        sp = float(np.mean(diagonal[positive]) / np.mean(ds[positive]))
        result.append(sp)
        total += sp * ds
        penalized |= positive
    result = np.asarray(result)
    use = (total > 0) & penalized & (diagonal > 0)
    dx, ds = diagonal[use], total[use]
    while np.mean(dx / (dx + ds)) > .4:
        result *= 10
        ds *= 10
    while np.mean(dx / (dx + ds)) < .4:
        result /= 10
        ds /= 10
    return result


class _GaussianGCV:
    def __init__(self, covariance, setup):
        self.B, self.transform, self.R, self.roots, self.initial, self.scales = setup
        rhs = self.transform.T @ (self.B.T @ covariance @ self.B).ravel()
        self.y0 = linalg.solve_triangular(self.R.T, rhs, lower=True)
        self.yy = float(np.sum(covariance ** 2))
        self.n = covariance.size
        self.calls = 0

    def fit(self, log_sp, derivatives=False):
        sp = np.exp(log_sp)
        augmented = np.vstack((self.R, *(np.sqrt(s) * root.T
                                        for s, root in zip(sp, self.roots))))
        U, d, Vh = linalg.svd(augmented, full_matrices=False, check_finite=False)
        keep = d >= d[0] * _RANK_TOL
        d, V = d[keep], Vh[keep].T
        U1 = U[:len(self.y0), keep]
        y1 = U1.T @ self.y0
        norm = max(self.yy - 2 * (y1 @ y1) + np.sum((U1 @ y1) ** 2), 0.0)
        edf = float(np.sum(U1 ** 2))
        delta = self.n - edf
        score = self.n * norm / delta ** 2 if delta != 0 else np.inf
        coeff = (V / d) @ y1
        self.calls += 1
        state = dict(score=float(score), edf=edf, coefficients=coeff, rank=len(d))
        if not derivatives:
            return state

        UtU = U1.T @ U1
        M, K, My, Ky, yK = [], [], [], [], []
        for root in self.roots:
            t = (V.T @ root) / d[:, None]
            m = t @ t.T
            ki = m @ UtU
            M.append(m)
            K.append(ki)
            My.append(m @ y1)
            Ky.append(ki @ y1)
            yK.append(ki.T @ y1)
        ddelta = np.array([s * np.trace(ki) for s, ki in zip(sp, K)])
        dnorm = np.array([2 * s * (y1 @ (my - ky))
                          for s, my, ky in zip(sp, My, Ky)])
        d2delta, d2norm = np.empty((2, 2)), np.empty((2, 2))
        for i in range(2):
            for j in range(i + 1):
                dd = -2 * sp[i] * sp[j] * np.sum(M[j] * K[i])
                dn = 2 * sp[i] * sp[j] * (My[i] @ Ky[j] + My[j] @ Ky[i]
                     - 2 * (My[i] @ My[j]) + yK[i] @ My[j])
                d2delta[i, j] = d2delta[j, i] = dd
                d2norm[i, j] = d2norm[j, i] = dn
            d2delta[i, i] += ddelta[i]
            d2norm[i, i] += dnorm[i]
        f = self.n / delta ** 2
        g = 2 * f * norm / delta
        grad = f * dnorm - g * ddelta
        hess = (-2 * f / delta * (np.outer(ddelta, dnorm) + np.outer(dnorm, ddelta))
                + f * d2norm + 3 * g / delta * np.outer(ddelta, ddelta)
                - g * d2delta)
        state.update(gradient=grad, hessian=(hess + hess.T) / 2)
        return state


def _magic(problem, tol=1e-7, max_half=15):
    """Two-penalty specialization of mgcv 1.9-4 magic's default search."""
    log_sp = np.log(problem.initial)
    state = problem.fit(log_sp)
    min_score = state["score"]
    d_score = 1e10
    grad = np.zeros(2)
    sd_step = np.zeros(2)
    n_step = np.zeros(2)
    use_sd, step_fail = False, False
    rms_grad = 0.0
    for iteration in range(1, 401):
        attempt = 0
        if iteration > 1:
            step = sd_step if use_sd else n_step
            while True:
                attempt += 1
                if attempt == 4 and not use_sd:
                    use_sd, step = True, sd_step
                candidate = log_sp + step
                state = problem.fit(candidate)
                if state["score"] < min_score:
                    d_score = min_score - state["score"]
                    min_score, log_sp = state["score"], candidate
                    break
                step /= 2
                if attempt == max_half - 1:
                    step[:] = 0
                if attempt == max_half:
                    break
        converged = False
        if iteration > 3:
            grad_norm = linalg.norm(grad)
            converged = (d_score <= tol * (1 + min_score)
                         and grad_norm <= tol ** (1 / 3) * (1 + abs(min_score)))
            if attempt == max_half:
                converged, step_fail = True, True
            if converged:
                rms_grad = grad_norm / np.sqrt(2)

        state = problem.fit(log_sp, derivatives=True)
        grad = state["gradient"]
        vals, vecs = linalg.eigh(state["hessian"])
        use_sd = bool(np.any(vals < 0))
        max_grad = np.max(np.abs(grad))
        if max_grad == 0:
            # Zero and exactly null-space responses have zero GCV gradient.
            converged, sd_step, n_step = True, np.zeros(2), np.zeros(2)
        else:
            if not use_sd:
                if np.any(vals == 0):
                    use_sd = True
                else:
                    n_step = -vecs @ ((vecs.T @ grad) / vals)
                    size = np.max(np.abs(n_step))
                    if size > 5:
                        n_step *= 5 / size
            sd_step = -grad / max_grad
        if converged:
            break
    else:
        raise RuntimeError("survival covariance GCV optimizer failed to converge after 400 iterations")

    # magic checks that an apparently converged optimum is not at infinity.
    for j in range(2):
        direction = 2.0 if grad[j] < 0 else -2.0
        for _ in range(5):
            candidate = log_sp.copy()
            candidate[j] += direction
            trial = problem.fit(candidate)
            if trial["score"] < min_score:
                log_sp, min_score = candidate, trial["score"]
            else:
                break
    state = problem.fit(log_sp)
    info = {"method": "mgcv-1.9-4 Gaussian cr tensor GCV",
            "sp": np.exp(log_sp), "edf": state["edf"], "gcv": state["score"],
            "rank": state["rank"], "iterations": iteration,
            "converged": not step_fail, "rms_gradient": float(rms_grad),
            "penalty_scale": problem.scales.copy()}
    if step_fail:
        warnings.warn("survival covariance GCV search could not decrease its score at convergence",
                      RuntimeWarning, stacklevel=3)
    return state["coefficients"], info


def smooth_covariance(covariance, k_rank):
    """Return the symmetrized R-style tensor GAM fit and optimizer diagnostics."""
    covariance = np.asarray(covariance, dtype=float)
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
            or covariance.shape[0] < 3 or not np.all(np.isfinite(covariance))):
        raise ValueError("smoothing requires a finite square covariance with at least three times")
    n = covariance.shape[0]
    k = min(max(int(k_rank), 3), n)
    setup = _tensor_setup(n, k)
    if not np.any(covariance):
        # GCV is identically zero. Like magic's unchanged initial-parameter
        # solution, there is no identifiable smoothing optimum.
        problem = _GaussianGCV(covariance, setup)
        state = problem.fit(np.log(setup[4]))
        return covariance.copy(), {"method": "mgcv-1.9-4 Gaussian cr tensor GCV",
            "sp": setup[4].copy(), "edf": state["edf"], "gcv": 0.0,
            "rank": k * k, "iterations": 0, "converged": True,
            "rms_gradient": 0.0, "penalty_scale": setup[5].copy(),
            "flat_score": True}
    problem = _GaussianGCV(covariance, setup)
    coefficients, info = _magic(problem)
    full = setup[1] @ coefficients
    fitted = setup[0] @ full.reshape(k, k) @ setup[0].T
    return (fitted + fitted.T) / 2, info
