"""Variable-importance summary table (port of R's importance.RLT).

Numpy-backed on purpose: pandas is not a dependency of rlt.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ImportanceTable", "build_importance_table"]

_SIG_BREAKS = (2.58, 1.96, 1.64)
_SIG_CODES = ("***", "**", "*")


def _sig_code(z):
    if z is None or np.isnan(z):
        return ""
    az = abs(z)
    for br, code in zip(_SIG_BREAKS, _SIG_CODES):
        if az >= br:
            return code
    return ""


class ImportanceTable:
    """Variable-importance table with optional SD / Z / significance.

    Mirrors R's ``importance.RLT`` data.frame: columns ``Variable`` and
    ``VI``, plus ``SD``, ``Z`` and ``Sig`` when variance estimation was
    enabled (``var_mode='matched'``) at fit time. Negative variance
    estimates give NaN SD / NaN Z and an empty significance code, exactly
    like R. ``repr()`` reproduces R's ``print.importance.RLT`` layout.
    """

    def __init__(self, variable, vi, var_vi=None):
        self.variable = list(variable)
        self.vi = np.asarray(vi, dtype=np.float64)
        p = self.vi.shape[0]
        if len(self.variable) != p:
            raise ValueError("variable names and VI length mismatch")
        self.has_variance = var_vi is not None
        if self.has_variance:
            vvi = np.asarray(var_vi, dtype=np.float64)
            if vvi.shape[0] != p:
                raise ValueError("VarVI length does not match VI length")
            sd = np.full(p, np.nan)
            ok = vvi >= 0
            sd[ok] = np.sqrt(vvi[ok])
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(ok, self.vi / sd, np.nan)
            self.sd = sd
            self.z = z
            self.sig = [_sig_code(zi) for zi in z]
        else:
            self.sd = None
            self.z = None
            self.sig = None

    def __len__(self):
        return self.vi.shape[0]

    def __repr__(self):
        lines = []
        if self.has_variance:
            assert self.sd is not None and self.z is not None \
                and self.sig is not None
            lines.append(f"{'Variable':<12} {'VI':>10} {'SD':>12} "
                         f"{'Z':>10}  Sig")
            lines.append("-" * 58)
            n_neg = 0
            for name, vi, sd, z, sig in zip(self.variable, self.vi, self.sd,
                                            self.z, self.sig):
                if np.isnan(sd):
                    sd_str, z_str, sig_str = "NA", "NA", ""
                    n_neg += 1
                else:
                    sd_str = f"{sd:12.6f}"
                    z_str = f"{z:10.2f}"
                    sig_str = sig
                lines.append(f"{str(name):<12} {vi:10.4f} {sd_str:>12} "
                             f"{z_str:>10}  {sig_str}")
            if n_neg > 0:
                lines.append("")
                lines.append(
                    f"Note: {n_neg} variable(s) with negative variance "
                    f"estimate (SD, Z shown as NA)")
        else:
            lines.append(f"{'Variable':<12} {'VI':>10}")
            lines.append("-" * 26)
            for name, vi in zip(self.variable, self.vi):
                lines.append(f"{str(name):<12} {vi:10.4f}")
        return "\n".join(lines)

    def _repr_html_(self):  # notebook-friendly
        return "<pre style='font-family:monospace'>" + repr(self) + "</pre>"


def build_importance_table(model):
    """Build an ImportanceTable from a fitted estimator (used by
    ``RLT_reg/cla/surv.importance_table()`` and ``rlt.importance()``)."""
    vi = np.asarray(model.varimp_, dtype=np.float64)
    p = vi.shape[0]
    varnames = getattr(model, "feature_names_in_", None)
    if varnames is None or len(varnames) != p:
        varnames = [f"V{i + 1}" for i in range(p)]
    else:
        varnames = [str(v) for v in varnames]
    vvi = getattr(model, "var_vi_", None)
    if vvi is not None and np.asarray(vvi).shape[0] == p:
        return ImportanceTable(variable=varnames, vi=vi,
                               var_vi=np.asarray(vvi, dtype=np.float64))
    return ImportanceTable(variable=varnames, vi=vi)
