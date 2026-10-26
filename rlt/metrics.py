"""Standalone metrics using the same native definitions as RLT."""

import numpy as np

from ._core import cindex as _cindex


def cindex(time, event, risk):
    """Harrell's concordance for right-censored outcomes; larger risk is worse.

    Inputs must be vectors of equal length. Times must be finite, and events
    exactly 0 or 1. Nonfinite risks or no comparable pairs return NaN, matching
    the corrected R helper. For distinct comparable times, tied risks count
    one half. When both events occur at the same time, RLT counts tied risks
    as concordant and unequal risks as one half. At an event/censoring tie,
    the observed event is treated as earlier. Two censored ties are omitted.
    """
    vectors = []
    for value, name in ((time, "time"), (event, "event"), (risk, "risk")):
        array = np.asarray(value)
        if array.ndim != 1 or np.iscomplexobj(array):
            raise ValueError(f"{name} must be a real vector")
        vectors.append(np.asarray(array, dtype=np.float64))
    time, event, risk = vectors
    if not time.size == event.size == risk.size:
        raise ValueError("time, event and risk must have matching lengths")
    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain finite values")
    if not np.all(np.isin(event, (0, 1))):
        raise ValueError("event must contain only 0 or 1")
    return float(_cindex(time, event.astype(np.int64), risk))
