"""rlt: Reinforcement Learning Trees for Python.

sklearn-compatible random forests for regression, classification, and
survival analysis, ported from the RLT R package.
"""

from .estimators import RLT_reg, RLT_cla, RLT_surv
from .importance import ImportanceTable
from .bands import get_surv_band
from .metrics import cindex


def importance(model):
    """Variable-importance summary for a fitted RLT model.

    Port of R's ``importance(fit)`` idiom; equivalent to
    ``model.importance_table()``. Returns an :class:`ImportanceTable`
    (Variable / VI, plus SD / Z / Sig when fitted with
    ``var_mode='matched'``).
    """
    return model.importance_table()


__version__ = "0.3.0"
__all__ = ["RLT_reg", "RLT_cla", "RLT_surv",
           "ImportanceTable", "importance", "get_surv_band", "cindex"]
