"""
damps — Spectral Domain Representation Calibration package
============================================================

Public API
----------

::

    from damps import DAMPS, SlimMomentumEncoder, DualPathKNN
    from damps import compute_avrf_prior, compute_avrf_logit
    from damps.graph import adj_nnz, adj_avg_degree
"""

from .core import DAMPS
from .momentum import SlimMomentumEncoder
from .graph import DualPathKNN, adj_nnz, adj_avg_degree
from .prior import compute_avrf_prior, compute_avrf_logit

__all__ = [
    "DAMPS",
    "SlimMomentumEncoder",
    "DualPathKNN",
    "adj_nnz",
    "adj_avg_degree",
    "compute_avrf_prior",
    "compute_avrf_logit",
]
