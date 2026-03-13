from importlib.metadata import version as _version

__version__ = _version("parametric_dr")

from .core import Parametric_tSNE
from .pacmap import Parametric_PaCMAP
from .umap import Parametric_UMAP
from .trimap import Parametric_TriMap
from .cebra import Parametric_CEBRA
from .temporal import TemporalMixin
from .metrics import (
    trustworthiness,
    continuity,
    neighborhood_preservation,
    shepard_correlation,
    trajectory_smoothness,
    compute_metrics,
)
from .analysis import (
    estimate_intrinsic_dim,
    sweep_dims,
)

__all__ = [
    "Parametric_tSNE",
    "Parametric_PaCMAP",
    "Parametric_UMAP",
    "Parametric_TriMap",
    "Parametric_CEBRA",
    "TemporalMixin",
    "trustworthiness",
    "continuity",
    "neighborhood_preservation",
    "shepard_correlation",
    "trajectory_smoothness",
    "compute_metrics",
    "estimate_intrinsic_dim",
    "sweep_dims",
]
