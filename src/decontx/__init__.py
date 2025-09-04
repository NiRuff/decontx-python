"""
DecontX: Decontamination of ambient RNA in single-cell RNA-seq data

Python implementation of the DecontX algorithm for estimating and removing
contamination in individual cells from ambient RNA.
"""

__version__ = "0.1.0"

from .core import (
    decontx,
    simulate_contamination,
    get_decontx_counts,
    get_decontx_contamination,
    get_decontx_clusters
)
from .plotting import (
    plot_decontx_contamination,
    plot_decontx_marker_percentage,
    plot_decontx_marker_expression
)

__all__ = [
    "decontx",
    "simulate_contamination",
    "DecontXModel",
    "plot_decontx_contamination",
    "plot_decontx_marker_percentage",
    "plot_decontx_marker_expression",
    "get_decontx_counts",
    "get_decontx_contamination",
    "get_decontx_clusters"
]

