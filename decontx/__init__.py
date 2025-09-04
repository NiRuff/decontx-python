"""
DecontX: Decontamination of ambient RNA in single-cell RNA-seq data

Python implementation of the DecontX algorithm for estimating and removing
contamination in individual cells from ambient RNA.
"""

__version__ = "0.1.0"

from .core import (
    decontx,
    # get_decontx_counts,
    # get_decontx_contamination,
    # get_decontx_clusters
)

from .simulation import simulate_contamination
from .plotting import (
    plot_contamination_umap,
    plot_marker_percentage,
    plot_marker_expression,
    plot_contamination_comparison
)

__all__ = [
    "decontx",
    "simulate_contamination",
    "plot_contamination_umap",
    "plot_marker_percentage",
    "plot_marker_expression",
    "plot_contamination_comparison",
    "get_decontx_counts",
    "get_decontx_contamination",
    "get_decontx_clusters"
]
