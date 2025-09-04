"""
DecontX: Decontamination of ambient RNA in single-cell RNA-seq data

Python implementation of the DecontX algorithm for estimating and removing
contamination in individual cells from ambient RNA.
"""

from .model import DecontXModel
from .plotting import plot_contamination_umap, plot_marker_expression

from .core import decontx as decontx
from .core import (
    simulate_contamination as simulate_contamination,
    get_decontx_counts,
    get_decontx_contamination,
    get_decontx_clusters
)

__version__ = "0.1.0"
__all__ = [
    "decontx",
    "simulate_contamination",
    "DecontXModel",
    "plot_contamination_umap",
    "plot_marker_expression",
]