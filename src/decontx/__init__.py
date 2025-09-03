"""
DecontX: Decontamination of ambient RNA in single-cell RNA-seq data

Python implementation of the DecontX algorithm for estimating and removing
contamination in individual cells from ambient RNA.
"""

from .core import decontx, simulate_contamination
from .model import DecontXModel
from .plotting import plot_contamination_umap, plot_marker_expression

__version__ = "0.1.0"
__all__ = [
    "decontx",
    "simulate_contamination",
    "DecontXModel",
    "plot_contamination_umap",
    "plot_marker_expression",
]