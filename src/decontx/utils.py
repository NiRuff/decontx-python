"""Utility functions for DecontX."""

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
from typing import Optional, Tuple
import warnings
import umap
from sklearn.cluster import DBSCAN

def initialize_clusters(
        adata: AnnData,
        var_genes: int = 5000,
        random_state: int = 12345
) -> np.ndarray:
    """
    Initialize cell clusters using UMAP + DBSCAN approach.

    This mirrors the R implementation's approach when no clusters are provided.
    """
    # Use scanpy for preprocessing and clustering
    adata_temp = adata.copy()

    # Find highly variable genes
    sc.pp.highly_variable_genes(adata_temp, n_top_genes=var_genes)
    adata_temp = adata_temp[:, adata_temp.var.highly_variable]

    # Normalize and log transform
    sc.pp.normalize_total(adata_temp)
    sc.pp.log1p(adata_temp)

    # PCA and UMAP
    sc.pp.pca(adata_temp, random_state=random_state)
    sc.pp.neighbors(adata_temp, random_state=random_state)
    sc.tl.umap(adata_temp, random_state=random_state)

    # Clustering
    sc.tl.leiden(adata_temp, random_state=random_state)

    return adata_temp.obs['leiden'].astype(int).values


def validate_inputs(adata: AnnData, z: Optional[str], batch_key: Optional[str]):
    """input validation matching R's error checking."""

    # Check for negative values
    if adata.X.min() < 0:
        raise ValueError("Count matrix contains negative values")

    # Check for missing values
    if np.any(np.isnan(adata.X.data if issparse(adata.X) else adata.X)):
        raise ValueError("Count matrix contains NaN values")

    # Check dimensions
    if adata.n_obs < 2:
        raise ValueError("At least 2 cells required for decontamination")

    if adata.n_vars < 2:
        raise ValueError("At least 2 genes required for decontamination")

    # Check cluster key
    if z is not None and isinstance(z, str) and z not in adata.obs:
        raise KeyError(f"Cluster key '{z}' not found in adata.obs")

    # Check batch key
    if batch_key is not None and batch_key not in adata.obs:
        raise KeyError(f"Batch key '{batch_key}' not found in adata.obs")

    # Check for sufficient cells per batch
    if batch_key is not None:
        batch_counts = adata.obs[batch_key].value_counts()
        small_batches = batch_counts[batch_counts < 10].index.tolist()
        if small_batches:
            warnings.warn(f"Small batches detected (< 10 cells): {small_batches}")
