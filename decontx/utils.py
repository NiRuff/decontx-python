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
        seed: int = 12345
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
    sc.pp.pca(adata_temp, random_state=seed)
    sc.pp.neighbors(adata_temp, random_state=seed)
    sc.tl.umap(adata_temp, random_state=seed)

    # Clustering
    sc.tl.leiden(adata_temp, random_state=seed)

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


def retrieve_feature_index_fast(features, search_space, exact_match=True):
    """Fast feature matching for numpy arrays"""
    n_features = len(features)
    indices = np.full(n_features, -1, dtype=np.int64)

    if exact_match:
        for i in range(n_features):
            matches = np.where(search_space == features[i])[0]
            if len(matches) > 0:
                indices[i] = matches[0]
    else:
        # Partial matching
        for i in range(n_features):
            feature = str(features[i])
            matches = []
            for j, item in enumerate(search_space):
                if feature in str(item):
                    matches.append(j)
            if len(matches) == 1:
                indices[i] = matches[0]
            elif len(matches) > 1:
                indices[i] = matches[0]  # Take first match

    return indices


def checkCountsDecon(counts):
    """Equivalent to R's .checkCountsDecon"""
    return _checkCountsDecon(counts)


def processCellLabels(z, numCells):
    """Equivalent to R's .processCellLabels"""
    if len(z) != numCells:
        raise ValueError(f"'z' must be of the same length as the number of cells ({numCells}).")

    unique_labels = np.unique(z)
    if len(unique_labels) < 2:
        raise ValueError("No need to decontaminate when only one cluster is in the dataset.")

    # Convert to factor-like behavior
    if not np.issubdtype(z.dtype, np.integer):
        # Map unique values to sequential integers
        label_map = {label: i + 1 for i, label in enumerate(unique_labels)}
        z = np.array([label_map[label] for label in z])

    return z.astype(int)


def checkDelta(delta):
    """Equivalent to R's .checkDelta"""
    if not isinstance(delta, (list, tuple, np.ndarray)) or len(delta) != 2:
        raise ValueError("'delta' needs to be a numeric vector of length 2 containing positive values.")

    if not all(isinstance(d, (int, float)) and d > 0 for d in delta):
        raise ValueError("'delta' needs to be a numeric vector of length 2 containing positive values.")

    return np.array(delta, dtype=float)

