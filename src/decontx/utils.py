"""Utility functions for DecontX."""

import numpy as np
import scanpy as sc
from anndata import AnnData
from typing import Optional


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

def process_var_genes(var_genes):
    """R's .processvarGenes"""
    if var_genes is None:
        var_genes = 5000
    elif var_genes < 2 or not isinstance(var_genes, int):
        raise ValueError("Parameter 'varGenes' must be an integer larger than 1")
    return var_genes

def process_dbscan_eps(dbscan_eps):
    """R's .processdbscanEps"""
    if dbscan_eps is None:
        dbscan_eps = 1
    elif dbscan_eps < 0:
        raise ValueError("Parameter 'dbscanEps' needs to be non-negative")
    return dbscan_eps

def check_delta(delta):
    """R's .checkDelta"""
    if not isinstance(delta, (list, tuple, np.ndarray)) or len(delta) != 2:
        raise ValueError("'delta' needs to be a numeric vector of length 2 containing positive values")
    if any(d < 0 for d in delta):
        raise ValueError("'delta' values must be positive")
    return delta

def decontx_initialize_z_exact(
        adata,
        var_genes: int = 5000,
        dbscan_eps: float = 1.0,
        random_state: int = 12345
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact equivalent of R's .decontxInitializeZ function.
    Matches R's preprocessing pipeline exactly.
    """
    # Work on copy to avoid modifying original
    adata_temp = adata.copy()

    # Filter genes (match R's filtering)
    sc.pp.filter_genes(adata_temp, min_counts=1)

    # Exact normalization matching R's scater::logNormCounts
    # This matches R's normalization more closely than scanpy's default
    sc.pp.normalize_total(adata_temp, target_sum=1e4)
    sc.pp.log1p(adata_temp)

    # Find highly variable genes (match R's approach)
    if adata_temp.n_vars > var_genes:
        sc.pp.highly_variable_genes(
            adata_temp,
            n_top_genes=var_genes,
            flavor='seurat_v3'  # Closer to R's approach
        )
        adata_temp = adata_temp[:, adata_temp.var.highly_variable]

    # PCA with same parameters as R
    sc.pp.pca(adata_temp, n_comps=30, random_state=random_state)

    # UMAP matching R's parameters exactly
    # R uses: minDist=0.01, spread=1, nNeighbors=15
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.01,
        spread=1.0,
        n_components=2,
        random_state=random_state,
        metric='euclidean'
    )
    umap_coords = reducer.fit_transform(adata_temp.obsm['X_pca'])

    # DBSCAN clustering with adaptive eps (matches R logic exactly)
    n_clusters = 1
    eps = dbscan_eps
    max_tries = 10

    while n_clusters <= 1 and eps > 0 and max_tries > 0:
        clusterer = DBSCAN(eps=eps, min_samples=3)
        cluster_labels = clusterer.fit_predict(umap_coords)

        # Count non-noise clusters
        n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
        eps *= 0.75  # Same reduction as R
        max_tries -= 1

    # Fallback to k-means if DBSCAN fails (matches R)
    if n_clusters <= 1:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(umap_coords)

    # Convert to 1-indexed (matches R)
    cluster_labels = cluster_labels + 1

    return cluster_labels, umap_coords