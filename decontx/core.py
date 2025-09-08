"""
Streamlined DecontX core functionality for scanpy integration.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
from anndata import AnnData
from scipy.sparse import issparse
import warnings
from datetime import datetime

from .model import DecontXModel


def decontx(
    adata: AnnData,
    cluster_key: str = "leiden",
    batch_key: Optional[str] = None,
    max_iter: int = 500,
    delta: Tuple[float, float] = (10.0, 10.0),
    estimate_delta: bool = True,
    convergence: float = 0.001,
    seed: int = 12345,
    copy: bool = False,
    verbose: bool = True
) -> Optional[AnnData]:
    """
    Remove ambient RNA contamination from single-cell RNA-seq data.

    This function implements the DecontX algorithm to estimate and remove
    cross-contamination from ambient RNA in each cell. Results are stored
    directly in the AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with raw counts in .X
    cluster_key : str, default "leiden"
        Column in .obs containing cluster labels
    batch_key : str, optional
        Column in .obs containing batch labels for separate processing
    max_iter : int, default 500
        Maximum number of EM algorithm iterations
    delta : tuple of float, default (10.0, 10.0)
        Beta distribution parameters (native_prior, contamination_prior)
    estimate_delta : bool, default True
        Whether to estimate delta parameters during EM
    convergence : float, default 0.001
        Convergence threshold for EM algorithm
    seed : int, default 12345
        Random seed for reproducibility
    copy : bool, default False
        Return copy of adata or modify in place
    verbose : bool, default True
        Print progress messages

    Returns
    -------
    AnnData or None
        If copy=True, returns modified AnnData object.
        If copy=False, modifies adata in place and returns None.

    Notes
    -----
    Results are stored in:
    - adata.obs['decontX_contamination']: Per-cell contamination estimates (0-1)
    - adata.layers['decontX_counts']: Decontaminated count matrix
    - adata.uns['decontX']: Model parameters and run information

    Examples
    --------
    >>> import scanpy as sc
    >>> import decontx
    >>>
    >>> # Standard scanpy workflow
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> sc.pp.filter_cells(adata, min_genes=200)
    >>> sc.pp.filter_genes(adata, min_cells=3)
    >>> sc.tl.leiden(adata)
    >>>
    >>> # Remove contamination
    >>> decontx.decontx(adata, cluster_key="leiden")
    >>>
    >>> # Check results
    >>> print(f"Mean contamination: {adata.obs['decontX_contamination'].mean():.1%}")
    """

    if copy:
        adata = adata.copy()

    start_time = datetime.now()

    if verbose:
        print("=" * 50)
        print("Starting DecontX")
        print("=" * 50)

    # Input validation
    _validate_inputs(adata, cluster_key, batch_key)

    # Get cluster labels
    if cluster_key not in adata.obs:
        raise KeyError(f"Cluster key '{cluster_key}' not found in adata.obs")

    z_labels = adata.obs[cluster_key].values
    z_labels = _process_cluster_labels(z_labels)

    if verbose:
        n_clusters = len(np.unique(z_labels))
        print(f"Processing {adata.n_obs} cells, {adata.n_vars} genes")
        print(f"Using {n_clusters} clusters from '{cluster_key}'")

    # Process batches if specified
    if batch_key is not None:
        if batch_key not in adata.obs:
            raise KeyError(f"Batch key '{batch_key}' not found in adata.obs")

        batch_labels = adata.obs[batch_key].values
        unique_batches = np.unique(batch_labels)

        if verbose:
            print(f"Processing {len(unique_batches)} batches separately")

        results = _process_batches(
            adata, z_labels, batch_labels, unique_batches,
            max_iter, delta, estimate_delta, convergence, seed, verbose
        )
    else:
        # Single batch processing
        if verbose:
            print("Processing as single batch")

        result = _run_decontx_single(
            adata.X, z_labels, max_iter, delta, estimate_delta,
            convergence, seed, verbose
        )
        results = {"all": result}

    # Store results
    _store_results(adata, results, z_labels, cluster_key, batch_key)

    # Store metadata
    _store_metadata(adata, delta, estimate_delta, max_iter, convergence, seed, start_time)

    if verbose:
        contamination = adata.obs['decontX_contamination']
        print(f"Mean contamination: {contamination.mean():.1%}")
        print(f"Highly contaminated cells (>50%): {(contamination > 0.5).sum()}")

        end_time = datetime.now()
        print("=" * 50)
        print(f"Completed DecontX in {end_time - start_time}")
        print("=" * 50)

    if copy:
        return adata
    return None


def _validate_inputs(adata: AnnData, cluster_key: str, batch_key: Optional[str]):
    """Validate input parameters."""

    # Check for negative values
    if adata.X.min() < 0:
        raise ValueError("Count matrix contains negative values")

    # Check for missing values
    if issparse(adata.X):
        if np.any(np.isnan(adata.X.data)):
            raise ValueError("Count matrix contains NaN values")
    else:
        if np.any(np.isnan(adata.X)):
            raise ValueError("Count matrix contains NaN values")

    # Check dimensions
    if adata.n_obs < 10:
        warnings.warn("Very few cells (<10) detected. Results may be unreliable.")

    if adata.n_vars < 100:
        warnings.warn("Very few genes (<100) detected. Results may be unreliable.")


def _process_cluster_labels(z: np.ndarray) -> np.ndarray:
    """Process cluster labels to ensure proper format."""

    z = np.asarray(z)

    # Check for sufficient clusters
    unique_labels = np.unique(z)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 clusters for decontamination")

    # Convert to sequential integers starting from 1 (R compatibility)
    if not np.issubdtype(z.dtype, np.integer):
        label_map = {label: i + 1 for i, label in enumerate(unique_labels)}
        z = np.array([label_map[x] for x in z])
    else:
        # Ensure sequential starting from 1
        min_label = np.min(z)
        if min_label <= 0:
            z = z - min_label + 1
        elif min_label > 1:
            label_map = {label: i + 1 for i, label in enumerate(np.sort(unique_labels))}
            z = np.array([label_map[x] for x in z])

    return z.astype(int)


def _process_batches(
    adata: AnnData,
    z_labels: np.ndarray,
    batch_labels: np.ndarray,
    unique_batches: np.ndarray,
    max_iter: int,
    delta: Tuple[float, float],
    estimate_delta: bool,
    convergence: float,
    seed: int,
    verbose: bool
) -> dict:
    """Process multiple batches separately."""

    batch_results = {}

    for batch in unique_batches:
        if verbose:
            print(f"  Processing batch '{batch}'...")

        # Get batch data
        batch_mask = batch_labels == batch
        batch_indices = np.where(batch_mask)[0]

        if issparse(adata.X):
            X_batch = adata.X[batch_mask].tocsr()
        else:
            X_batch = adata.X[batch_mask]

        z_batch = z_labels[batch_mask]

        # Run decontamination
        result = _run_decontx_single(
            X_batch, z_batch, max_iter, delta, estimate_delta,
            convergence, seed, verbose=False
        )

        # Store with batch info
        result['batch_indices'] = batch_indices
        result['batch_name'] = batch
        batch_results[batch] = result

        if verbose:
            contamination = result['contamination']
            print(f"    Mean contamination: {contamination.mean():.1%}")

    return batch_results


def _run_decontx_single(
    X: Union[np.ndarray, "csr_matrix"],
    z_labels: np.ndarray,
    max_iter: int,
    delta: Tuple[float, float],
    estimate_delta: bool,
    convergence: float,
    seed: int,
    verbose: bool = True
) -> dict:
    """Run DecontX on a single batch."""

    model = DecontXModel(
        max_iter=max_iter,
        delta=delta,
        estimate_delta=estimate_delta,
        convergence=convergence,
        seed=seed,
        verbose=verbose
    )

    result = model.fit_transform(X, z_labels)
    return result


def _store_results(
    adata: AnnData,
    results: dict,
    z_labels: np.ndarray,
    cluster_key: str,
    batch_key: Optional[str]
):
    """Store decontX results in AnnData object."""

    n_cells = adata.n_obs
    n_genes = adata.n_vars

    if len(results) == 1 and "all" in results:
        # Single batch
        result = results["all"]
        adata.layers['decontX_counts'] = result['decontaminated_counts']
        adata.obs['decontX_contamination'] = result['contamination']
    else:
        # Multiple batches - combine results
        decontx_counts = np.zeros((n_cells, n_genes))
        contamination = np.zeros(n_cells)

        for batch_name, result in results.items():
            if 'batch_indices' in result:
                batch_indices = result['batch_indices']
                decontx_counts[batch_indices] = result['decontaminated_counts']
                contamination[batch_indices] = result['contamination']

        adata.layers['decontX_counts'] = decontx_counts
        adata.obs['decontX_contamination'] = contamination

    # Store cluster labels used
    adata.obs['decontX_clusters'] = pd.Categorical(z_labels)


def _store_metadata(
    adata: AnnData,
    delta: Tuple[float, float],
    estimate_delta: bool,
    max_iter: int,
    convergence: float,
    seed: int,
    start_time: datetime
):
    """Store run parameters and metadata."""

    end_time = datetime.now()

    metadata = {
        'parameters': {
            'delta': delta,
            'estimate_delta': estimate_delta,
            'max_iter': max_iter,
            'convergence': convergence,
            'seed': seed
        },
        'runtime': {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds()
        },
        'version': "0.2.0"
    }

    adata.uns['decontX'] = metadata


# Utility functions for accessing results

def get_decontx_counts(adata: AnnData) -> np.ndarray:
    """Get decontaminated counts matrix."""
    if 'decontX_counts' not in adata.layers:
        raise KeyError("DecontX counts not found. Run decontx() first.")
    return adata.layers['decontX_counts']


def get_decontx_contamination(adata: AnnData) -> np.ndarray:
    """Get contamination estimates."""
    if 'decontX_contamination' not in adata.obs:
        raise KeyError("DecontX contamination not found. Run decontx() first.")
    return adata.obs['decontX_contamination'].values


def get_decontx_clusters(adata: AnnData) -> np.ndarray:
    """Get cluster labels used by DecontX."""
    if 'decontX_clusters' not in adata.obs:
        raise KeyError("DecontX clusters not found. Run decontx() first.")
    return adata.obs['decontX_clusters'].values