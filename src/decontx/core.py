"""Core DecontX functionality."""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
import scanpy as sc
from anndata import AnnData
from .model import DecontXModel
from .utils import initialize_clusters, validate_inputs


def decontx(
        adata: AnnData,
        z: Optional[Union[str, np.ndarray]] = None,
        batch_key: Optional[str] = None,
        max_iter: int = 500,
        convergence_threshold: float = 0.001,
        delta: Tuple[float, float] = (10.0, 10.0),
        estimate_delta: bool = True,
        var_genes: int = 5000,
        random_state: int = 12345,
        copy: bool = False,
        verbose: bool = True,
) -> Optional[AnnData]:
    """
    Estimate and remove ambient RNA contamination using DecontX.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with cells in rows and genes in columns.
    z : str or array-like, optional
        Cell cluster assignments. If str, uses adata.obs[z]. If None,
        performs automatic clustering.
    batch_key : str, optional
        Key in adata.obs for batch information.
    max_iter : int, default 500
        Maximum number of EM iterations.
    convergence_threshold : float, default 0.001
        Convergence threshold for EM algorithm.
    delta : tuple, default (10.0, 10.0)
        Beta distribution parameters for contamination proportion.
    estimate_delta : bool, default True
        Whether to estimate delta parameters during inference.
    var_genes : int, default 5000
        Number of highly variable genes for clustering (if z is None).
    random_state : int, default 12345
        Random seed for reproducibility.
    copy : bool, default False
        Return a copy instead of modifying adata in place.
    verbose : bool, default True
        Print progress messages.

    Returns
    -------
    AnnData or None
        If copy=True, returns new AnnData with decontaminated counts
        in .layers['decontx_counts'] and contamination estimates in
        .obs['decontx_contamination']. Otherwise modifies adata in place.
    """
    if copy:
        adata = adata.copy()

    # Validate inputs
    validate_inputs(adata, z, batch_key)

    # Get cluster labels
    if z is None:
        if verbose:
            print("No cluster labels provided. Performing automatic clustering...")
        z_labels = initialize_clusters(adata, var_genes=var_genes,
                                       random_state=random_state)
    elif isinstance(z, str):
        z_labels = adata.obs[z].values
    else:
        z_labels = np.asarray(z)

    # Handle batches
    if batch_key is not None:
        batches = adata.obs[batch_key].unique()
        contamination = np.zeros(adata.n_obs)
        decontx_counts = adata.X.copy()

        for batch in batches:
            if verbose:
                print(f"Processing batch: {batch}")
            batch_mask = adata.obs[batch_key] == batch
            batch_adata = adata[batch_mask].copy()
            batch_z = z_labels[batch_mask]

            # Run DecontX on batch
            model = DecontXModel(
                max_iter=max_iter,
                convergence_threshold=convergence_threshold,
                delta=delta,
                estimate_delta=estimate_delta,
                random_state=random_state,
                verbose=verbose
            )

            batch_result = model.fit_transform(batch_adata.X, batch_z)
            contamination[batch_mask] = batch_result['contamination']
            decontx_counts[batch_mask] = batch_result['decontaminated_counts']
    else:
        # Single batch processing
        model = DecontXModel(
            max_iter=max_iter,
            convergence_threshold=convergence_threshold,
            delta=delta,
            estimate_delta=estimate_delta,
            random_state=random_state,
            verbose=verbose
        )

        result = model.fit_transform(adata.X, z_labels)
        contamination = result['contamination']
        decontx_counts = result['decontaminated_counts']

    # Store results
    adata.layers['decontx_counts'] = decontx_counts
    adata.obs['decontx_contamination'] = contamination
    adata.obs['decontx_clusters'] = z_labels

    if copy:
        return adata


def simulate_contamination(
        n_cells: int = 300,
        n_genes: int = 100,
        n_clusters: int = 3,
        n_counts_range: Tuple[int, int] = (500, 1000),
        contamination_params: Tuple[float, float] = (1.0, 10.0),
        random_state: int = 12345,
) -> dict:
    """
    Simulate contaminated count matrix for testing DecontX.

    Returns dictionary with 'counts', 'native_counts', 'contamination_true',
    'clusters', and other simulation parameters.
    """
    # Implementation would mirror the R simulateContamination function
    # This is a placeholder for the initial structure
    pass