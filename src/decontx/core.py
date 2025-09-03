"""
Core DecontX functionality with complete feature parity to R version.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, List, Dict
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse, csr_matrix
from tqdm import tqdm
import warnings

from .model import DecontXModel
from .utils import initialize_clusters, validate_inputs
from .fast_ops import calculate_native_matrix_fast


def decontx(
    adata: AnnData,
    z: Optional[Union[str, np.ndarray]] = None,
    batch_key: Optional[str] = None,
    background: Optional[AnnData] = None,
    bg_batch_key: Optional[str] = None,
    max_iter: int = 500,
    convergence_threshold: float = 0.001,
    iter_loglik: int = 10,
    delta: Tuple[float, float] = (10.0, 10.0),
    estimate_delta: bool = True,
    var_genes: int = 5000,
    dbscan_eps: float = 1.0,
    random_state: int = 12345,
    copy: bool = False,
    verbose: bool = True,
    logfile: Optional[str] = None,
) -> Optional[AnnData]:
    """
    Estimate and remove ambient RNA contamination using DecontX.

    Full implementation matching R package functionality.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with cells × genes.
    z : str, array-like, or None
        Cell cluster assignments. If str, uses adata.obs[z].
        If None, performs automatic clustering.
    batch_key : str, optional
        Key in adata.obs for batch information. DecontX runs
        separately on each batch.
    background : AnnData, optional
        Empty droplet matrix for empirical contamination distribution.
        Should have same genes as adata.
    bg_batch_key : str, optional
        Batch key for background matrix.
    max_iter : int, default 500
        Maximum number of EM iterations.
    convergence_threshold : float, default 0.001
        Convergence threshold for EM algorithm.
    iter_loglik : int, default 10
        Calculate log-likelihood every N iterations.
    delta : tuple, default (10.0, 10.0)
        Beta distribution parameters for contamination proportion prior.
    estimate_delta : bool, default True
        Whether to update delta during inference.
    var_genes : int, default 5000
        Number of variable genes for clustering (if z is None).
    dbscan_eps : float, default 1.0
        DBSCAN epsilon parameter for clustering.
    random_state : int, default 12345
        Random seed for reproducibility.
    copy : bool, default False
        Return a copy instead of modifying in place.
    verbose : bool, default True
        Print progress messages.
    logfile : str, optional
        File to redirect log messages.

    Returns
    -------
    AnnData or None
        If copy=True, returns new AnnData with:
        - .layers['decontX_counts']: decontaminated counts
        - .obs['decontX_contamination']: contamination estimates
        - .obs['decontX_clusters']: cluster labels used
        - .obsm['X_decontX_umap']: UMAP if clustering performed
        - .uns['decontX']: dictionary with all parameters
    """
    if copy:
        adata = adata.copy()

    # Setup logging
    log_messages = []
    def log(msg):
        if verbose:
            print(msg)
        log_messages.append(msg)
        if logfile:
            with open(logfile, 'a') as f:
                f.write(msg + '\n')

    log("=" * 50)
    log("Starting DecontX")
    log("=" * 50)

    # Validate inputs
    validate_inputs(adata, z, batch_key)

    # Process background matrix if provided
    background_dict = {}
    if background is not None:
        background_dict = _process_background(
            adata, background, batch_key, bg_batch_key, log
        )

    # Get or generate cluster labels
    z_labels, umap_coords = _process_clusters(
        adata, z, var_genes, dbscan_eps, random_state, log
    )

    # Process batches
    if batch_key is not None:
        results = _process_batches(
            adata, z_labels, batch_key, background_dict,
            max_iter, convergence_threshold, iter_loglik,
            delta, estimate_delta, random_state, verbose, log
        )
    else:
        # Single batch processing
        X_bg = background_dict.get('all', None)
        results = _run_decontx_single(
            adata.X, z_labels, X_bg,
            max_iter, convergence_threshold, iter_loglik,
            delta, estimate_delta, random_state, verbose, log
        )
        results['batch'] = 'all'

    # Store results in AnnData
    _store_results(adata, results, z_labels, umap_coords)

    log("=" * 50)
    log("Completed DecontX")
    log("=" * 50)

    # Store log
    if 'decontX' not in adata.uns:
        adata.uns['decontX'] = {}
    adata.uns['decontX']['log'] = log_messages

    if copy:
        return adata
    return None


def _process_background(
    adata: AnnData,
    background: AnnData,
    batch_key: Optional[str],
    bg_batch_key: Optional[str],
    log
) -> Dict:
    """Process background/empty droplet matrix."""

    # Check for overlapping barcodes
    if background.obs_names is not None and adata.obs_names is not None:
        overlap = adata.obs_names.intersection(background.obs_names)
        if len(overlap) > 0:
            log(f".. Removing {len(overlap)} overlapping cells from background")
            background = background[~background.obs_names.isin(overlap)]

    # Check genes match
    if not all(g in background.var_names for g in adata.var_names):
        warnings.warn("Not all genes in adata found in background. Reordering...")
        common_genes = adata.var_names.intersection(background.var_names)
        adata = adata[:, common_genes]
        background = background[:, common_genes]

    # Organize by batch
    bg_dict = {}
    if batch_key is not None and bg_batch_key is not None:
        for batch in adata.obs[batch_key].unique():
            if batch in background.obs[bg_batch_key].values:
                bg_dict[batch] = background[background.obs[bg_batch_key] == batch].X
    else:
        bg_dict['all'] = background.X

    return bg_dict


def _process_clusters(
    adata: AnnData,
    z: Optional[Union[str, np.ndarray]],
    var_genes: int,
    dbscan_eps: float,
    random_state: int,
    log
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Get or generate cluster labels."""

    umap_coords = None

    if z is None:
        log(".. No cluster labels provided. Performing automatic clustering...")
        z_labels, umap_coords = _initialize_clusters_umap(
            adata, var_genes, dbscan_eps, random_state
        )
        log(f".... Generated {len(np.unique(z_labels))} clusters")
    elif isinstance(z, str):
        if z not in adata.obs:
            raise KeyError(f"Cluster key '{z}' not found in adata.obs")
        z_labels = adata.obs[z].values
    else:
        z_labels = np.asarray(z)

    # Convert to numeric if needed
    if not np.issubdtype(z_labels.dtype, np.integer):
        unique_labels = np.unique(z_labels)
        label_map = {label: i+1 for i, label in enumerate(unique_labels)}
        z_labels = np.array([label_map[x] for x in z_labels])

    return z_labels, umap_coords


def _initialize_clusters_umap(
    adata: AnnData,
    var_genes: int = 5000,
    dbscan_eps: float = 1.0,
    random_state: int = 12345
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize clusters using variable genes -> PCA -> UMAP -> DBSCAN.
    Matches R's .decontxInitializeZ function.
    """
    import umap
    from sklearn.cluster import DBSCAN

    # Work on a copy
    adata_temp = adata.copy()

    # Normalize and find variable genes
    sc.pp.normalize_total(adata_temp)
    sc.pp.log1p(adata_temp)
    sc.pp.highly_variable_genes(adata_temp, n_top_genes=min(var_genes, adata.n_vars))
    adata_temp = adata_temp[:, adata_temp.var.highly_variable]

    # PCA
    sc.pp.pca(adata_temp, random_state=random_state)

    # UMAP
    reducer = umap.UMAP(n_neighbors=15, random_state=random_state)
    umap_coords = reducer.fit_transform(adata_temp.obsm['X_pca'][:, :30])

    # DBSCAN clustering with adaptive eps
    n_clusters = 1
    eps = dbscan_eps
    max_tries = 10

    while n_clusters <= 1 and eps > 0 and max_tries > 0:
        clusterer = DBSCAN(eps=eps, min_samples=3)
        clusters = clusterer.fit_predict(umap_coords)
        n_clusters = len(np.unique(clusters[clusters >= 0]))
        eps *= 0.75
        max_tries -= 1

    # If DBSCAN fails, fallback to k-means
    if n_clusters <= 1:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=random_state)
        clusters = kmeans.fit_predict(umap_coords)

    # Convert to 1-indexed
    clusters = clusters + 1

    return clusters, umap_coords


def _run_decontx_single(
    X: Union[np.ndarray, csr_matrix],
    z_labels: np.ndarray,
    X_background: Optional[Union[np.ndarray, csr_matrix]],
    max_iter: int,
    convergence_threshold: float,
    iter_loglik: int,
    delta: Tuple[float, float],
    estimate_delta: bool,
    random_state: int,
    verbose: bool,
    log
) -> Dict:
    """Run DecontX on a single batch."""

    log(f".... Estimating contamination")

    # Initialize model
    model = DecontXModel(
        max_iter=max_iter,
        convergence_threshold=convergence_threshold,
        delta=delta,
        estimate_delta=estimate_delta,
        iter_loglik=iter_loglik,
        random_state=random_state,
        verbose=verbose
    )

    # Fit model
    result = model.fit_transform(X, z_labels, X_background)

    log(f"...... Median contamination: {np.median(result['contamination']):.2%}")
    log(f"...... Range: {np.min(result['contamination']):.2%} - {np.max(result['contamination']):.2%}")

    return result


def _process_batches(
    adata: AnnData,
    z_labels: np.ndarray,
    batch_key: str,
    background_dict: Dict,
    max_iter: int,
    convergence_threshold: float,
    iter_loglik: int,
    delta: Tuple[float, float],
    estimate_delta: bool,
    random_state: int,
    verbose: bool,
    log
) -> Dict:
    """Process multiple batches separately."""

    batches = adata.obs[batch_key].unique()
    log(f".. Processing {len(batches)} batches")

    # Initialize result arrays
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    decontx_counts = np.zeros((n_cells, n_genes))
    contamination = np.zeros(n_cells)
    batch_results = {}

    for batch in batches:
        log(f".. Analyzing batch '{batch}'")

        batch_mask = adata.obs[batch_key] == batch
        batch_idx = np.where(batch_mask)[0]

        # Get batch data
        if issparse(adata.X):
            X_batch = adata.X[batch_mask].tocsr()
        else:
            X_batch = adata.X[batch_mask]

        z_batch = z_labels[batch_mask]
        X_bg = background_dict.get(batch, None)

        # Run DecontX
        result = _run_decontx_single(
            X_batch, z_batch, X_bg,
            max_iter, convergence_threshold, iter_loglik,
            delta, estimate_delta, random_state, verbose, log
        )

        # Store results
        decontx_counts[batch_idx] = result['decontaminated_counts']
        contamination[batch_idx] = result['contamination']
        batch_results[batch] = result

    return {
        'decontaminated_counts': decontx_counts,
        'contamination': contamination,
        'batch_results': batch_results
    }


def _store_results(
    adata: AnnData,
    results: Dict,
    z_labels: np.ndarray,
    umap_coords: Optional[np.ndarray]
):
    """Store DecontX results in AnnData object."""

    # Store decontaminated counts
    adata.layers['decontX_counts'] = results['decontaminated_counts']

    # Store contamination estimates
    adata.obs['decontX_contamination'] = results['contamination']

    # Store cluster labels
    adata.obs['decontX_clusters'] = pd.Categorical(z_labels)

    # Store UMAP if generated
    if umap_coords is not None:
        adata.obsm['X_decontX_umap'] = umap_coords

    # Store parameters and estimates in uns
    if 'decontX' not in adata.uns:
        adata.uns['decontX'] = {}

    adata.uns['decontX']['parameters'] = {
        'delta': results.get('delta', None),
        'theta': results.get('theta', None),
        'phi': results.get('phi', None),
        'eta': results.get('eta', None),
        'log_likelihood': results.get('log_likelihood', [])
    }

    if 'batch_results' in results:
        adata.uns['decontX']['batch_results'] = results['batch_results']


def simulate_contamination(
    n_cells: int = 300,
    n_genes: int = 100,
    n_clusters: int = 3,
    n_range: Tuple[int, int] = (500, 1000),
    beta: float = 0.1,
    delta: Union[float, Tuple[float, float]] = (1.0, 10.0),
    num_markers: int = 3,
    random_state: int = 12345,
) -> Dict:
    """
    Simulate contaminated count matrix for testing DecontX.

    Full implementation of R's simulateContamination function.
    """
    np.random.seed(random_state)

    # Generate contamination proportions
    if isinstance(delta, (int, float)):
        delta = (delta, delta)
    contamination_props = np.random.beta(delta[0], delta[1], size=n_cells)

    # Assign cells to clusters
    z = np.random.choice(n_clusters, size=n_cells, replace=True) + 1

    # Generate total counts per cell
    n_counts = np.random.randint(n_range[0], n_range[1], size=n_cells)

    # Split into native and contamination counts
    contam_counts = np.random.binomial(n_counts, contamination_props)
    native_counts = n_counts - contam_counts

    # Generate expression distributions (phi) using Dirichlet
    from scipy.stats import dirichlet
    phi = dirichlet.rvs([beta] * n_genes, size=n_clusters)

    # Add marker genes
    markers_per_cluster = []
    available_genes = list(range(n_genes))

    for k in range(n_clusters):
        if len(available_genes) >= num_markers:
            markers = np.random.choice(available_genes, num_markers, replace=False)
            markers_per_cluster.append(markers)

            # Set high expression for markers
            phi[k, markers] = np.max(phi[k]) * 2

            # Set zero expression in other clusters
            for other_k in range(n_clusters):
                if other_k != k:
                    phi[other_k, markers] = 1e-10

            # Remove used markers
            available_genes = [g for g in available_genes if g not in markers]

    # Normalize phi
    phi = phi / phi.sum(axis=1, keepdims=True)

    # Generate native expression matrix
    native_matrix = np.zeros((n_cells, n_genes))
    for i in range(n_cells):
        cluster = z[i] - 1
        native_matrix[i] = np.random.multinomial(native_counts[i], phi[cluster])

    # Generate contamination distributions (eta)
    eta = np.zeros((n_clusters, n_genes))
    for k in range(n_clusters):
        other_clusters = [j for j in range(n_clusters) if j != k]
        if other_clusters:
            eta[k] = phi[other_clusters].mean(axis=0)
        else:
            eta[k] = 1.0 / n_genes

    # Generate contamination matrix
    contam_matrix = np.zeros((n_cells, n_genes))
    for i in range(n_cells):
        cluster = z[i] - 1
        contam_matrix[i] = np.random.multinomial(contam_counts[i], eta[cluster])

    # Combine to get observed counts
    observed_matrix = native_matrix + contam_matrix

    return {
        'native_counts': native_matrix,
        'observed_counts': observed_matrix,
        'contamination_counts': contam_matrix,
        'contamination': contamination_props,
        'z': z,
        'phi': phi,
        'eta': eta,
        'markers': markers_per_cluster,
        'n_counts': n_counts
    }