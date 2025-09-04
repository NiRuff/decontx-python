import numpy as np
import warnings
from typing import Dict, Tuple, Union



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
    """ simulation matching R's simulateContamination exactly."""

    np.random.seed(random_state)

    # Handle delta parameter like R
    if isinstance(delta, (int, float)):
        delta_params = (delta, delta)
    else:
        delta_params = delta

    # Generate contamination proportions
    contamination_props = np.random.beta(delta_params[0], delta_params[1], size=n_cells)

    # Assign cells to clusters (ensuring all clusters represented)
    z = np.random.choice(n_clusters, size=n_cells, replace=True) + 1

    # Ensure all clusters have at least one cell
    unique_z, counts = np.unique(z, return_counts=True)
    if len(unique_z) < n_clusters:
        # Fill in missing clusters
        for missing_k in range(1, n_clusters + 1):
            if missing_k not in unique_z:
                # Replace a random cell
                random_idx = np.random.randint(n_cells)
                z[random_idx] = missing_k
        warnings.warn(f"Only {len(unique_z)} clusters generated, adjusted to {n_clusters}")

    # Generate total counts per cell
    n_counts = np.random.randint(n_range[0], n_range[1] + 1, size=n_cells)

    # Split into native and contamination counts
    contam_counts = np.random.binomial(n_counts, contamination_props)
    native_counts = n_counts - contam_counts

    # Generate expression distributions using Dirichlet (exact R method)
    phi = np.random.dirichlet([beta] * n_genes, size=n_clusters)

    # Add marker genes (exact R logic)
    if n_clusters * num_markers > n_genes:
        raise ValueError("num_markers * n_clusters cannot exceed n_genes")

    markers_per_cluster = []
    available_genes = list(range(n_genes))

    for k in range(n_clusters):
        if len(available_genes) >= num_markers:
            markers = np.random.choice(available_genes, num_markers, replace=False)
            markers_per_cluster.append(markers)

            # Set high expression in this cluster
            phi[k, markers] = np.max(phi[k]) * 2

            # Zero expression in other clusters
            for other_k in range(n_clusters):
                if other_k != k:
                    phi[other_k, markers] = 1e-10

            # Remove from available
            available_genes = [g for g in available_genes if g not in markers]
        else:
            markers_per_cluster.append([])

    # Renormalize phi
    phi = phi / phi.sum(axis=1, keepdims=True)

    # Generate native expression matrix
    native_matrix = np.zeros((n_cells, n_genes), dtype=int)
    for i in range(n_cells):
        cluster = z[i] - 1
        if native_counts[i] > 0:
            native_matrix[i] = np.random.multinomial(native_counts[i], phi[cluster])

    # Generate contamination distributions (exact R method)
    eta = np.zeros((n_clusters, n_genes))
    for k in range(n_clusters):
        # Sum expression from other clusters
        other_expression = np.zeros(n_genes)
        total_other = 0

        for other_k in range(n_clusters):
            if other_k != k:
                cluster_cells = (z == other_k + 1)
                if np.any(cluster_cells):
                    other_expression += native_matrix[cluster_cells].sum(axis=0)
                    total_other += np.sum(cluster_cells)

        if total_other > 0:
            eta[k] = (other_expression + 1e-20) / (other_expression.sum() + n_genes * 1e-20)
        else:
            eta[k] = np.ones(n_genes) / n_genes

    # Generate contamination matrix
    contam_matrix = np.zeros((n_cells, n_genes), dtype=int)
    for i in range(n_cells):
        cluster = z[i] - 1
        if contam_counts[i] > 0:
            contam_matrix[i] = np.random.multinomial(contam_counts[i], eta[cluster])

    # Combine matrices
    observed_matrix = native_matrix + contam_matrix

    return {
        'nativeCounts': native_matrix,
        'observedCounts': observed_matrix,
        'contaminationCounts': contam_matrix,
        'contamination': contamination_props,
        'z': z,
        'phi': phi,
        'eta': eta,
        'markers': markers_per_cluster,
        'NByC': n_counts,
        'numMarkers': num_markers
    }
