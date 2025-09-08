"""
Fast operations for DecontX using Numba JIT compilation.
Equivalent to the Rcpp functions in the R version.
"""

import numpy as np
from numba import jit, prange
from scipy.sparse import csr_matrix
from typing import Tuple


# Force compilation with dummy data to avoid first-run compilation overhead
def _precompile_functions():
    """Precompile Numba functions to avoid runtime compilation."""
    # Set number of threads for parallel execution
    from numba import set_num_threads
    import os

    # Use all available cores
    n_cores = os.cpu_count()
    set_num_threads(n_cores)

    # Rest of the precompilation code remains the same
    dummy_counts = np.random.rand(10, 20).astype(np.float64)
    dummy_z = np.array([1, 1, 2, 2, 3, 3, 1, 2, 3, 1], dtype=np.int32)
    dummy_theta = np.random.rand(10).astype(np.float64)
    dummy_phi = np.random.rand(3, 20).astype(np.float64)
    dummy_eta = np.random.rand(3, 20).astype(np.float64)
    dummy_delta = np.array([10.0, 10.0])
    dummy_colsums = dummy_counts.sum(axis=1)

    # Precompile main functions
    decontx_em_exact(dummy_counts, dummy_colsums, dummy_theta, True,
                     dummy_eta, dummy_phi, dummy_z, True, dummy_delta, 1e-20)
    decontx_initialize_exact(dummy_counts, dummy_theta, dummy_z, 1e-20)
    decontx_log_likelihood_exact(dummy_counts, dummy_theta, dummy_eta,
                                 dummy_phi, dummy_z, 1e-20)
    calculate_native_matrix_fast(dummy_counts, dummy_theta, dummy_phi,
                                 dummy_eta, dummy_z)


@jit(nopython=True, parallel=True)
def col_sum_by_group(X: np.ndarray, groups: np.ndarray, K: int) -> np.ndarray:
    """
    Fast column sum by group for dense matrices.
    Equivalent to R's colSumByGroup.

    Parameters
    ----------
    X : array, shape (n_features, n_samples)
        Input matrix
    groups : array, shape (n_samples,)
        Group assignments (1-indexed)
    K : int
        Number of groups

    Returns
    -------
    array, shape (n_features, K)
        Column sums by group
    """
    n_features, n_samples = X.shape
    result = np.zeros((n_features, K))

    for j in prange(n_samples):
        group = groups[j] - 1  # Convert to 0-indexed
        if 0 <= group < K:
            for i in range(n_features):
                result[i, group] += X[i, j]

    return result


@jit(nopython=True)
def col_sum_by_group_sparse_data(
        data: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        groups: np.ndarray,
        K: int,
        n_features: int
) -> np.ndarray:
    """
    Fast column sum by group for sparse matrices (CSR format).
    Works with the raw sparse matrix arrays.
    """
    result = np.zeros((n_features, K))
    n_samples = len(groups)

    for j in range(n_samples):
        group = groups[j] - 1
        if 0 <= group < K:
            for idx in range(indptr[j], indptr[j + 1]):
                i = indices[idx]
                result[i, group] += data[idx]

    return result


def col_sum_by_group_sparse(X: csr_matrix, groups: np.ndarray, K: int) -> np.ndarray:
    """
    Wrapper for sparse matrix group sums.
    """
    return col_sum_by_group_sparse_data(
        X.data, X.indices, X.indptr, groups, K, X.shape[0]
    )


@jit(nopython=True, parallel=True)
def fast_norm_prop(X: np.ndarray, alpha: float = 1e-10) -> np.ndarray:
    """
    Fast column-wise normalization to proportions.
    Equivalent to R's fastNormProp.
    """
    n_rows, n_cols = X.shape
    result = np.zeros_like(X, dtype=np.float64)

    for j in prange(n_cols):
        col_sum = 0.0
        for i in range(n_rows):
            col_sum += X[i, j] + alpha

        for i in range(n_rows):
            result[i, j] = (X[i, j] + alpha) / col_sum

    return result


@jit(nopython=True, parallel=True)
def fast_norm_prop_log(X: np.ndarray, alpha: float = 1e-10) -> np.ndarray:
    """
    Fast log-transformed normalization.
    Equivalent to R's fastNormPropLog.
    """
    n_rows, n_cols = X.shape
    result = np.zeros_like(X, dtype=np.float64)

    for j in prange(n_cols):
        col_sum = 0.0
        for i in range(n_rows):
            col_sum += X[i, j] + alpha

        for i in range(n_rows):
            result[i, j] = np.log((X[i, j] + alpha) / col_sum + 1e-20)

    return result


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def decontx_em_exact(
        counts,
        counts_colsums,
        theta,
        estimate_eta,
        eta,
        phi,
        z,
        estimate_delta,
        delta,
        pseudocount=1e-20
):
    """
    OPTIMIZED: Fast EM step with parallelized E-step.
    Drop-in replacement - same interface, same results, faster.
    """
    n_cells, n_genes = counts.shape
    n_clusters = phi.shape[0]

    # E-step: PARALLELIZED calculation of expected native counts
    native_counts = np.zeros((n_cells, n_genes), dtype=np.float64)

    # Parallel over cells (independent calculations)
    for j in prange(n_cells):  # <-- KEY CHANGE: prange instead of range
        cluster_idx = z[j] - 1
        theta_j = theta[j]
        one_minus_theta = 1.0 - theta_j

        phi_vec = phi[cluster_idx, :]
        eta_vec = eta[cluster_idx, :]

        # Vectorized calculation for all genes
        p_native = theta_j * phi_vec + pseudocount
        p_contam = one_minus_theta * eta_vec + pseudocount
        total = p_native + p_contam

        # Compute native counts for this cell
        native_counts[j, :] = counts[j, :] * (p_native / total)

    # M-step: Update parameters (remains sequential due to dependencies)

    # Update theta
    native_sums = np.sum(native_counts, axis=1)

    if estimate_delta:
        # Method of moments for delta
        proportions = native_sums / (counts_colsums + pseudocount)
        mean_prop = np.mean(proportions)
        var_prop = np.var(proportions)

        if var_prop > 0 and var_prop < mean_prop * (1 - mean_prop):
            precision = (mean_prop * (1 - mean_prop) / var_prop - 1)
            if precision > 0:
                delta[0] = max(0.1, min(1000.0, mean_prop * precision))
                delta[1] = max(0.1, min(1000.0, (1 - mean_prop) * precision))

    # Vectorized theta update
    theta_new = (native_sums + delta[0] - 1) / (counts_colsums + delta[0] + delta[1] - 2)
    theta[:] = np.maximum(pseudocount, np.minimum(1.0 - pseudocount, theta_new))

    # Update phi
    phi_new = np.zeros_like(phi)
    for k in range(n_clusters):
        mask = (z == k + 1)
        if np.any(mask):
            cluster_native = np.sum(native_counts[mask, :], axis=0)
            total = np.sum(cluster_native) + n_genes * pseudocount
            phi_new[k, :] = (cluster_native + pseudocount) / total

    phi[:] = phi_new

    # Update eta if needed
    if estimate_eta:
        eta_new = np.zeros_like(eta)
        contam_counts = counts - native_counts

        for k in range(n_clusters):
            other_mask = (z != k + 1)  # FROM OTHER clusters
            if np.any(other_mask):
                eta_counts = np.sum(contam_counts[other_mask, :], axis=0)
                eta_total = np.sum(eta_counts) + n_genes * pseudocount

                if eta_total > n_genes * pseudocount:
                    eta_new[k, :] = (eta_counts + pseudocount) / eta_total
                else:
                    eta_new[k, :] = 1.0 / n_genes

        eta[:] = eta_new

    contamination = 1.0 - theta
    return theta, phi, eta, delta, contamination


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def calculate_native_matrix_fast(
        counts: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        eta: np.ndarray,
        z: np.ndarray
) -> np.ndarray:
    """
    OPTIMIZED: Parallel calculation of native counts.
    Drop-in replacement with better parallelization.
    """
    n_cells, n_genes = counts.shape
    native_counts = np.zeros_like(counts, dtype=np.float64)

    # Parallelize over cells (independent calculations)
    for j in prange(n_cells):  # <-- KEY CHANGE: prange for parallelization
        cluster = z[j] - 1
        theta_j = theta[j]
        one_minus_theta = 1.0 - theta_j

        # Vectorized calculation for all genes in this cell
        phi_vec = phi[cluster, :]
        eta_vec = eta[cluster, :]

        p_native = theta_j * phi_vec + 1e-20
        p_contam = one_minus_theta * eta_vec + 1e-20
        p_total = p_native + p_contam

        native_counts[j, :] = counts[j, :] * (p_native / p_total)

    return native_counts


@jit(nopython=True)
def calculate_log_likelihood_fast(
        counts: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        eta: np.ndarray,
        z: np.ndarray
) -> float:
    """
    Fast log-likelihood calculation.
    Equivalent to R's decontXLogLik.
    """
    n_cells, n_genes = counts.shape
    log_lik = 0.0

    for j in range(n_cells):
        cluster = z[j] - 1
        for g in range(n_genes):
            if counts[j, g] > 0:
                mixture = theta[j] * phi[cluster, g] + (1 - theta[j]) * eta[cluster, g]
                log_lik += counts[j, g] * np.log(mixture + 1e-20)

    return log_lik


@jit(nopython=True)
def nonzero(X: np.ndarray) -> np.ndarray:
    """
    Get row and column indices of non-zero elements.
    Equivalent to R's nonzero function.
    """
    n_rows, n_cols = X.shape
    indices = []

    for i in range(n_rows):
        for j in range(n_cols):
            if X[i, j] != 0:
                indices.append([i, j])

    if len(indices) > 0:
        return np.array(indices)
    else:
        return np.zeros((0, 2), dtype=np.int64)


@jit(nopython=True, parallel=True, cache=True)
def decontx_initialize_exact(
        counts: np.ndarray,
        theta: np.ndarray,
        z: np.ndarray,
        pseudocount: float = 1e-20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OPTIMIZED: Faster initialization with parallel weighted count computation.
    Drop-in replacement with same results.
    """
    n_cells, n_genes = counts.shape
    n_clusters = len(np.unique(z))

    # Pre-allocate
    phi = np.zeros((n_clusters, n_genes))
    eta = np.zeros((n_clusters, n_genes))

    # PARALLEL computation of weighted counts
    weighted_native = np.zeros_like(counts)
    weighted_contam = np.zeros_like(counts)

    for j in prange(n_cells):  # <-- Parallelized
        weighted_native[j, :] = counts[j, :] * theta[j]
        weighted_contam[j, :] = counts[j, :] * (1.0 - theta[j])

    # Calculate phi and eta for each cluster (sequential - has dependencies)
    for k in range(n_clusters):
        cluster_mask = (z == k + 1)

        # Phi: native expression for cluster k
        phi_counts = np.sum(weighted_native[cluster_mask, :], axis=0)
        phi_total = np.sum(phi_counts) + n_genes * pseudocount
        phi[k, :] = (phi_counts + pseudocount) / phi_total

        # Eta: contamination FROM other clusters
        other_mask = ~cluster_mask
        eta_counts = np.sum(weighted_contam[other_mask, :], axis=0)
        eta_total = np.sum(eta_counts) + n_genes * pseudocount

        if eta_total > n_genes * pseudocount:
            eta[k, :] = (eta_counts + pseudocount) / eta_total
        else:
            eta[k, :] = 1.0 / n_genes

    return phi, eta


@jit(nopython=True, cache=True)
def decontx_log_likelihood_exact(
        counts: np.ndarray,
        theta: np.ndarray,
        eta: np.ndarray,
        phi: np.ndarray,
        z: np.ndarray,
        pseudocount: float = 1e-20
) -> float:
    """
    Sequential log-likelihood calculation to avoid compilation issues.
    Still fast due to vectorization within each cell.
    """
    n_cells, n_genes = counts.shape
    log_likelihood = 0.0

    for j in range(n_cells):
        cluster_idx = z[j] - 1

        # Vectorized calculation for all genes in this cell
        mixture_probs = (theta[j] * phi[cluster_idx, :] +
                         (1.0 - theta[j]) * eta[cluster_idx, :] +
                         pseudocount)

        # Only sum where counts > 0
        mask = counts[j, :] > 0
        if np.any(mask):
            log_likelihood += np.sum(counts[j, mask] * np.log(mixture_probs[mask]))

    return log_likelihood

@jit(nopython=True, parallel=True)
def fast_norm_prop_sqrt(X: np.ndarray, alpha: float = 1e-10) -> np.ndarray:
    """
    Fast column-wise normalization to proportions with square root transformation.
    Equivalent to R's fastNormPropSqrt.
    """
    n_rows, n_cols = X.shape
    result = np.zeros_like(X, dtype=np.float64)

    for j in prange(n_cols):
        # First pass: compute sum of square roots
        col_sum = 0.0
        for i in range(n_rows):
            col_sum += np.sqrt(X[i, j] + alpha)

        # Second pass: normalize
        for i in range(n_rows):
            result[i, j] = np.sqrt(X[i, j] + alpha) / col_sum

    return result


@jit(nopython=True)
def col_sum_by_group_change_sparse(
        data: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        px: np.ndarray,
        group: np.ndarray,
        pgroup: np.ndarray,
        K: int,
        n_features: int
) -> np.ndarray:
    """
    Column sum by group with change tracking for sparse matrices.
    Equivalent to R's colSumByGroupChangeSparse.

    This tracks changes when reassigning cells from one group to another.
    px: index of cell being reassigned
    pgroup: previous group assignment
    """
    result = np.zeros((n_features, K))
    n_samples = len(group)

    for j in range(n_samples):
        current_group = group[j] - 1  # Convert to 0-indexed

        # Handle the cell being changed
        if j == px:
            # Remove contribution from previous group and add to new group
            prev_group = pgroup - 1  # Convert to 0-indexed

            # Add to new group
            if 0 <= current_group < K:
                for idx in range(indptr[j], indptr[j + 1]):
                    i = indices[idx]
                    result[i, current_group] += data[idx]

            # Subtract from previous group (handled implicitly by not adding)
            continue

        # Regular processing for other cells
        if 0 <= current_group < K:
            for idx in range(indptr[j], indptr[j + 1]):
                i = indices[idx]
                result[i, current_group] += data[idx]

    return result


def col_sum_by_group_change_sparse_wrapper(
        X_sparse,
        px: int,
        group: np.ndarray,
        pgroup: int,
        K: int
) -> np.ndarray:
    """Wrapper for sparse matrix group sums with change tracking."""
    X_csr = X_sparse.tocsr()
    return col_sum_by_group_change_sparse(
        X_csr.data, X_csr.indices, X_csr.indptr, px, group, pgroup, K, X_csr.shape[0]
    )


@jit(nopython=True)
def row_sum_by_group_sparse_data(
        data: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        group: np.ndarray,
        L: int,
        n_features: int
) -> np.ndarray:
    """
    Row sum by group for sparse matrices.
    Equivalent to R's rowSumByGroupSparse.
    """
    result = np.zeros((n_features, L))
    n_samples = len(group)

    for j in range(n_samples):
        group_idx = group[j] - 1  # Convert to 0-indexed
        if 0 <= group_idx < L:
            for idx in range(indptr[j], indptr[j + 1]):
                feature_idx = indices[idx]
                result[feature_idx, group_idx] += data[idx]

    return result


def row_sum_by_group_sparse(X_sparse, group: np.ndarray, L: int) -> np.ndarray:
    """Wrapper for sparse matrix row sums by group."""
    X_csr = X_sparse.tocsr()
    return row_sum_by_group_sparse_data(
        X_csr.data, X_csr.indices, X_csr.indptr, group, L, X_csr.shape[0]
    )


@jit(nopython=True)
def row_sum_by_group_change_sparse_data(
        data: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        px: int,
        group: np.ndarray,
        pgroup: int,
        L: int,
        n_features: int
) -> np.ndarray:
    """
    Row sum by group with change tracking for sparse matrices.
    Equivalent to R's rowSumByGroupChangeSparse.
    """
    result = np.zeros((n_features, L))
    n_samples = len(group)

    for j in range(n_samples):
        current_group = group[j] - 1  # Convert to 0-indexed

        # Handle the cell being changed
        if j == px:
            # Add to new group only
            if 0 <= current_group < L:
                for idx in range(indptr[j], indptr[j + 1]):
                    feature_idx = indices[idx]
                    result[feature_idx, current_group] += data[idx]
            continue

        # Regular processing for other cells
        if 0 <= current_group < L:
            for idx in range(indptr[j], indptr[j + 1]):
                feature_idx = indices[idx]
                result[feature_idx, current_group] += data[idx]

    return result


def row_sum_by_group_change_sparse(
        X_sparse,
        px: int,
        group: np.ndarray,
        pgroup: int,
        L: int
) -> np.ndarray:
    """Wrapper for sparse matrix row sums by group with change tracking."""
    X_csr = X_sparse.tocsr()
    return row_sum_by_group_change_sparse_data(
        X_csr.data, X_csr.indices, X_csr.indptr, px, group, pgroup, L, X_csr.shape[0]
    )


@jit(nopython=True)
def retrieve_feature_index_fast(features, search_space, exact_match=True):
    """Equivalent to R's retrieveFeatureIndex core logic"""
    n_features = len(features)
    n_search = len(search_space)
    indices = np.full(n_features, -1, dtype=np.int64)

    if exact_match:
        for i in range(n_features):
            feature = features[i]
            for j in range(n_search):
                if search_space[j] == feature:
                    indices[i] = j
                    break
    else:
        # Partial matching - simplified version
        for i in range(n_features):
            feature = features[i]
            matches = []
            for j in range(n_search):
                if feature in search_space[j]:
                    matches.append(j)

            if len(matches) == 1:
                indices[i] = matches[0]
            elif len(matches) > 1:
                indices[i] = matches[0]  # Take first match like R

    return indices


@jit(nopython=True)
def calculate_log_messages_time():
    """For timestamp functionality matching R"""
    # This is simplified - in practice you'd want to use datetime
    # Numba doesn't support datetime directly
    return 0.0  # Placeholder for timestamp


# Call precompilation when module loads
_precompile_functions()