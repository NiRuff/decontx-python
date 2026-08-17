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
    from scipy.sparse import csr_matrix as _csr
    dummy_dense = np.random.rand(10, 20).astype(np.float64)
    dummy_dense[dummy_dense < 0.5] = 0.0  # make sparse
    dummy_csr = _csr(dummy_dense)
    dummy_indptr = dummy_csr.indptr.astype(np.int64)
    dummy_indices = dummy_csr.indices.astype(np.int64)
    dummy_data = dummy_csr.data.astype(np.float64)
    dummy_z = np.array([1, 1, 2, 2, 3, 3, 1, 2, 3, 1], dtype=np.int32)
    dummy_theta = np.random.rand(10).astype(np.float64)
    dummy_phi = np.random.rand(3, 20).astype(np.float64)
    dummy_eta = np.random.rand(3, 20).astype(np.float64)
    dummy_delta = np.array([10.0, 10.0])
    dummy_colsums = np.asarray(dummy_csr.sum(axis=1)).ravel().astype(np.float64)

    # Precompile main functions with CSR signatures
    decontx_initialize_exact(dummy_indptr, dummy_indices, dummy_data,
                             10, 20, dummy_theta, dummy_z, 1e-20)
    decontx_em_exact(dummy_indptr, dummy_indices, dummy_data, 10, 20,
                     dummy_colsums, dummy_theta, True,
                     dummy_eta, dummy_phi, dummy_z, True, dummy_delta, 1e-20)
    decontx_log_likelihood_exact(dummy_indptr, dummy_indices, dummy_data, 10,
                                 dummy_theta, dummy_eta, dummy_phi, dummy_z, 1e-20)
    calculate_native_matrix_fast(dummy_indptr, dummy_indices, dummy_data, 10,
                                 dummy_theta, dummy_phi, dummy_eta, dummy_z)


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


@jit(nopython=True, fastmath=True, cache=True)
def decontx_em_exact(
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        n_cells: int,
        n_genes: int,
        counts_colsums: np.ndarray,
        theta: np.ndarray,
        estimate_eta: bool,
        eta: np.ndarray,
        phi: np.ndarray,
        z: np.ndarray,
        estimate_delta: bool,
        delta: np.ndarray,
        pseudocount: float = 1e-20
):
    """
    Sparse EM step operating only on CSR nonzeros.

    E-step complexity: O(nnz) instead of O(n_cells * n_genes).
    M-step uses serial scatter-add into phi_acc to avoid races.
    """
    n_clusters = phi.shape[0]
    nnz = len(data)

    # E-step: compute nc_data aligned to CSR nonzero positions
    nc_data = np.zeros(nnz, dtype=np.float64)

    for j in prange(n_cells):
        cluster_idx = z[j] - 1
        theta_j = theta[j]
        one_minus_theta = 1.0 - theta_j
        for idx in range(indptr[j], indptr[j + 1]):
            g = indices[idx]
            count = data[idx]
            p_native = theta_j * phi[cluster_idx, g]
            p_contam = one_minus_theta * eta[cluster_idx, g]
            total = p_native + p_contam + pseudocount
            nc_data[idx] = count * (p_native + pseudocount) / total

    # M-step: native row sums (for theta update)
    native_sums = np.zeros(n_cells, dtype=np.float64)
    for j in range(n_cells):
        s = 0.0
        for idx in range(indptr[j], indptr[j + 1]):
            s += nc_data[idx]
        native_sums[j] = s

    # Update delta
    if estimate_delta:
        proportions = native_sums / (counts_colsums + pseudocount)
        mean_prop = 0.0
        for j in range(n_cells):
            mean_prop += proportions[j]
        mean_prop /= n_cells

        var_prop = 0.0
        for j in range(n_cells):
            d = proportions[j] - mean_prop
            var_prop += d * d
        var_prop /= n_cells

        if var_prop > 0.0 and var_prop < mean_prop * (1.0 - mean_prop):
            precision = mean_prop * (1.0 - mean_prop) / var_prop - 1.0
            delta[0] = max(0.1, min(1000.0, mean_prop * precision))
            delta[1] = max(0.1, min(1000.0, (1.0 - mean_prop) * precision))

    # Update theta
    for j in range(n_cells):
        t = (native_sums[j] + delta[0] - 1.0) / (counts_colsums[j] + delta[0] + delta[1] - 2.0)
        theta[j] = max(pseudocount, min(1.0 - pseudocount, t))

    # Update phi: serial scatter-add to avoid race conditions
    phi_acc = np.zeros((n_clusters, n_genes), dtype=np.float64)
    for j in range(n_cells):
        k = z[j] - 1
        for idx in range(indptr[j], indptr[j + 1]):
            g = indices[idx]
            phi_acc[k, g] += nc_data[idx]

    # Normalize phi
    for k in range(n_clusters):
        total = pseudocount * n_genes
        for g in range(n_genes):
            total += phi_acc[k, g]
        for g in range(n_genes):
            phi[k, g] = (phi_acc[k, g] + pseudocount) / total

    # Update eta: native expression from OTHER clusters
    if estimate_eta:
        # global native sum per gene
        global_acc = np.zeros(n_genes, dtype=np.float64)
        for k in range(n_clusters):
            for g in range(n_genes):
                global_acc[g] += phi_acc[k, g]

        for k in range(n_clusters):
            total = pseudocount * n_genes
            for g in range(n_genes):
                other_native = global_acc[g] - phi_acc[k, g]
                total += other_native
            for g in range(n_genes):
                other_native = global_acc[g] - phi_acc[k, g]
                eta[k, g] = (other_native + pseudocount) / total

    contamination = 1.0 - theta
    return theta, phi, eta, delta, contamination


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def calculate_native_matrix_fast(
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        n_cells: int,
        theta: np.ndarray,
        phi: np.ndarray,
        eta: np.ndarray,
        z: np.ndarray
) -> np.ndarray:
    """
    Sparse native count calculation aligned to CSR nonzeros.

    Returns nc_data: float64 array of same length as data, giving native
    count estimates at each CSR position. The caller assembles the sparse
    matrix using the original indptr/indices.

    Each cell j writes only to its own indptr slice — race-free with prange.
    """
    nc_data = np.zeros(len(data), dtype=np.float64)

    for j in prange(n_cells):
        cluster = z[j] - 1
        theta_j = theta[j]
        one_minus_theta = 1.0 - theta_j
        for idx in range(indptr[j], indptr[j + 1]):
            g = indices[idx]
            p_native = theta_j * phi[cluster, g] + 1e-20
            p_contam = one_minus_theta * eta[cluster, g] + 1e-20
            nc_data[idx] = data[idx] * p_native / (p_native + p_contam)

    return nc_data


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


@jit(nopython=True, cache=True)
def decontx_initialize_exact(
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        n_cells: int,
        n_genes: int,
        theta: np.ndarray,
        z: np.ndarray,
        pseudocount: float = 1e-20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize phi and eta from CSR nonzeros.

    phi[k, g] = weighted-native expression for cluster k (theta-weighted CSR accumulation)
    eta[k, g] = weighted-native expression from OTHER clusters (ambient signal source)
    """
    n_clusters = 0
    for j in range(n_cells):
        if z[j] > n_clusters:
            n_clusters = z[j]

    phi_acc = np.zeros((n_clusters, n_genes))
    eta_acc = np.zeros((n_clusters, n_genes))

    # Accumulate theta-weighted native counts per cluster and globally
    global_acc = np.zeros(n_genes)

    for j in range(n_cells):
        k = z[j] - 1
        w = theta[j]
        for idx in range(indptr[j], indptr[j + 1]):
            g = indices[idx]
            wv = data[idx] * w
            phi_acc[k, g] += wv
            global_acc[g] += wv

    # eta[k] = global weighted-native minus cluster k's contribution
    for k in range(n_clusters):
        for g in range(n_genes):
            eta_acc[k, g] = global_acc[g] - phi_acc[k, g]

    # Normalize phi and eta
    phi = np.zeros((n_clusters, n_genes))
    eta = np.zeros((n_clusters, n_genes))

    for k in range(n_clusters):
        phi_total = pseudocount * n_genes
        eta_total = pseudocount * n_genes
        for g in range(n_genes):
            phi_total += phi_acc[k, g]
            eta_total += eta_acc[k, g]

        if phi_total > 0:
            for g in range(n_genes):
                phi[k, g] = (phi_acc[k, g] + pseudocount) / phi_total
        else:
            for g in range(n_genes):
                phi[k, g] = 1.0 / n_genes

        if eta_total > 0:
            for g in range(n_genes):
                eta[k, g] = (eta_acc[k, g] + pseudocount) / eta_total
        else:
            for g in range(n_genes):
                eta[k, g] = 1.0 / n_genes

    return phi, eta


@jit(nopython=True, cache=True, fastmath=True)
def decontx_log_likelihood_exact(
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        n_cells: int,
        theta: np.ndarray,
        eta: np.ndarray,
        phi: np.ndarray,
        z: np.ndarray,
        pseudocount: float = 1e-20
) -> float:
    """
    O(nnz) log-likelihood iterating only over CSR nonzeros.
    """
    log_likelihood = 0.0

    for j in range(n_cells):
        cluster_idx = z[j] - 1
        theta_j = theta[j]
        one_minus_theta = 1.0 - theta_j
        for idx in range(indptr[j], indptr[j + 1]):
            g = indices[idx]
            count = data[idx]
            mixture = theta_j * phi[cluster_idx, g] + one_minus_theta * eta[cluster_idx, g]
            log_likelihood += count * np.log(mixture + pseudocount)

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