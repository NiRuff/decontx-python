"""
Fast operations for DecontX using Numba JIT compilation.
Equivalent to the Rcpp functions in the R version.
"""

import numpy as np
from numba import jit, prange
from scipy.sparse import csr_matrix
from typing import Tuple


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


@jit(nopython=True)
def decontx_em_step(
        counts: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        eta: np.ndarray,
        z: np.ndarray,
        delta: np.ndarray,
        estimate_delta: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast EM iteration - main computational bottleneck.
    Equivalent to R's decontXEM function.
    """
    n_cells, n_genes = counts.shape
    n_clusters = phi.shape[0]

    # E-step: Calculate responsibilities
    log_pr = np.zeros((n_cells, n_genes))
    log_pc = np.zeros((n_cells, n_genes))

    for j in range(n_cells):
        cluster = z[j] - 1  # Convert to 0-indexed
        for g in range(n_genes):
            log_pr[j, g] = np.log(phi[cluster, g] + 1e-20) + np.log(theta[j] + 1e-20)
            log_pc[j, g] = np.log(eta[cluster, g] + 1e-20) + np.log(1 - theta[j] + 1e-20)

    # Calculate posterior probabilities
    pr = np.zeros_like(log_pr)
    for j in range(n_cells):
        for g in range(n_genes):
            max_val = max(log_pr[j, g], log_pc[j, g])
            pr[j, g] = np.exp(log_pr[j, g] - max_val) / (
                    np.exp(log_pr[j, g] - max_val) + np.exp(log_pc[j, g] - max_val)
            )

    # M-step: Update parameters
    # Update theta
    new_theta = np.zeros(n_cells)
    for j in range(n_cells):
        native_sum = 0.0
        total_sum = 0.0
        for g in range(n_genes):
            native_sum += counts[j, g] * pr[j, g]
            total_sum += counts[j, g]

        new_theta[j] = (native_sum + delta[0]) / (total_sum + delta[0] + delta[1])

    # Update phi
    new_phi = np.zeros_like(phi)
    for k in range(n_clusters):
        for g in range(n_genes):
            numerator = 0.0
            for j in range(n_cells):
                if z[j] - 1 == k:
                    numerator += counts[j, g] * pr[j, g]
            new_phi[k, g] = numerator + 1e-20

        # Normalize
        phi_sum = np.sum(new_phi[k, :])
        if phi_sum > 0:
            new_phi[k, :] /= phi_sum

    # Update eta
    new_eta = np.zeros_like(eta)
    for k in range(n_clusters):
        for g in range(n_genes):
            numerator = 0.0
            for j in range(n_cells):
                if z[j] - 1 != k:  # From other clusters
                    numerator += counts[j, g] * (1 - pr[j, g])
            new_eta[k, g] = numerator + 1e-20

        # Normalize
        eta_sum = np.sum(new_eta[k, :])
        if eta_sum > 0:
            new_eta[k, :] /= eta_sum

    # Update delta if requested (simplified - proper Dirichlet fitting needed)
    new_delta = delta.copy()
    if estimate_delta:
        mean_theta = np.mean(new_theta)
        var_theta = np.var(new_theta)
        if var_theta > 0 and var_theta < mean_theta * (1 - mean_theta):
            common = mean_theta * (1 - mean_theta) / var_theta - 1
            new_delta[0] = mean_theta * common
            new_delta[1] = (1 - mean_theta) * common

    return new_theta, new_phi, new_eta, new_delta


@jit(nopython=True, parallel=True)
def calculate_native_matrix_fast(
        counts: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        eta: np.ndarray,
        z: np.ndarray
) -> np.ndarray:
    """
    Fast calculation of native (decontaminated) counts.
    Equivalent to R's calculateNativeMatrix.
    """
    n_cells, n_genes = counts.shape
    native_counts = np.zeros_like(counts, dtype=np.float64)

    for j in prange(n_cells):
        cluster = z[j] - 1
        for g in range(n_genes):
            log_native = np.log(phi[cluster, g] + 1e-20) + np.log(theta[j] + 1e-20)
            log_contam = np.log(eta[cluster, g] + 1e-20) + np.log(1 - theta[j] + 1e-20)

            # Stable computation of p_native
            max_val = max(log_native, log_contam)
            p_native = np.exp(log_native - max_val) / (
                    np.exp(log_native - max_val) + np.exp(log_contam - max_val)
            )

            native_counts[j, g] = counts[j, g] * p_native

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