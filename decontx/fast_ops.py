"""
Fast operations for DecontX using Numba JIT compilation.
Equivalent to the Rcpp functions in the R version.
"""

import numpy as np
from numba import jit, prange
from typing import Tuple


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def decontx_em_exact(
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        n_genes: int,
        counts_colsums: np.ndarray,
        theta: np.ndarray,
        estimate_eta: bool,
        eta: np.ndarray,
        phi: np.ndarray,
        z: np.ndarray,
        estimate_delta: bool,
        delta: np.ndarray,
        chunk_bounds: np.ndarray,
        native_sums: np.ndarray,
        phi_buf: np.ndarray,
        phi_acc: np.ndarray,
        pseudocount: float = 1e-20
):
    """
    One fused sparse EM step over CSR nonzeros.

    A single parallel pass over cell chunks computes the E-step, the per-cell
    native sums and per-chunk phi accumulators (no per-nonzero array is stored).
    theta, phi, eta and delta are updated in place; workspace arrays
    (native_sums, phi_buf, phi_acc) are supplied by the caller to avoid
    per-iteration allocation. Returns max |theta_new - theta_old|.
    """
    n_cells = theta.shape[0]
    n_clusters = phi.shape[0]
    n_chunks = chunk_bounds.shape[0] - 1

    # Fused E-step: per-cell native sums + per-chunk phi accumulation
    for c in prange(n_chunks):
        for k in range(n_clusters):
            for g in range(n_genes):
                phi_buf[c, k, g] = 0.0
        for j in range(chunk_bounds[c], chunk_bounds[c + 1]):
            k = z[j] - 1
            theta_j = theta[j]
            one_minus_theta = 1.0 - theta_j
            s = 0.0
            for idx in range(indptr[j], indptr[j + 1]):
                g = indices[idx]
                p_native = theta_j * phi[k, g]
                p_contam = one_minus_theta * eta[k, g]
                nc = data[idx] * (p_native + pseudocount) / (p_native + p_contam + pseudocount)
                s += nc
                phi_buf[c, k, g] += nc
            native_sums[j] = s

    # Update delta (Beta moment matching on native proportions)
    if estimate_delta:
        mean_prop = 0.0
        for j in range(n_cells):
            mean_prop += native_sums[j] / (counts_colsums[j] + pseudocount)
        mean_prop /= n_cells

        var_prop = 0.0
        for j in range(n_cells):
            d = native_sums[j] / (counts_colsums[j] + pseudocount) - mean_prop
            var_prop += d * d
        var_prop /= n_cells

        if var_prop > 0.0 and var_prop < mean_prop * (1.0 - mean_prop):
            precision = mean_prop * (1.0 - mean_prop) / var_prop - 1.0
            delta[0] = max(0.1, min(1000.0, mean_prop * precision))
            delta[1] = max(0.1, min(1000.0, (1.0 - mean_prop) * precision))

    # Update theta
    theta_change = 0.0
    for j in range(n_cells):
        t = (native_sums[j] + delta[0] - 1.0) / (counts_colsums[j] + delta[0] + delta[1] - 2.0)
        t = max(pseudocount, min(1.0 - pseudocount, t))
        theta_change = max(theta_change, abs(t - theta[j]))
        theta[j] = t

    # Reduce chunk accumulators and normalize phi
    for k in prange(n_clusters):
        total = pseudocount * n_genes
        for g in range(n_genes):
            a = 0.0
            for c in range(n_chunks):
                a += phi_buf[c, k, g]
            phi_acc[k, g] = a
            total += a
        for g in range(n_genes):
            phi[k, g] = (phi_acc[k, g] + pseudocount) / total

    # Update eta: native expression from OTHER clusters
    if estimate_eta:
        global_acc = np.zeros(n_genes, dtype=np.float64)
        global_total = 0.0
        for k in range(n_clusters):
            for g in range(n_genes):
                global_acc[g] += phi_acc[k, g]
        for g in range(n_genes):
            global_total += global_acc[g]

        for k in range(n_clusters):
            phi_total = 0.0
            for g in range(n_genes):
                phi_total += phi_acc[k, g]
            total = pseudocount * n_genes + (global_total - phi_total)
            for g in range(n_genes):
                eta[k, g] = (global_acc[g] - phi_acc[k, g] + pseudocount) / total

    return theta_change

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

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
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

    for j in prange(n_cells):
        cluster_idx = z[j] - 1
        theta_j = theta[j]
        one_minus_theta = 1.0 - theta_j
        cell_ll = 0.0
        for idx in range(indptr[j], indptr[j + 1]):
            g = indices[idx]
            count = data[idx]
            mixture = theta_j * phi[cluster_idx, g] + one_minus_theta * eta[cluster_idx, g]
            cell_ll += count * np.log(mixture + pseudocount)
        log_likelihood += cell_ll

    return log_likelihood
