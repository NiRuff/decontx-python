"""
DecontX Bayesian mixture model implementation.
Complete variational inference following Yang et al. (2020).
"""

import numpy as np
from numba import get_num_threads
from scipy.stats import beta
from scipy.sparse import issparse, csr_matrix
from typing import Tuple, Optional, Dict
import warnings


# Import the fast operations
from .fast_ops import (
    decontx_em_exact,
    decontx_initialize_exact,
    decontx_log_likelihood_exact,
    calculate_native_matrix_fast
)


class DecontXModel:
    """
    Fixed DecontX model - correct interpretation of theta and contamination.
    """

    def __init__(self, **kwargs):
        self.max_iter = kwargs.get('max_iter', 500)
        self.convergence_threshold = kwargs.get('convergence', 0.001)
        self.delta = np.array(kwargs.get('delta', [10.0, 10.0]))
        self.estimate_delta = kwargs.get('estimate_delta', True)
        self.iter_loglik = kwargs.get('iter_loglik', 10)
        self.seed = kwargs.get('seed', 12345)
        self.verbose = kwargs.get('verbose', True)

    def fit_transform(self, X, z, X_background=None):
        """
        Sparse EM: keeps X as CSR throughout, no dense n_cells×n_genes allocation.
        """
        np.random.seed(self.seed)

        # Ensure CSR — never densify
        if not issparse(X):
            X = csr_matrix(X)
        if X.format != 'csr':
            X = X.tocsr()
        X = X.astype(np.float64, copy=False)

        z = np.ascontiguousarray(z, dtype=np.int32)
        n_cells, n_genes = X.shape
        n_clusters = int(z.max())  # kernels index with z - 1

        # Pull out CSR arrays for numba
        indptr = X.indptr
        indices = X.indices
        data = np.ascontiguousarray(X.data, dtype=np.float64)

        # Initialize theta from Beta distribution
        theta = beta.rvs(self.delta[0], self.delta[1], size=n_cells, random_state=self.seed)
        theta = np.ascontiguousarray(theta, dtype=np.float64)

        # Initialize phi and eta from CSR nonzeros
        phi, eta = decontx_initialize_exact(indptr, indices, data, n_cells, n_genes, theta, z, 1e-20)

        # Handle background if provided
        if X_background is not None:
            if issparse(X_background):
                bg_total = np.asarray(X_background.sum(axis=0)).ravel()
            else:
                bg_total = X_background.sum(axis=0)
            bg_sum = bg_total.sum()
            if bg_sum > 0:
                eta_bg = (bg_total + 1e-20) / (bg_sum + n_genes * 1e-20)
                eta = np.tile(eta_bg, (n_clusters, 1))

        # Row sums from sparse (avoids dense sum)
        counts_colsums = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)

        # Balance cells across chunks by nonzeros for the parallel E-step
        max_chunks_mem = max(1, int(256e6 // (8 * n_clusters * n_genes)))
        n_chunks = max(1, min(get_num_threads(), n_cells, max_chunks_mem))
        targets = np.linspace(0, indptr[-1], n_chunks + 1)
        chunk_bounds = np.searchsorted(indptr, targets, side='left').astype(np.int64)
        chunk_bounds[0], chunk_bounds[-1] = 0, n_cells
        chunk_bounds = np.maximum.accumulate(chunk_bounds)

        # Workspace reused across iterations
        native_sums = np.empty(n_cells, dtype=np.float64)
        phi_buf = np.empty((n_chunks, n_clusters, n_genes), dtype=np.float64)
        phi_acc = np.empty((n_clusters, n_genes), dtype=np.float64)
        delta = np.ascontiguousarray(self.delta, dtype=np.float64)

        # EM algorithm
        log_likelihood_history = []

        for iteration in range(self.max_iter):
            # EM step — updates theta, phi, eta, delta in place
            theta_change = decontx_em_exact(
                indptr, indices, data, n_genes,
                counts_colsums, theta,
                X_background is None,
                eta, phi, z,
                self.estimate_delta,
                delta,
                chunk_bounds, native_sums, phi_buf, phi_acc,
                1e-20
            )

            # Convergence check is O(n_cells) and done every iteration
            converged = theta_change < self.convergence_threshold

            # Log-likelihood every iter_loglik iterations and at convergence (as in R)
            if converged or iteration % self.iter_loglik == 0:
                log_lik = decontx_log_likelihood_exact(
                    indptr, indices, data, n_cells, theta, eta, phi, z, 1e-20
                )
                log_likelihood_history.append(log_lik)
                if self.verbose:
                    print(f"Iter {iteration}: LL={log_lik:.1f}, change={theta_change:.4f}, "
                          f"mean_contam={(1-theta.mean()):.3f}", flush=True)

            if converged:
                if self.verbose:
                    print(f"Converged at iteration {iteration}", flush=True)
                break

        self.delta = delta

        # Final native counts — nc_data aligned to CSR nonzeros
        nc_data = calculate_native_matrix_fast(indptr, indices, data, n_cells, theta, phi, eta, z)
        nc_data_rounded = np.round(nc_data).astype(np.int32)

        # Assemble sparse output — same sparsity pattern as input
        decontaminated = csr_matrix(
            (nc_data_rounded, X.indices.copy(), X.indptr.copy()),
            shape=(n_cells, n_genes)
        )

        contamination = 1.0 - theta

        return {
            'contamination': contamination,
            'decontaminated_counts': decontaminated,
            'theta': theta,
            'phi': phi,
            'eta': eta,
            'delta': self.delta,
            'z': z,
            'log_likelihood': log_likelihood_history
        }