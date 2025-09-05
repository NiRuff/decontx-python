"""
DecontX Bayesian mixture model implementation.
Complete variational inference following Yang et al. (2020).
"""

import numpy as np
from numba import jit, prange
from scipy import optimize
from scipy.special import digamma, polygamma, gammaln
from scipy.stats import beta, dirichlet
from scipy.sparse import issparse, csr_matrix
from typing import Tuple, Optional, Dict
import warnings


# Import the fast operations
from .fast_ops import (
    decontx_em_exact,
    decontx_initialize_exact,
    decontx_log_likelihood_exact,
    col_sum_by_group_change_sparse_wrapper,
    row_sum_by_group_change_sparse,
    fast_norm_prop_sqrt,
    calculate_native_matrix_fast
)

def sparse_to_dense_cached(X):
    """Convert sparse matrix to dense, with caching for repeated calls."""
    if not hasattr(X, '_dense_cache'):
        if issparse(X):
            X._dense_cache = X.toarray()
        else:
            X._dense_cache = X
    return X._dense_cache


class DecontXModel:
    """
    Optimized DecontX model with massive performance improvements.
    """

    def __init__(self, **kwargs):
        self.max_iter = kwargs.get('max_iter', 500)
        self.convergence_threshold = kwargs.get('convergence', 0.001)
        self.delta = np.array(kwargs.get('delta', [10.0, 10.0]))
        self.estimate_delta = kwargs.get('estimate_delta', True)
        self.iter_loglik = kwargs.get('iter_loglik', 10)
        self.seed = kwargs.get('seed', 12345)
        self.verbose = kwargs.get('verbose', True)

    def _r_exact_initialization(self, X, z, theta, pseudocount=1e-20):
        """
        Wrapper for R-exact initialization using fast compiled function.
        """
        # Just call the fast compiled version
        return decontx_initialize_exact(X, theta, z, pseudocount)

    def fit_transform(self, X, z, X_background=None):
        """
        Optimized fit using compiled functions.
        """
        np.random.seed(self.seed)

        # Convert to dense ONCE at the beginning
        if issparse(X):
            X = X.toarray()

        # Ensure proper types
        X = np.ascontiguousarray(X, dtype=np.float64)
        z = np.ascontiguousarray(z, dtype=np.int32)

        n_cells, n_genes = X.shape
        n_clusters = len(np.unique(z))

        # Initialize
        from scipy.stats import beta
        theta = beta.rvs(self.delta[0], self.delta[1], size=n_cells, random_state=self.seed)
        theta = np.ascontiguousarray(theta, dtype=np.float64)

        # Use initialization (now the method exists again)
        phi, eta = self._r_exact_initialization(X, z, theta)

        # Handle background if provided
        if X_background is not None:
            if issparse(X_background):
                X_background = X_background.toarray()
            bg_total = X_background.sum(axis=0)
            bg_sum = bg_total.sum()
            if bg_sum > 0:
                eta_bg = (bg_total + 1e-20) / (bg_sum + n_genes * 1e-20)
                eta = np.tile(eta_bg, (n_clusters, 1))

        # Pre-compute column sums once
        counts_colsums = np.ascontiguousarray(X.sum(axis=1), dtype=np.float64)

        # EM algorithm
        log_likelihood_history = []

        for iteration in range(self.max_iter):
            theta_old = theta.copy()

            # Use fast compiled EM step
            theta, phi, eta, delta_new, contamination = decontx_em_exact(
                counts=X,
                counts_colsums=counts_colsums,
                theta=theta,
                estimate_eta=(X_background is None),
                eta=eta,
                phi=phi,
                z=z,
                estimate_delta=self.estimate_delta,
                delta=self.delta,
                pseudocount=1e-20
            )

            if self.estimate_delta:
                self.delta = delta_new

            # Check convergence periodically
            if iteration % self.iter_loglik == 0:
                log_lik = decontx_log_likelihood_exact(X, theta, eta, phi, z, 1e-20)
                log_likelihood_history.append(log_lik)

                theta_change = np.max(np.abs(theta - theta_old))

                if self.verbose and iteration % 10 == 0:
                    print(f"Iter {iteration}: LL={log_lik:.1f}, change={theta_change:.4f}")

                if theta_change < self.convergence_threshold:
                    if self.verbose:
                        print(f"Converged at iteration {iteration}")
                    break

        # Calculate final decontaminated counts
        decontaminated = calculate_native_matrix_fast(X, theta, phi, eta, z)
        decontaminated = np.round(decontaminated).astype(np.int32)

        return {
            'contamination': 1.0 - theta,
            'decontaminated_counts': decontaminated,
            'theta': theta,
            'phi': phi,
            'eta': eta,
            'delta': self.delta,
            'z': z,
            'log_likelihood': log_likelihood_history
        }

    def _calculate_log_likelihood(self, X, theta, phi, eta, z, pseudocount=1e-20):
        """
        Calculate log-likelihood exactly as R does.
        Fixed numerical stability issues.
        """
        log_lik = 0.0
        n_cells, n_genes = X.shape

        for j in range(n_cells):
            cluster_idx = z[j] - 1  # Convert to 0-indexed
            for g in range(n_genes):
                if issparse(X):
                    count = X[j, g]
                else:
                    count = X[j, g]

                if count > 0:
                    # R's exact mixture probability formula
                    mixture_prob = (theta[j] * phi[cluster_idx, g] +
                                    (1 - theta[j]) * eta[cluster_idx, g] + pseudocount)
                    log_lik += count * np.log(mixture_prob)

        return log_lik

    def _em_step(self, X, z, theta, phi, eta, delta, estimate_delta, pseudocount=1e-20):
        """
        Single EM step using the optimized Numba function.
        This maintains R parity while being orders of magnitude faster.
        """
        # Convert sparse to dense if needed (Numba works better with dense)
        if issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        # Get column sums once (avoid repeated computation)
        counts_colsums = X_dense.sum(axis=1)

        # Use the optimized Numba function from fast_ops.py
        theta, phi, eta, delta, contamination = decontx_em_exact(
            counts=X_dense,
            counts_colsums=counts_colsums,
            theta=theta,
            estimate_eta=True,  # Always estimate eta unless using background
            eta=eta,
            phi=phi,
            z=z,
            estimate_delta=estimate_delta,
            delta=delta,
            pseudocount=pseudocount
        )

        return theta, phi, eta

    def fit_transform(self, X, z, X_background=None):
        """
        Fit using exact R algorithm with proper convergence checking.
        """
        np.random.seed(self.seed)

        if not isinstance(X, np.ndarray):
            if hasattr(X, 'toarray'):
                X_dense = X.toarray()
            else:
                X_dense = np.asarray(X)
        else:
            X_dense = X

        z = np.asarray(z).astype(int)
        n_cells, n_genes = X_dense.shape
        n_clusters = len(np.unique(z))

        from scipy.stats import beta

        # R initializes theta as the proportion of NATIVE counts
        # So if delta = [10, 10], theta ~ Beta(10, 10) which centers around 0.5
        theta = beta.rvs(self.delta[0], self.delta[1], size=n_cells, random_state=self.seed)

        # Initialize phi and eta exactly like R
        phi, eta = self._r_exact_initialization(X_dense, z, theta)

        # Handle background exactly like R (if provided)
        if X_background is not None:
            if not isinstance(X_background, np.ndarray):
                if hasattr(X_background, 'toarray'):
                    X_background = X_background.toarray()
                else:
                    X_background = np.asarray(X_background)

            # Use background to estimate eta (ambient profile)
            bg_total = X_background.sum(axis=0)
            if hasattr(bg_total, 'A1'):
                bg_total = bg_total.A1
            bg_sum = bg_total.sum()

            if bg_sum > 0:
                eta_bg = (bg_total + 1e-20) / (bg_sum + n_genes * 1e-20)
                # Set same eta for all clusters when using background
                for k in range(n_clusters):
                    eta[k, :] = eta_bg

        # EM algorithm with R's convergence checking
        log_likelihood_history = []
        prev_theta = theta.copy()

        # Pre-convert to dense for entire EM (more efficient than converting each iteration)
        if issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        for iteration in range(self.max_iter):
            # Store old theta for convergence check
            theta_old = theta.copy()

            # Use optimized EM step
            theta, phi, eta = self._em_step(
                X_dense, z, theta, phi, eta,
                self.delta, self.estimate_delta
            )

            # Check convergence periodically (not every iteration for speed)
            if iteration % self.iter_loglik == 0 or iteration == self.max_iter - 1:
                # Use optimized log likelihood calculation
                log_lik = decontx_log_likelihood_exact(
                    X_dense, theta, eta, phi, z, 1e-20
                )
                log_likelihood_history.append(log_lik)

                # Convergence check
                theta_change = np.mean(np.abs(theta - theta_old))

                if self.verbose and iteration % 50 == 0:
                    print(f"Iteration {iteration}: LL = {log_lik:.2f}, change = {theta_change:.6f}")

                if theta_change < self.convergence_threshold:
                    if self.verbose:
                        print(f"Converged at iteration {iteration}")
                    break

        # Calculate final contamination
        contamination = 1.0 - theta

        # Use optimized native matrix calculation
        decontaminated = calculate_native_matrix_fast(
            counts=X_dense,
            theta=theta,
            phi=phi,
            eta=eta,
            z=z
        )

        # Round to integers
        decontaminated = np.round(decontaminated).astype(int)

        # Store results
        self.phi_ = phi
        self.eta_ = eta
        self.theta_ = theta
        self.log_likelihood_ = log_likelihood_history

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


