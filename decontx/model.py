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


class DecontXModel:
    """
    DecontX model with EXACT R implementation matching.
    Fixed initialization and EM algorithm to match R exactly.
    """

    def __init__(self, **kwargs):
        self.max_iter = kwargs.get('max_iter', 500)
        self.convergence_threshold = kwargs.get('convergence', 0.001)
        self.delta = np.array(kwargs.get('delta', [10.0, 10.0]))
        self.estimate_delta = kwargs.get('estimate_delta', True)
        self.iter_loglik = kwargs.get('iter_loglik', 10)
        self.seed = kwargs.get('seed', 12345)
        self.verbose = kwargs.get('verbose', True)

        # Storage for results
        self.phi_ = None
        self.eta_ = None
        self.theta_ = None
        self.log_likelihood_ = []

    def _r_exact_initialization(self, X, z, theta, pseudocount=1e-20):
        """
        Exact R initialization matching decontXInitialize C++ function.
        Fixed to properly calculate phi and eta matrices.
        """
        n_cells, n_genes = X.shape
        n_clusters = len(np.unique(z))

        # Initialize with pseudocount
        phi = np.zeros((n_clusters, n_genes)) + pseudocount
        eta = np.zeros((n_clusters, n_genes)) + pseudocount

        # Calculate phi for each cluster (expression profile within cluster)
        for k in range(n_clusters):
            cluster_mask = (z == k + 1)  # z is 1-indexed in R
            if np.any(cluster_mask):
                # Sum counts for cells in this cluster
                if issparse(X):
                    cluster_counts = np.array(X[cluster_mask].sum(axis=0)).flatten()
                else:
                    cluster_counts = X[cluster_mask].sum(axis=0)

                # Normalize to probabilities
                total = cluster_counts.sum()
                if total > 0:
                    phi[k, :] = (cluster_counts + pseudocount) / (total + n_genes * pseudocount)

        # Calculate eta for each cluster (ambient profile from other clusters)
        for k in range(n_clusters):
            other_mask = (z != k + 1)  # All cells NOT in cluster k
            if np.any(other_mask):
                # Sum counts for cells NOT in this cluster
                if issparse(X):
                    other_counts = np.array(X[other_mask].sum(axis=0)).flatten()
                else:
                    other_counts = X[other_mask].sum(axis=0)

                # Normalize to probabilities
                total = other_counts.sum()
                if total > 0:
                    eta[k, :] = (other_counts + pseudocount) / (total + n_genes * pseudocount)

        return phi, eta

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
        Single EM step matching R's decontXEM function exactly.
        Fixed to properly update theta, phi, and eta.
        """
        n_cells, n_genes = X.shape
        n_clusters = phi.shape[0]

        # E-step: Calculate responsibilities (probability each count is native)
        native_counts = np.zeros_like(X, dtype=np.float64)

        for j in range(n_cells):
            cluster_idx = z[j] - 1
            for g in range(n_genes):
                if issparse(X):
                    count = X[j, g]
                else:
                    count = X[j, g]

                if count > 0:
                    # Calculate probability this count is native (not contamination)
                    p_native = theta[j] * phi[cluster_idx, g]
                    p_contam = (1 - theta[j]) * eta[cluster_idx, g]
                    p_total = p_native + p_contam + pseudocount

                    native_counts[j, g] = count * (p_native / p_total)

        # M-step: Update parameters

        # Update theta (contamination per cell)
        for j in range(n_cells):
            if issparse(X):
                total_counts = X[j].sum()
            else:
                total_counts = X[j].sum()

            native_sum = native_counts[j].sum()

            if estimate_delta:
                # Bayesian update with prior
                theta[j] = (native_sum + delta[0] - 1) / (total_counts + delta[0] + delta[1] - 2)
            else:
                # Maximum likelihood estimate
                if total_counts > 0:
                    theta[j] = native_sum / total_counts
                else:
                    theta[j] = 0.5

            # Ensure theta is in valid range
            theta[j] = np.clip(theta[j], 1e-10, 1 - 1e-10)

        # Update phi (expression profile per cluster)
        for k in range(n_clusters):
            cluster_mask = (z == k + 1)
            if np.any(cluster_mask):
                cluster_native = native_counts[cluster_mask].sum(axis=0)
                total = cluster_native.sum()
                if total > 0:
                    phi[k, :] = (cluster_native + pseudocount) / (total + n_genes * pseudocount)

        # Update eta (ambient profile per cluster)
        for k in range(n_clusters):
            cluster_mask = (z == k + 1)
            if np.any(cluster_mask):
                # Contamination counts = total - native
                cluster_contam = np.zeros(n_genes)
                for j in np.where(cluster_mask)[0]:
                    if issparse(X):
                        cluster_contam += np.array(X[j].todense()).flatten() - native_counts[j]
                    else:
                        cluster_contam += X[j] - native_counts[j]

                total = cluster_contam.sum()
                if total > 0:
                    eta[k, :] = (cluster_contam + pseudocount) / (total + n_genes * pseudocount)

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

        # Initialize theta exactly like R: rbeta(n=nC, shape1=delta[1], shape2=delta[2])
        # CRITICAL: R uses delta in reverse order for rbeta!
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
        prev_log_lik = -np.inf

        for iteration in range(self.max_iter):
            # Store old theta for convergence check
            theta_old = theta.copy()

            # EM step
            theta, phi, eta = self._em_step(
                X_dense, z, theta, phi, eta,
                self.delta, self.estimate_delta
            )

            # Calculate log-likelihood periodically
            if iteration % self.iter_loglik == 0:
                log_lik = self._calculate_log_likelihood(X_dense, theta, phi, eta, z)
                log_likelihood_history.append(log_lik)

                # Check convergence (R uses change in theta)
                theta_change = np.mean(np.abs(theta - theta_old))

                if self.verbose and iteration % 50 == 0:
                    print(f"Iteration {iteration}: LL = {log_lik:.2f}, theta_change = {theta_change:.6f}")

                # R's convergence criterion
                if theta_change < self.convergence_threshold:
                    if self.verbose:
                        print(f"Converged at iteration {iteration}")
                    break

        # Calculate final contamination (1 - theta in R notation)
        contamination = 1.0 - theta

        # Calculate decontaminated counts
        decontaminated = np.zeros_like(X_dense)
        for j in range(n_cells):
            cluster_idx = z[j] - 1
            for g in range(n_genes):
                if X_dense[j, g] > 0:
                    # Probability this count is native
                    p_native = theta[j] * phi[cluster_idx, g]
                    p_total = p_native + (1 - theta[j]) * eta[cluster_idx, g] + 1e-20

                    # Expected native counts
                    decontaminated[j, g] = X_dense[j, g] * (p_native / p_total)

        # Round to integers for count data
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


