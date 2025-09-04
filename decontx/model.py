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
    """

    def __init__(self, **kwargs):
        self.max_iter = kwargs.get('max_iter', 500)
        self.convergence_threshold = kwargs.get('convergence', 0.001)  # Note: 'convergence' in R
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
        """
        n_cells, n_genes = X.shape
        n_clusters = len(np.unique(z))

        # Initialize phi and eta exactly like R
        phi = np.full((n_clusters, n_genes), pseudocount)
        eta = np.full((n_clusters, n_genes), pseudocount)

        # R's exact phi initialization
        for k in range(n_clusters):
            cluster_mask = (z == k + 1)  # z is 1-indexed
            if np.any(cluster_mask):
                cluster_counts = X[cluster_mask].sum(axis=0)
                if hasattr(cluster_counts, 'A1'):  # sparse matrix
                    cluster_counts = cluster_counts.A1

                total_cluster_counts = cluster_counts.sum()
                if total_cluster_counts > 0:
                    # R's exact normalization with pseudocount
                    phi[k, :] = (cluster_counts + pseudocount) / (total_cluster_counts + n_genes * pseudocount)

        # R's exact eta initialization (contamination from other clusters)
        for k in range(n_clusters):
            other_counts = np.zeros(n_genes)

            for other_k in range(n_clusters):
                if other_k != k:
                    other_mask = (z == other_k + 1)
                    if np.any(other_mask):
                        other_cluster_counts = X[other_mask].sum(axis=0)
                        if hasattr(other_cluster_counts, 'A1'):
                            other_cluster_counts = other_cluster_counts.A1
                        other_counts += other_cluster_counts

            total_other_counts = other_counts.sum()
            if total_other_counts > 0:
                eta[k, :] = (other_counts + pseudocount) / (total_other_counts + n_genes * pseudocount)
            else:
                # Uniform if no other clusters
                eta[k, :] = 1.0 / n_genes

        return phi, eta

    def _r_exact_em_step(self, X, z, theta, phi, eta, counts_colsums, pseudocount=1e-20):
        """
        Exact R EM step matching decontXEM C++ function.
        """
        n_cells, n_genes = X.shape
        n_clusters = len(np.unique(z))

        # E-step: Calculate responsibilities (exact R formula)
        estRmat = np.zeros_like(X, dtype=np.float64)

        for j in range(n_cells):
            cluster_idx = z[j] - 1  # Convert to 0-indexed
            for g in range(n_genes):
                if X[j, g] > 0:
                    # R's exact probability calculations
                    log_native = np.log(phi[cluster_idx, g] + pseudocount) + np.log(theta[j] + pseudocount)
                    log_contam = np.log(eta[cluster_idx, g] + pseudocount) + np.log(1 - theta[j] + pseudocount)

                    # Numerically stable computation (R method)
                    max_val = max(log_native, log_contam)
                    exp_native = np.exp(log_native - max_val)
                    exp_contam = np.exp(log_contam - max_val)

                    p_native = exp_native / (exp_native + exp_contam)
                    estRmat[j, g] = X[j, g] * p_native

        # M-step: Update parameters (exact R formulas)

        # Update theta (exact R formula)
        estRmat_colsums = estRmat.sum(axis=1)
        new_theta = (estRmat_colsums + self.delta[0]) / (counts_colsums + self.delta[0] + self.delta[1])

        # Update phi (native expression) - exact R formula
        new_phi = np.zeros_like(phi)
        for k in range(n_clusters):
            cluster_sum = np.zeros(n_genes)
            cluster_mask = (z == k + 1)

            if np.any(cluster_mask):
                cluster_sum = estRmat[cluster_mask].sum(axis=0)
                if hasattr(cluster_sum, 'A1'):
                    cluster_sum = cluster_sum.A1

            phi_sum = cluster_sum.sum() + n_genes * pseudocount
            new_phi[k, :] = (cluster_sum + pseudocount) / phi_sum

        # Update eta (contamination) - exact R formula
        new_eta = eta.copy()
        if True:  # Always update eta in this version
            for k in range(n_clusters):
                contam_sum = np.zeros(n_genes)

                # Sum contamination from all OTHER clusters
                for j in range(n_cells):
                    if z[j] - 1 != k:  # From other clusters
                        contam_counts = X[j, :] - estRmat[j, :]
                        contam_counts = np.maximum(contam_counts, 0)  # Ensure non-negative
                        contam_sum += contam_counts

                eta_sum = contam_sum.sum() + n_genes * pseudocount
                if eta_sum > 0:
                    new_eta[k, :] = (contam_sum + pseudocount) / eta_sum

        # Calculate contamination (exact R formula)
        contamination = 1.0 - new_theta

        return new_theta, new_phi, new_eta, contamination, estRmat

    def _r_exact_log_likelihood(self, X, z, theta, phi, eta, pseudocount=1e-20):
        """
        Exact R log-likelihood calculation matching decontXLogLik.
        """
        log_lik = 0.0
        n_cells, n_genes = X.shape

        for j in range(n_cells):
            cluster_idx = z[j] - 1
            for g in range(n_genes):
                if X[j, g] > 0:
                    # R's exact mixture probability formula
                    mixture_prob = (theta[j] * phi[cluster_idx, g] +
                                    (1 - theta[j]) * eta[cluster_idx, g] + pseudocount)
                    log_lik += X[j, g] * np.log(mixture_prob)

        return log_lik

    def fit_transform(self, X, z, X_background=None):
        """
        Fit using exact R algorithm.
        """
        np.random.seed(self.seed)

        if not isinstance(X, np.ndarray):
            X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)

        z = np.asarray(z).astype(int)
        n_cells, n_genes = X.shape
        n_clusters = len(np.unique(z))

        # Initialize theta exactly like R: rbeta(n=nC, shape1=delta[1], shape2=delta[2])
        from scipy.stats import beta
        theta = beta.rvs(self.delta[0], self.delta[1], size=n_cells, random_state=self.seed)

        # Initialize phi and eta exactly like R
        phi, eta = self._r_exact_initialization(X, z, theta)

        # Handle background exactly like R (if provided)
        if X_background is not None:
            if not isinstance(X_background, np.ndarray):
                X_background = X_background.toarray() if hasattr(X_background, 'toarray') else np.asarray(X_background)

            # Use background to estimate eta (exact R method)
            bg_total = X_background.sum(axis=0)
            if hasattr(bg_total, 'A1'):
                bg_total = bg_total.A1
            bg_sum = bg_total.sum()

            if bg_sum > 0:
                eta_bg = (bg_total + 1e-20) / (bg_sum + n_genes * 1e-20)
                # Replicate for all clusters
                eta = np.tile(eta_bg, (n_clusters, 1))

        # Prepare for EM iterations
        prev_theta = theta.copy()
        counts_colsums = X.sum(axis=1)
        if hasattr(counts_colsums, 'A1'):
            counts_colsums = counts_colsums.A1

        self.log_likelihood_ = []

        for iteration in range(self.max_iter):
            # Exact R EM step
            theta, phi, eta, contamination, estRmat = self._r_exact_em_step(
                X, z, theta, phi, eta, counts_colsums
            )

            # Check convergence (exact R method)
            max_change = np.max(np.abs(theta - prev_theta))

            # Calculate log-likelihood (exact R timing)
            if (iteration + 1) % self.iter_loglik == 0 or max_change < self.convergence_threshold:
                ll = self._r_exact_log_likelihood(X, z, theta, phi, eta)
                self.log_likelihood_.append(ll)

                if self.verbose:
                    print(f"Iteration {iteration + 1}: max_change={max_change:.6f}, LL={ll:.2f}")

            if max_change < self.convergence_threshold:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            prev_theta = theta.copy()

        # Calculate decontaminated counts (exact R method)
        decontaminated_counts = np.zeros_like(X, dtype=np.float64)
        for j in range(n_cells):
            cluster_idx = z[j] - 1
            for g in range(n_genes):
                if X[j, g] > 0:
                    log_native = np.log(phi[cluster_idx, g] + 1e-20) + np.log(theta[j] + 1e-20)
                    log_contam = np.log(eta[cluster_idx, g] + 1e-20) + np.log(1 - theta[j] + 1e-20)

                    max_val = max(log_native, log_contam)
                    p_native = np.exp(log_native - max_val) / (
                            np.exp(log_native - max_val) + np.exp(log_contam - max_val)
                    )
                    decontaminated_counts[j, g] = X[j, g] * p_native

        # Store parameters
        self.phi_ = phi
        self.eta_ = eta
        self.theta_ = theta

        return {
            'decontaminated_counts': decontaminated_counts,
            'contamination': contamination,
            'theta': theta,
            'phi': phi,
            'eta': eta,
            'delta': self.delta,
            'log_likelihood': self.log_likelihood_,
        }


