"""
DecontX Bayesian mixture model implementation.
Complete variational inference following Yang et al. (2020).
"""

import numpy as np
from numba import jit, prange
from scipy import optimize
from scipy.special import digamma, polygamma, gammaln
from scipy.stats import beta, dirichlet
from typing import Tuple, Optional, Dict
import warnings

# Import the fast operations
from .fast_ops import (
    decontx_initialize_exact,
    decontx_em_exact,
    decontx_log_likelihood_exact
)


class DecontXModel:
    """
     DecontX model matching R implementation exactly.
    """

    def __init__(self, **kwargs):
        # Same parameters as original but with exact R behavior
        self.max_iter = kwargs.get('max_iter', 500)
        self.convergence_threshold = kwargs.get('convergence_threshold', 0.001)
        self.delta = np.array(kwargs.get('delta', [10.0, 10.0]))
        self.estimate_delta = kwargs.get('estimate_delta', True)
        self.iter_loglik = kwargs.get('iter_loglik', 10)
        self.random_state = kwargs.get('random_state', 12345)
        self.verbose = kwargs.get('verbose', True)

        # Storage for results
        self.phi_ = None
        self.eta_ = None
        self.theta_ = None
        self.log_likelihood_ = []

    def fit_transform(
            self,
            X: np.ndarray,
            z: np.ndarray,
            X_background: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Fit using exact R algorithm.
        """
        np.random.seed(self.random_state)

        if not isinstance(X, np.ndarray):
            X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)

        z = np.asarray(z).astype(int)
        n_cells, n_genes = X.shape
        n_clusters = len(np.unique(z))

        # Initialize parameters exactly like R
        theta = beta.rvs(self.delta[0], self.delta[1], size=n_cells, random_state=self.random_state)
        phi, eta = decontx_initialize_exact(X, theta, z)

        # Use empirical distribution for eta if background provided
        if X_background is not None:
            if not isinstance(X_background, np.ndarray):
                X_background = X_background.toarray() if hasattr(X_background, 'toarray') else np.asarray(X_background)

            # Empirical contamination distribution
            bg_counts = np.sum(X_background, axis=0) + 1e-20
            eta_empirical = bg_counts / np.sum(bg_counts)
            eta = np.tile(eta_empirical, (n_clusters, 1))
            estimate_eta = False
        else:
            estimate_eta = True

        # Run exact EM algorithm
        prev_theta = theta.copy()
        counts_colsums = np.sum(X, axis=1)

        for iteration in range(self.max_iter):
            theta, phi, eta, delta, contamination = decontx_em_exact(
                X, counts_colsums, theta, estimate_eta, eta, phi, z,
                self.estimate_delta, self.delta, 1e-20
            )

            # Update delta
            if self.estimate_delta:
                self.delta = delta

            # Check convergence
            max_change = np.max(np.abs(theta - prev_theta))

            # Calculate log-likelihood
            if (iteration + 1) % self.iter_loglik == 0 or max_change < self.convergence_threshold:
                ll = decontx_log_likelihood_exact(X, theta, eta, phi, z)
                self.log_likelihood_.append(ll)

                if self.verbose:
                    print(f"Iteration {iteration + 1}: max_change={max_change:.6f}, LL={ll:.2f}")

            if max_change < self.convergence_threshold:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            prev_theta = theta.copy()

        # Calculate decontaminated counts
        decontaminated_counts = self._calculate_native_counts_exact(X, z, theta, phi, eta)

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

    def _calculate_native_counts_exact(self, X, z, theta, phi, eta):
        """Calculate native counts using exact R logic."""
        n_cells, n_genes = X.shape
        native_counts = np.zeros_like(X, dtype=np.float64)

        for j in range(n_cells):
            cluster_idx = z[j] - 1
            for g in range(n_genes):
                log_native = np.log(phi[cluster_idx, g] + 1e-20) + np.log(theta[j] + 1e-20)
                log_contam = np.log(eta[cluster_idx, g] + 1e-20) + np.log(1 - theta[j] + 1e-20)

                # Numerically stable computation
                max_val = max(log_native, log_contam)
                p_native = np.exp(log_native - max_val) / (
                        np.exp(log_native - max_val) + np.exp(log_contam - max_val)
                )
                native_counts[j, g] = X[j, g] * p_native

        return native_counts


