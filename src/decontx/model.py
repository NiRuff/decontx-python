"""
DecontX Bayesian mixture model implementation.
Complete variational inference following Yang et al. (2020).
"""

import numpy as np
from numba import jit, prange
from scipy import optimize
from scipy.special import digamma, polygamma, gammaln
from scipy.stats import beta, dirichlet
# from sklearn.mixture import BayesianGaussianMixture
import scanpy as sc
import umap
from sklearn.cluster import DBSCAN
from typing import Tuple, Optional, Dict
import warnings


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



#
# def fit_dirichlet_moments(data: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
#     """
#     Fit Dirichlet distribution parameters using method of moments.
#     More sophisticated version matching R's MCMCprecision::fit_dirichlet behavior.
#     """
#     # Remove rows with all zeros or invalid values
#     valid_mask = np.all(np.isfinite(data), axis=1) & (np.sum(data, axis=1) > 0)
#     clean_data = data[valid_mask]
#
#     if len(clean_data) < 2:
#         return np.array([1.0, 1.0])
#
#     # Normalize to ensure they sum to 1
#     clean_data = clean_data / clean_data.sum(axis=1, keepdims=True)
#
#     # Method of moments initialization
#     means = np.mean(clean_data, axis=0)
#
#     # Calculate sample covariance
#     centered = clean_data - means
#     cov_matrix = np.cov(centered.T)
#
#     # Method of moments estimate for concentration
#     var_sum = np.trace(cov_matrix)
#     mean_sum_sq = np.sum(means ** 2)
#
#     if var_sum > 0 and mean_sum_sq > 0:
#         alpha_sum = (mean_sum_sq - var_sum) / var_sum
#         if alpha_sum > 0:
#             alphas = means * alpha_sum
#             # Ensure minimum values
#             alphas = np.maximum(alphas, 0.1)
#             return alphas
#
#     # Fallback to equal parameters
#     return np.ones(data.shape[1]) * 0.5
#
#
# def fit_beta_precise(theta_values: np.ndarray) -> np.ndarray:
#     """
#     Precise beta distribution fitting matching R's approach.
#     """
#     # Remove invalid values
#     valid_theta = theta_values[np.isfinite(theta_values)]
#     valid_theta = np.clip(valid_theta, 1e-10, 1 - 1e-10)
#
#     if len(valid_theta) < 2:
#         return np.array([10.0, 10.0])
#
#     # Method of moments
#     mean_val = np.mean(valid_theta)
#     var_val = np.var(valid_theta)
#
#     if var_val > 0 and var_val < mean_val * (1 - mean_val):
#         # Method of moments estimator
#         common = mean_val * (1 - mean_val) / var_val - 1
#         if common > 0:
#             alpha = mean_val * common
#             beta_param = (1 - mean_val) * common
#             return np.array([max(alpha, 0.1), max(beta_param, 0.1)])
#
#     # Fallback using MLE
#     def neg_log_likelihood(params):
#         alpha, beta_param = params
#         if alpha <= 0 or beta_param <= 0:
#             return np.inf
#         try:
#             return -np.sum(beta.logpdf(valid_theta, alpha, beta_param))
#         except:
#             return np.inf
#
#     try:
#         result = optimize.minimize(
#             neg_log_likelihood,
#             x0=[1.0, 1.0],
#             bounds=[(0.1, 100), (0.1, 100)],
#             method='L-BFGS-B'
#         )
#         if result.success:
#             return result.x
#     except:
#         pass
#
#     return np.array([10.0, 10.0])