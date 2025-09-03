"""
DecontX Bayesian mixture model implementation.
Complete variational inference following Yang et al. (2020).
"""

import numpy as np
from scipy.special import digamma, polygamma, gammaln, logsumexp
from scipy.stats import beta
from scipy.sparse import issparse, csr_matrix
from sklearn.preprocessing import normalize
from typing import Dict, Tuple, Optional
import warnings


class DecontXModel:
    """
    DecontX Bayesian mixture model for ambient RNA decontamination.

    This implements the variational inference algorithm described in
    Yang et al. (2020) Genome Biology.
    """

    def __init__(
        self,
        max_iter: int = 500,
        convergence_threshold: float = 0.001,
        delta: Tuple[float, float] = (10.0, 10.0),
        estimate_delta: bool = True,
        iter_loglik: int = 10,
        random_state: int = 12345,
        verbose: bool = True,
    ):
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold
        self.delta = np.array(delta)
        self.estimate_delta = estimate_delta
        self.iter_loglik = iter_loglik
        self.random_state = random_state
        self.verbose = verbose

        # Model parameters (set during fitting)
        self.phi_ = None  # Native expression distributions
        self.eta_ = None  # Contamination distributions
        self.theta_ = None  # Contamination proportions per cell
        self.log_likelihood_ = []

    def fit_transform(
        self,
        X: np.ndarray,
        z: np.ndarray,
        X_background: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Fit DecontX model and return decontaminated counts.

        Parameters
        ----------
        X : array-like, shape (n_cells, n_genes)
            Count matrix
        z : array-like, shape (n_cells,)
            Cluster assignments (1-indexed)
        X_background : array-like, optional
            Empty droplet counts for empirical contamination distribution

        Returns
        -------
        dict
            Dictionary containing decontaminated counts and parameters
        """
        np.random.seed(self.random_state)

        # Convert to numpy arrays
        if issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = np.asarray(X)

        z = np.asarray(z).astype(int)
        n_cells, n_genes = X_dense.shape
        n_clusters = len(np.unique(z))

        if self.verbose:
            print(f"Fitting DecontX: {n_cells} cells, {n_genes} genes, {n_clusters} clusters")

        # Initialize parameters
        theta, phi, eta = self._initialize_parameters(X_dense, z, n_clusters, X_background)

        # Initialize variational parameters
        gamma = np.zeros((n_cells, 2))
        gamma[:, 0] = self.delta[0]
        gamma[:, 1] = self.delta[1]

        # Run variational EM
        theta, phi, eta, contamination = self._variational_em(
            X_dense, z, theta, phi, eta, gamma, n_clusters
        )

        # Calculate decontaminated counts
        decontaminated_counts = self._calculate_native_counts(X_dense, z, theta, phi, eta)

        return {
            'decontaminated_counts': decontaminated_counts,
            'contamination': contamination,
            'theta': theta,
            'phi': phi,
            'eta': eta,
            'delta': self.delta,
            'log_likelihood': self.log_likelihood_,
        }

    def _initialize_parameters(
        self,
        X: np.ndarray,
        z: np.ndarray,
        n_clusters: int,
        X_background: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize model parameters."""
        n_cells, n_genes = X.shape

        # Initialize theta from beta distribution
        theta = beta.rvs(self.delta[0], self.delta[1], size=n_cells)

        # Initialize phi (native expression)
        phi = np.zeros((n_clusters, n_genes)) + 1e-20
        for k in range(n_clusters):
            cluster_cells = (z == k + 1)  # 1-indexed clusters
            if np.sum(cluster_cells) > 0:
                cluster_counts = X[cluster_cells].sum(axis=0) + 1e-20
                phi[k] = cluster_counts / cluster_counts.sum()

        # Initialize eta (contamination)
        if X_background is not None:
            # Use empirical distribution from background
            if issparse(X_background):
                background_sum = np.array(X_background.sum(axis=0)).flatten() + 1e-20
            else:
                background_sum = X_background.sum(axis=0) + 1e-20
            eta_empirical = background_sum / background_sum.sum()
            eta = np.tile(eta_empirical, (n_clusters, 1))
        else:
            # Initialize as weighted combination of other clusters
            eta = np.zeros((n_clusters, n_genes)) + 1e-20
            for k in range(n_clusters):
                other_clusters = np.arange(n_clusters) != k
                if np.sum(other_clusters) > 0:
                    eta[k] = phi[other_clusters].mean(axis=0)
                    eta[k] = eta[k] / eta[k].sum()

        return theta, phi, eta

    def _variational_em(
        self,
        X: np.ndarray,
        z: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        eta: np.ndarray,
        gamma: np.ndarray,
        n_clusters: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run variational EM algorithm."""
        n_cells, n_genes = X.shape
        prev_theta = theta.copy()

        for iteration in range(self.max_iter):
            # E-step: Update variational parameters pi for each transcript
            pi = self._e_step(X, z, theta, phi, eta, gamma)

            # M-step: Update model parameters
            theta, phi, eta, gamma = self._m_step(
                X, z, pi, gamma, n_clusters
            )

            # Calculate contamination
            contamination = theta

            # Check convergence
            max_change = np.max(np.abs(theta - prev_theta))

            # Calculate log-likelihood periodically
            if (iteration + 1) % self.iter_loglik == 0 or max_change < self.convergence_threshold:
                ll = self._calculate_log_likelihood(X, z, phi, eta, theta)
                self.log_likelihood_.append(ll)

                if self.verbose:
                    print(f"Iteration {iteration + 1}: max_change={max_change:.6f}, LL={ll:.2f}")

            if max_change < self.convergence_threshold:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            prev_theta = theta.copy()

        self.phi_ = phi
        self.eta_ = eta
        self.theta_ = theta

        return theta, phi, eta, contamination

    def _e_step(
        self,
        X: np.ndarray,
        z: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        eta: np.ndarray,
        gamma: np.ndarray
    ) -> np.ndarray:
        """E-step: Update variational parameters."""
        n_cells, n_genes = X.shape

        # Pre-compute digamma values
        digamma_gamma1 = digamma(gamma[:, 0])
        digamma_gamma2 = digamma(gamma[:, 1])
        digamma_gamma_sum = digamma(gamma[:, 0] + gamma[:, 1])

        # Calculate pi for each cell
        pi = np.zeros((n_cells, 2))

        for j in range(n_cells):
            cluster_idx = z[j] - 1  # Convert to 0-indexed

            # Log probabilities for native expression
            log_phi = np.log(phi[cluster_idx] + 1e-20)
            log_native = log_phi + (digamma_gamma1[j] - digamma_gamma_sum[j])

            # Log probabilities for contamination
            log_eta = np.log(eta[cluster_idx] + 1e-20)
            log_contam = log_eta + (digamma_gamma2[j] - digamma_gamma_sum[j])

            # Average over genes with counts
            gene_weights = X[j] / (X[j].sum() + 1e-20)
            pi[j, 0] = np.sum(gene_weights * np.exp(log_native - np.logaddexp(log_native, log_contam)))
            pi[j, 1] = 1 - pi[j, 0]

        return pi

    def _m_step(
        self,
        X: np.ndarray,
        z: np.ndarray,
        pi: np.ndarray,
        gamma: np.ndarray,
        n_clusters: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """M-step: Update model parameters."""
        n_cells, n_genes = X.shape

        # Update gamma (variational parameters for theta)
        X_colsums = X.sum(axis=1)
        gamma[:, 0] = self.delta[0] + pi[:, 0] * X_colsums
        gamma[:, 1] = self.delta[1] + pi[:, 1] * X_colsums

        # Update theta
        theta = gamma[:, 0] / (gamma[:, 0] + gamma[:, 1])

        # Update phi (native expression distributions)
        phi = np.zeros((n_clusters, n_genes)) + 1e-20
        for k in range(n_clusters):
            cluster_mask = (z == k + 1)
            if np.sum(cluster_mask) > 0:
                # Weight counts by pi (probability of being native)
                weighted_counts = X[cluster_mask] * pi[cluster_mask, 0:1]
                phi[k] = weighted_counts.sum(axis=0) + 1e-20
                phi[k] = phi[k] / phi[k].sum()

        # Update eta (contamination distributions)
        # unless using fixed empirical distribution
        if not hasattr(self, '_use_empirical_eta'):
            eta = np.zeros((n_clusters, n_genes)) + 1e-20
            for k in range(n_clusters):
                cluster_mask = (z == k + 1)
                other_mask = ~cluster_mask
                if np.sum(other_mask) > 0:
                    # Contamination comes from other clusters
                    weighted_counts = X[other_mask] * pi[other_mask, 1:2]
                    eta[k] = weighted_counts.sum(axis=0) + 1e-20
                    eta[k] = eta[k] / eta[k].sum()

        # Update delta if requested
        if self.estimate_delta:
            # Fit beta distribution to theta values
            self.delta = self._fit_beta_mm(theta)
            gamma[:, 0] = self.delta[0] + pi[:, 0] * X_colsums
            gamma[:, 1] = self.delta[1] + pi[:, 1] * X_colsums
            theta = gamma[:, 0] / (gamma[:, 0] + gamma[:, 1])

        return theta, phi, eta, gamma

    def _calculate_native_counts(
        self,
        X: np.ndarray,
        z: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        eta: np.ndarray
    ) -> np.ndarray:
        """Calculate decontaminated (native) counts."""
        n_cells, n_genes = X.shape
        native_counts = np.zeros_like(X, dtype=np.float64)

        for j in range(n_cells):
            cluster_idx = z[j] - 1

            # Calculate posterior probability each count is native
            log_native = np.log(phi[cluster_idx] + 1e-20) + np.log(theta[j] + 1e-20)
            log_contam = np.log(eta[cluster_idx] + 1e-20) + np.log(1 - theta[j] + 1e-20)

            p_native = np.exp(log_native - np.logaddexp(log_native, log_contam))
            native_counts[j] = X[j] * p_native

        return native_counts

    def _calculate_log_likelihood(
        self,
        X: np.ndarray,
        z: np.ndarray,
        phi: np.ndarray,
        eta: np.ndarray,
        theta: np.ndarray
    ) -> float:
        """Calculate log-likelihood of the model."""
        ll = 0.0
        n_cells = X.shape[0]

        for j in range(n_cells):
            cluster_idx = z[j] - 1
            mixture = theta[j] * phi[cluster_idx] + (1 - theta[j]) * eta[cluster_idx]
            ll += np.sum(X[j] * np.log(mixture + 1e-20))

        return ll

    def _fit_beta_mm(self, x: np.ndarray) -> np.ndarray:
        """
        Fit beta distribution using method of moments.
        Simple alternative to R's MCMCprecision::fit_dirichlet.
        """
        x = np.clip(x, 1e-10, 1 - 1e-10)
        mean = np.mean(x)
        var = np.var(x)

        if var < mean * (1 - mean):
            common = mean * (1 - mean) / var - 1
            a = mean * common
            b = (1 - mean) * common
        else:
            a = 10.0
            b = 10.0

        return np.array([a, b])