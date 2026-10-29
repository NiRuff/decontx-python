"""Small decontX-style simulator with known ground truth (used by tests only)."""
import numpy as np
import scipy.sparse as sp


def simulate(n_cells=800, n_genes=1500, n_clusters=5, seed=0):
    """Native counts ~ Mult(phi_k); ambient counts ~ Mult(eta_k), where eta_k is the
    size-weighted mix of the other clusters. Returns (X, native, z, true_contamination)."""
    rng = np.random.default_rng(seed)
    z = rng.choice(n_clusters, n_cells, p=rng.dirichlet(np.full(n_clusters, 3.0)))
    phi = rng.dirichlet(np.full(n_genes, 0.05), size=n_clusters)
    w = np.bincount(z, minlength=n_clusters) / n_cells
    eta = np.array([np.delete(w, k) @ np.delete(phi, k, axis=0) for k in range(n_clusters)])
    eta /= eta.sum(1, keepdims=True)
    n_umi = rng.lognormal(np.log(2000), 0.5, n_cells).astype(int) + 200
    contam = rng.beta(2, 12, n_cells)
    native = np.zeros((n_cells, n_genes))
    total = np.zeros((n_cells, n_genes))
    for j in range(n_cells):
        n_nat = rng.binomial(n_umi[j], 1 - contam[j])
        native[j] = rng.multinomial(n_nat, phi[z[j]])
        total[j] = native[j] + rng.multinomial(n_umi[j] - n_nat, eta[z[j]])
    keep = total.sum(0) > 0
    total, native = total[:, keep], native[:, keep]
    true_c = 1 - native.sum(1) / total.sum(1)
    return sp.csr_matrix(total), sp.csr_matrix(native), z, true_c
