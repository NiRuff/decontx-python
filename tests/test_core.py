"""Tests for decontx sparse EM rewrite."""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from anndata import AnnData
from decontx import decontx


def _make_adata(n_cells=100, n_genes=50, sparse=False, seed=42):
    """Build a minimal AnnData with 2 dummy clusters."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32)
    if sparse:
        X = csr_matrix(X)
    adata = AnnData(X)
    # Two-cluster assignment required by decontx
    adata.obs['leiden'] = (rng.integers(0, 2, size=n_cells).astype(str))
    return adata


def test_decontx_dense_input():
    """Dense input still works after sparse rewrite."""
    adata = _make_adata(sparse=False)
    result = decontx(adata, copy=True, verbose=False)

    assert 'decontX_counts' in result.layers
    assert 'decontX_contamination' in result.obs
    assert result.layers['decontX_counts'].shape == (100, 50)
    assert len(result.obs['decontX_contamination']) == 100


def test_decontx_sparse_input():
    """Sparse input produces sparse output with correct sparsity pattern."""
    adata = _make_adata(sparse=True)
    original_nnz = adata.X.nnz

    result = decontx(adata, copy=True, verbose=False)

    layer = result.layers['decontX_counts']
    assert issparse(layer), "decontX_counts layer should be sparse"
    assert layer.shape == (100, 50)

    # Same nonzero positions as input (native counts are 0 only where input is 0)
    assert layer.nnz == original_nnz, (
        f"sparsity pattern changed: input nnz={original_nnz}, output nnz={layer.nnz}"
    )


def test_decontx_contamination_range():
    """Contamination estimates must lie in [0, 1]."""
    adata = _make_adata(sparse=True)
    result = decontx(adata, copy=True, verbose=False)

    contam = result.obs['decontX_contamination'].values
    assert (contam >= 0).all() and (contam <= 1).all(), (
        f"contamination out of [0,1]: min={contam.min():.4f}, max={contam.max():.4f}"
    )


def test_decontx_nonnegative_counts():
    """Corrected counts must be >= 0."""
    adata = _make_adata(sparse=True)
    result = decontx(adata, copy=True, verbose=False)

    layer = result.layers['decontX_counts']
    data = layer.data if issparse(layer) else layer
    assert (data >= 0).all(), "corrected counts contain negatives"


def test_decontx_counts_le_raw():
    """Per-cell total corrected counts should not exceed raw counts."""
    adata = _make_adata(sparse=True)
    result = decontx(adata, copy=True, verbose=False)

    raw_sums = np.asarray(adata.X.sum(axis=1)).ravel()
    layer = result.layers['decontX_counts']
    corrected_sums = np.asarray(layer.sum(axis=1)).ravel()

    # Allow small floating-point overage from rounding
    assert (corrected_sums <= raw_sums + 1).all(), (
        "corrected cell totals exceed raw counts (more than rounding tolerance)"
    )


def test_decontx_uns_metadata():
    """uns['decontX'] must be present after a run."""
    adata = _make_adata(sparse=True)
    result = decontx(adata, copy=True, verbose=False)

    assert 'decontX' in result.uns
    assert 'parameters' in result.uns['decontX']
