"""Basic tests for decontx functionality."""

import pytest
import numpy as np
from anndata import AnnData
from decontx import decontx


def test_decontx_basic():
    """Test basic decontx functionality."""
    # Create simple test data
    np.random.seed(42)
    n_cells, n_genes = 100, 50
    X = np.random.poisson(5, size=(n_cells, n_genes))
    adata = AnnData(X)

    # Run decontx
    result = decontx(adata, copy=True, verbose=False)

    # Check results
    assert 'decontx_counts' in result.layers
    assert 'decontx_contamination' in result.obs
    assert result.layers['decontx_counts'].shape == X.shape
    assert len(result.obs['decontx_contamination']) == n_cells