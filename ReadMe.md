# DecontX Python

A Python implementation of DecontX for removing ambient RNA contamination from single-cell RNA-seq data, designed for seamless integration with scanpy workflows.

## Overview

DecontX is a Bayesian method to estimate and remove cross-contamination from ambient RNA in droplet-based single-cell RNA-seq data. This Python implementation provides near-perfect parity with the original R version (correlation > 0.999) while enabling pure Python workflows without R dependencies.

**Key Features:**
- 🐍 Pure Python implementation (no R required)
- 🔬 Seamless scanpy integration
- ⚡ Numba-accelerated performance
- 📊 Bayesian contamination estimation per cell
- 🎯 Validated against original R implementation

## Installation

```bash
pip install decontx-python
```

## Quick Start

```python
import scanpy as sc
import decontx

# Load and preprocess data with scanpy
adata = sc.read_h5ad("pbmc.h5ad")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata)

# Remove ambient RNA contamination
decontx.decontx(adata, cluster_key="leiden")

# Access results
contamination = adata.obs['decontX_contamination']
clean_counts = adata.layers['decontX_counts']

print(f"Mean contamination: {contamination.mean():.1%}")
print(f"Highly contaminated cells (>50%): {(contamination > 0.5).sum()}")
```

## Why DecontX?

Ambient RNA contamination occurs when mRNA from lysed/stressed cells gets captured in droplets with other cells, causing:
- Cross-contamination between cell types
- Blurred cell type boundaries  
- False positive marker gene expression
- Reduced clustering quality

DecontX models each cell as a mixture of:
1. **Native transcripts** from the cell's true type
2. **Contaminating transcripts** from other cell types in the sample

## Validation: R vs Python Implementation

Our Python implementation achieves near-perfect concordance with the original R version:

<img width="2250" height="1500" alt="decontx_comparison_real_data" src="https://github.com/user-attachments/assets/fc1358fa-1f54-42d9-953d-d2281d90d2d5" />

**PBMC 3K Dataset Results:**
- **Correlation: 0.999** between R and Python implementations
- **Mean Absolute Error: <1%** across all parameter settings
- **Identical statistical properties** (mean, std, range) 
- **Per-cluster consistency** maintained across cell types

## Method Comparison

Based on our benchmarking study:

| Method | Ambient RNA Removed | Precision | Conservativeness |
|--------|-------------------|-----------|------------------|
| **SoupX** | ~65% | High | Very conservative |
| **DecontX** | ~90% | Medium-High | Balanced |
| **CellBender** | ~90% | Medium | More aggressive |

**Recommendation**: 
- Use **SoupX** for maximum safety and minimal false positives
- Use **DecontX** for balanced contamination removal in standard workflows  
- Use **CellBender** when you can replace your entire preprocessing pipeline

## API Reference

### Main Function

```python
decontx.decontx(
    adata,
    cluster_key="leiden",
    max_iter=500,
    delta=(10.0, 10.0),
    estimate_delta=True,
    convergence=0.001,
    copy=False
)
```

**Parameters:**
- `adata`: AnnData object with raw counts in `.X`
- `cluster_key`: Column in `.obs` containing cluster labels
- `max_iter`: Maximum EM iterations (default: 500)
- `delta`: Beta prior parameters for contamination (default: (10,10))
- `estimate_delta`: Whether to estimate delta parameters (default: True)
- `convergence`: Convergence threshold (default: 0.001)
- `copy`: Return copy or modify in place (default: False)

**Returns:**
Results stored in `adata`:
- `adata.obs['decontX_contamination']`: Per-cell contamination estimates
- `adata.layers['decontX_counts']`: Decontaminated count matrix
- `adata.uns['decontX']`: Model parameters and metadata

### Utility Functions

```python
# Get decontaminated counts as array
clean_counts = decontx.get_decontx_counts(adata)

# Get contamination estimates
contamination = decontx.get_decontx_contamination(adata)

# Simple simulation for testing
sim_data = decontx.simulate_contamination(n_cells=1000, n_genes=2000)
```

## Performance Notes

- Python implementation is ~2-3x slower than R version
- Performance acceptable for typical datasets (<50k cells)
- Numba JIT compilation provides significant speedup after first run
- Memory usage scales linearly with dataset size

## Integration with Existing Workflows

DecontX fits naturally into scanpy workflows:

```python
# Standard scanpy analysis
sc.tl.leiden(adata, resolution=0.5)
sc.tl.rank_genes_groups(adata, 'leiden')

# Add decontamination
decontx.decontx(adata, cluster_key='leiden')

# Continue with decontaminated data
adata.X = adata.layers['decontX_counts']
sc.pp.log1p(adata)  # Re-log transform clean counts
sc.pp.scale(adata)
sc.tl.pca(adata)
sc.pl.pca_variance_ratio(adata, n_pcs=50)
```

## Citation

If you use DecontX in your research, please cite:

> Yang, S., Corbett, S.E., Koga, Y. et al. Decontamination of ambient RNA in single-cell RNA-seq with DecontX. Genome Biol 21, 57 (2020). https://doi.org/10.1186/s13059-020-1950-6

## Issues and Support

- 🐛 Report bugs: [GitHub Issues](https://github.com/yourusername/decontx-python/issues)
- 📖 Documentation: [Read the Docs](https://decontx-python.readthedocs.io)
- 💬 Questions: [GitHub Discussions](https://github.com/yourusername/decontx-python/discussions)

## License

MIT License - see LICENSE file for details.