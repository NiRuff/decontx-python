# DecontX Python
[![PyPI version](https://badge.fury.io/py/decontx-python.svg)](https://badge.fury.io/py/decontx-python)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

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

Validated against R decontX 1.10.0 (R 4.6.1) on the 10x Genomics 10k human/mouse mixture
(hgmm 3' v3.1; 9,588 single-species cells, species used as cluster labels, default parameters):

- **Per-cell contamination:** mean absolute difference to R < 0.0001, max < 0.001
- **Ground truth:** estimates correlate with the observed cross-species read fraction
  (r = 0.998 human, r = 1.000 mouse); R and Python show the same slight underestimation
 
![decontx-python vs. R decontX and vs. ground truth on 10x hgmm 10k](https://raw.githubusercontent.com/NiRuff/decontx-python/master/docs/validation_hgmm10k.png)

*Left: per-cell contamination, Python vs. R. Right: Python estimates vs. the fraction of reads from the other species (dashed: identity). With species as cluster labels, DecontX models only cross-species contamination, which is what the x-axis measures.*

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

## Performance Notes

- Sparse, parallel EM (numba); hgmm 10k (9.6k cells, 69k genes): ~1 s with 12 threads
- Note: R decontX always computes a UMAP, even with given clusters, so end-to-end runtimes are not directly comparable
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