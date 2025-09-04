"""
DecontX plotting functions matching R implementation.
Add these to plotting.py or create new plotting.py file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Union, Dict
import warnings


def plot_contamination_umap(
        adata=None,
        result_dict=None,
        umap_key='X_decontX_umap',
        contamination_key='decontX_contamination',
        batch=None,
        color_scale=None,
        size=1,
        figsize=(8, 6),
        title="DecontX Contamination",
        save=None
):
    """
    Plot contamination on UMAP coordinates.
    Equivalent to R's plotDecontXContamination.

    Parameters
    ----------
    adata : AnnData, optional
        Annotated data object with DecontX results
    result_dict : dict, optional
        Result dictionary from decontx() function
    umap_key : str
        Key for UMAP coordinates in adata.obsm
    contamination_key : str
        Key for contamination values in adata.obs
    batch : str, optional
        Batch to plot if multiple batches
    color_scale : list, optional
        Color scale for contamination
    size : float
        Point size
    figsize : tuple
        Figure size
    title : str
        Plot title
    save : str, optional
        Path to save figure
    """
    if color_scale is None:
        color_scale = ['blue', 'green', 'yellow', 'orange', 'red']

    # Get data from either adata or result_dict
    if adata is not None:
        if umap_key not in adata.obsm:
            raise KeyError(f"UMAP coordinates '{umap_key}' not found in adata.obsm")
        if contamination_key not in adata.obs:
            raise KeyError(f"Contamination values '{contamination_key}' not found in adata.obs")

        umap_coords = adata.obsm[umap_key]
        contamination = adata.obs[contamination_key].values

    elif result_dict is not None:
        # Handle batch results
        if 'batch_results' in result_dict:
            if batch is None:
                batch = list(result_dict['batch_results'].keys())[0]

            if batch not in result_dict['batch_results']:
                raise KeyError(f"Batch '{batch}' not found")

            batch_result = result_dict['batch_results'][batch]
            umap_coords = batch_result.get('UMAP')
            contamination = batch_result.get('contamination')
        else:
            # Single batch result
            umap_coords = result_dict.get('UMAP')
            contamination = result_dict.get('contamination')

        if umap_coords is None or contamination is None:
            raise ValueError("UMAP coordinates or contamination values not found in result")
    else:
        raise ValueError("Either adata or result_dict must be provided")

    # Remove NaN values
    valid_mask = ~(np.isnan(umap_coords[:, 0]) | np.isnan(umap_coords[:, 1]) | np.isnan(contamination))
    umap_coords = umap_coords[valid_mask]
    contamination = contamination[valid_mask]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=contamination,
        s=size,
        cmap=plt.cm.get_cmap('viridis'),  # Will customize below
        vmin=0,
        vmax=1
    )

    # Customize colormap to match R version
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list("contamination", color_scale)
    scatter.set_cmap(custom_cmap)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Contamination', rotation=270, labelpad=20)

    # Formatting
    ax.set_xlabel('DecontX_UMAP_1')
    ax.set_ylabel('DecontX_UMAP_2')
    ax.set_title(title)
    ax.grid(False)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')

    return fig


def plot_marker_percentage(
        adata,
        markers: Dict[str, List[str]],
        group_clusters: Optional[Dict] = None,
        layer_keys: List[str] = ['X', 'decontX_counts'],
        cluster_key: str = 'decontX_clusters',
        threshold: float = 1,
        ncol: Optional[int] = None,
        figsize: Optional[tuple] = None,
        save: Optional[str] = None
):
    """
    Plot percentage of cells expressing markers by cluster.
    Equivalent to R's plotDecontXMarkerPercentage.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    markers : dict
        Dictionary mapping cell type names to marker gene lists
    group_clusters : dict, optional
        Dictionary to regroup clusters
    layer_keys : list
        List of layers/matrices to compare (e.g., ['X', 'decontX_counts'])
    cluster_key : str
        Key for cluster labels in adata.obs
    threshold : float
        Minimum expression threshold
    ncol : int, optional
        Number of columns in subplot grid
    figsize : tuple, optional
        Figure size
    save : str, optional
        Path to save figure
    """
    if ncol is None:
        ncol = int(np.ceil(np.sqrt(len(markers))))

    if figsize is None:
        nrow = int(np.ceil(len(markers) / ncol))
        figsize = (4 * ncol, 3 * nrow)

    # Get cluster labels
    if cluster_key not in adata.obs:
        raise KeyError(f"Cluster key '{cluster_key}' not found in adata.obs")

    clusters = adata.obs[cluster_key].values

    # Apply cluster groupings if provided
    if group_clusters is not None:
        cluster_mapping = {}
        for group_name, cluster_list in group_clusters.items():
            for cluster in cluster_list:
                cluster_mapping[cluster] = group_name

        # Remap clusters
        clusters = np.array([cluster_mapping.get(c, c) for c in clusters])
        cluster_names = list(group_clusters.keys())
    else:
        cluster_names = np.unique(clusters)

    # Prepare data for plotting
    results = []

    for marker_type, marker_genes in markers.items():
        # Find marker genes in data
        marker_indices = []
        found_markers = []

        for marker in marker_genes:
            if marker in adata.var_names:
                marker_indices.append(np.where(adata.var_names == marker)[0][0])
                found_markers.append(marker)

        if not marker_indices:
            warnings.warn(f"No markers found for {marker_type}")
            continue

        # Calculate percentages for each layer
        for layer_name in layer_keys:
            if layer_name == 'X':
                expression_data = adata.X
            else:
                if layer_name not in adata.layers:
                    warnings.warn(f"Layer '{layer_name}' not found")
                    continue
                expression_data = adata.layers[layer_name]

            # Convert to dense if sparse
            if hasattr(expression_data, 'toarray'):
                expression_data = expression_data.toarray()

            for cluster_name in cluster_names:
                cluster_mask = clusters == cluster_name
                if not np.any(cluster_mask):
                    continue

                cluster_data = expression_data[cluster_mask][:, marker_indices]

                # Check if any marker is expressed above threshold
                expressing_cells = np.any(cluster_data >= threshold, axis=1)
                percentage = np.mean(expressing_cells) * 100

                results.append({
                    'marker_type': marker_type,
                    'cluster': cluster_name,
                    'layer': layer_name,
                    'percentage': percentage
                })

    # Create DataFrame
    df = pd.DataFrame(results)

    if df.empty:
        raise ValueError("No valid marker data found")

    # Create subplots
    fig, axes = plt.subplots(
        nrows=int(np.ceil(len(markers) / ncol)),
        ncols=ncol,
        figsize=figsize,
        squeeze=False
    )
    axes = axes.flatten()

    # Plot each marker type
    for i, (marker_type, marker_genes) in enumerate(markers.items()):
        ax = axes[i]

        marker_data = df[df['marker_type'] == marker_type]

        if marker_data.empty:
            ax.text(0.5, 0.5, f'No data for\n{marker_type}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Create bar plot
        pivot_data = marker_data.pivot(index='cluster', columns='layer', values='percentage')

        pivot_data.plot(kind='bar', ax=ax, width=0.8)

        ax.set_title(f'{marker_type}')
        ax.set_xlabel('Cell Type' if group_clusters else 'Cluster')
        ax.set_ylabel('% Cells Expressing Markers')
        ax.set_ylim(0, 100)

        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)

        # Add percentage labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', rotation=0, size=8)

        # Legend
        if len(layer_keys) > 1:
            ax.legend(title='Data', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.legend().set_visible(False)

    # Hide unused subplots
    for i in range(len(markers), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')

    return fig


def plot_marker_expression(
        adata,
        markers: Union[List[str], Dict[str, List[str]]],
        group_clusters: Optional[Dict] = None,
        layer_keys: List[str] = ['X', 'decontX_counts'],
        cluster_key: str = 'decontX_clusters',
        log_transform: bool = False,
        ncol: Optional[int] = None,
        figsize: Optional[tuple] = None,
        plot_dots: bool = False,
        dot_size: float = 0.1,
        save: Optional[str] = None
):
    """
    Plot marker gene expression distributions.
    Equivalent to R's plotDecontXMarkerExpression.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    markers : list or dict
        Marker genes to plot
    group_clusters : dict, optional
        Dictionary to regroup clusters
    layer_keys : list
        Layers to compare
    cluster_key : str
        Key for cluster labels
    log_transform : bool
        Whether to log1p transform expression
    ncol : int, optional
        Number of columns in grid
    figsize : tuple, optional
        Figure size
    plot_dots : bool
        Whether to add dots to violin plots
    dot_size : float
        Size of dots
    save : str, optional
        Path to save figure
    """
    # Handle markers format
    if isinstance(markers, list):
        markers = {f'Marker_{i + 1}': [m] for i, m in enumerate(markers)}
    elif isinstance(markers, dict):
        # Flatten marker dict to list
        all_markers = []
        marker_labels = []
        for marker_type, marker_list in markers.items():
            all_markers.extend(marker_list)
            marker_labels.extend([marker_type] * len(marker_list))

    if ncol is None:
        ncol = min(3, len(all_markers))

    if figsize is None:
        nrow = int(np.ceil(len(all_markers) / ncol))
        figsize = (4 * ncol, 3 * nrow)

    # Get cluster information
    clusters = adata.obs[cluster_key].values

    # Apply cluster groupings
    if group_clusters is not None:
        cluster_mapping = {}
        for group_name, cluster_list in group_clusters.items():
            for cluster in cluster_list:
                cluster_mapping[cluster] = group_name
        clusters = np.array([cluster_mapping.get(c, c) for c in clusters])

    # Prepare data
    plot_data = []

    for marker in all_markers:
        if marker not in adata.var_names:
            warnings.warn(f"Marker '{marker}' not found in gene names")
            continue

        marker_idx = np.where(adata.var_names == marker)[0][0]

        for layer_name in layer_keys:
            if layer_name == 'X':
                expression_data = adata.X[:, marker_idx]
            else:
                if layer_name not in adata.layers:
                    continue
                expression_data = adata.layers[layer_name][:, marker_idx]

            # Convert to dense if needed
            if hasattr(expression_data, 'toarray'):
                expression_data = expression_data.toarray().flatten()

            # Log transform if requested
            if log_transform:
                expression_data = np.log1p(expression_data)

            # Add to plot data
            for i, (expr, cluster) in enumerate(zip(expression_data, clusters)):
                plot_data.append({
                    'marker': marker,
                    'expression': expr,
                    'cluster': cluster,
                    'layer': layer_name,
                    'cell_id': i
                })

    # Create DataFrame
    df = pd.DataFrame(plot_data)

    if df.empty:
        raise ValueError("No expression data found for markers")

    # Create subplots
    fig, axes = plt.subplots(
        nrows=int(np.ceil(len(all_markers) / ncol)),
        ncols=ncol,
        figsize=figsize,
        squeeze=False
    )
    axes = axes.flatten()

    # Plot each marker
    for i, marker in enumerate(all_markers):
        if marker not in df['marker'].values:
            continue

        ax = axes[i]
        marker_data = df[df['marker'] == marker]

        # Create violin plot
        if len(layer_keys) > 1:
            sns.violinplot(
                data=marker_data,
                x='cluster',
                y='expression',
                hue='layer',
                ax=ax,
                inner=None if plot_dots else 'quartiles'
            )
        else:
            sns.violinplot(
                data=marker_data,
                x='cluster',
                y='expression',
                ax=ax,
                inner=None if plot_dots else 'quartiles'
            )

        # Add dots if requested
        if plot_dots:
            if len(layer_keys) > 1:
                sns.stripplot(
                    data=marker_data,
                    x='cluster',
                    y='expression',
                    hue='layer',
                    ax=ax,
                    size=dot_size,
                    alpha=0.6,
                    dodge=True
                )
            else:
                sns.stripplot(
                    data=marker_data,
                    x='cluster',
                    y='expression',
                    ax=ax,
                    size=dot_size,
                    alpha=0.6
                )

        # Formatting
        ax.set_title(marker)
        ax.set_xlabel('')
        ax.set_ylabel('Expression (log1p)' if log_transform else 'Expression')
        ax.tick_params(axis='x', rotation=45)

        # Remove duplicate legends
        if len(layer_keys) > 1 and i > 0:
            ax.legend().set_visible(False)

    # Hide unused subplots
    for i in range(len(all_markers), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')

    return fig


def plot_contamination_comparison(
        adata,
        contamination_key='decontX_contamination',
        cluster_key='decontX_clusters',
        figsize=(12, 8),
        save=None
):
    """
    Create comprehensive contamination comparison plots.
    """
    contamination = adata.obs[contamination_key].values
    clusters = adata.obs[cluster_key].values

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Histogram of contamination
    axes[0, 0].hist(contamination, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Contamination')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Contamination')
    axes[0, 0].axvline(np.median(contamination), color='red', linestyle='--',
                       label=f'Median: {np.median(contamination):.3f}')
    axes[0, 0].legend()

    # Box plot by cluster
    df = pd.DataFrame({'contamination': contamination, 'cluster': clusters})
    sns.boxplot(data=df, x='cluster', y='contamination', ax=axes[0, 1])
    axes[0, 1].set_title('Contamination by Cluster')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Scatter plot: total UMIs vs contamination
    total_counts = np.array(adata.X.sum(axis=1)).flatten()
    axes[1, 0].scatter(total_counts, contamination, alpha=0.6, s=1)
    axes[1, 0].set_xlabel('Total UMI Counts')
    axes[1, 0].set_ylabel('Contamination')
    axes[1, 0].set_title('Contamination vs Total Counts')

    # Summary statistics
    stats_text = f"""
    Summary Statistics:

    Mean: {np.mean(contamination):.3f}
    Median: {np.median(contamination):.3f}
    Std: {np.std(contamination):.3f}

    Min: {np.min(contamination):.3f}
    Max: {np.max(contamination):.3f}

    Cells > 50% contamination: {np.sum(contamination > 0.5):,}
    Cells > 70% contamination: {np.sum(contamination > 0.7):,}
    """

    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Summary Statistics')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')

    return fig


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
Example usage:

# After running decontx
import decontx

# Run DecontX
decontx.decontx(adata, copy=False)

# Plot contamination on UMAP
fig1 = plot_contamination_umap(adata)

# Plot marker percentages
markers = {
    'T_cells': ['CD3D', 'CD3E'],
    'B_cells': ['CD79A', 'CD79B', 'MS4A1'],
    'Monocytes': ['LYZ', 'S100A8']
}

fig2 = plot_marker_percentage(
    adata, 
    markers,
    layer_keys=['X', 'decontX_counts']
)

# Plot marker expression
fig3 = plot_marker_expression(
    adata,
    markers,
    log_transform=True,
    plot_dots=True
)

# Comprehensive comparison
fig4 = plot_contamination_comparison(adata)
"""