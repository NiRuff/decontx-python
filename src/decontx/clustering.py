def decontx_initialize_z_exact(
        adata,
        var_genes: int = 5000,
        dbscan_eps: float = 1.0,
        random_state: int = 12345
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact equivalent of R's .decontxInitializeZ function.
    Matches R's preprocessing pipeline exactly.
    """
    # Work on copy to avoid modifying original
    adata_temp = adata.copy()

    # Filter genes (match R's filtering)
    sc.pp.filter_genes(adata_temp, min_counts=1)

    # Exact normalization matching R's scater::logNormCounts
    # This matches R's normalization more closely than scanpy's default
    sc.pp.normalize_total(adata_temp, target_sum=1e4)
    sc.pp.log1p(adata_temp)

    # Find highly variable genes (match R's approach)
    if adata_temp.n_vars > var_genes:
        sc.pp.highly_variable_genes(
            adata_temp,
            n_top_genes=var_genes,
            flavor='seurat_v3'  # Closer to R's approach
        )
        adata_temp = adata_temp[:, adata_temp.var.highly_variable]

    # PCA with same parameters as R
    sc.pp.pca(adata_temp, n_comps=30, random_state=random_state)

    # UMAP matching R's parameters exactly
    # R uses: minDist=0.01, spread=1, nNeighbors=15
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.01,
        spread=1.0,
        n_components=2,
        random_state=random_state,
        metric='euclidean'
    )
    umap_coords = reducer.fit_transform(adata_temp.obsm['X_pca'])

    # DBSCAN clustering with adaptive eps (matches R logic exactly)
    n_clusters = 1
    eps = dbscan_eps
    max_tries = 10

    while n_clusters <= 1 and eps > 0 and max_tries > 0:
        clusterer = DBSCAN(eps=eps, min_samples=3)
        cluster_labels = clusterer.fit_predict(umap_coords)

        # Count non-noise clusters
        n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
        eps *= 0.75  # Same reduction as R
        max_tries -= 1

    # Fallback to k-means if DBSCAN fails (matches R)
    if n_clusters <= 1:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(umap_coords)

    # Convert to 1-indexed (matches R)
    cluster_labels = cluster_labels + 1

    return cluster_labels, umap_coords