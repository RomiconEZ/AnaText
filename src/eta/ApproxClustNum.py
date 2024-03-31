import numpy as np
import umap.umap_ as umap
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import calinski_harabasz_score


def best_score_bisecting_kmeans(umap_features: np.ndarray, n_clusters: int) -> float:
    """
    Conducts three iterations of BisectingKMeans clustering on UMAP features and selects
    the iteration with the highest Calinski-Harabasz score, indicating the best cluster
    separation.

    Parameters
    ----------
    umap_features: np.ndarray
        Array of UMAP features with the shape of (n_samples, dim), where n_samples is the
        number of samples and dim is the dimensionality of the features.
    n_clusters: int
        The number of clusters to attempt to find in the data.

    Returns
    -------
    best_score: float
        The highest Calinski-Harabasz score achieved across the three iterations,
        signifying the quality of the best clustering found.
    """
    best_score = -1

    for _ in range(5):  # Выполнение 5 итераций
        clusters = BisectingKMeans(n_clusters=n_clusters)
        predictions = clusters.fit_predict(umap_features)
        score = calinski_harabasz_score(umap_features, predictions)

        if score > best_score:  # Выбор лучшей итерации
            best_score = score

    return best_score


def approximate_clusters(features: np.ndarray, min_n: int, max_n: int, verbose) -> int:
    """
    Approximation of clusters with the use of umap and k-means
    ...
    Parameters
    ----------
        features: np.ndarray
            array of embeddings with the shape of (n_samples, dim)
        min_n: int
            lower bound of n_clusters
        max_n:
            upper bound of n_clusters
    Returns
    -------
        best_num_of_clust : int
            number of clusters that gave the best metric
    """
    if verbose:
        print("Reducing dimensionality...")
    # Reduction of dimension of embedding vectors
    umap_features = umap.UMAP(n_components=20, n_neighbors=6, metric="cosine", verbose=verbose).fit_transform(features)

    if verbose:
        print("Started training...")

    scores = []

    # Iterating over the number of clusters
    for n_clusters in range(min_n, max_n + 1):
        if verbose:
            print("Testing n_clusters =", n_clusters)

        score = best_score_bisecting_kmeans(umap_features, n_clusters)
        scores.append(score)
        if verbose:
            print("Calinski-harabasz score = ", score)

    # Normalize scores and cluster numbers
    scores = np.array(scores)
    normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
    normalized_clusters = (np.arange(min_n, max_n + 1) - min_n) / (max_n - min_n)

    beta = 1
    new_metric = (
        (1 + beta**2)
        * (normalized_scores * normalized_clusters)
        / ((beta**2) * normalized_scores + normalized_clusters)
    )

    # Select the best clustering based on the new metric
    best_num_of_clust = min_n + np.argmax(new_metric)

    if verbose:
        print("Selected n_clusters =", best_num_of_clust)

    return best_num_of_clust


def approximate_n_clusters(features: np.ndarray, min_n: int = 2, max_n: int = 40, verbose=False) -> int:
    """
    Calculating the approximate number of clusters.
    """
    return approximate_clusters(features, min_n, max_n, verbose)
