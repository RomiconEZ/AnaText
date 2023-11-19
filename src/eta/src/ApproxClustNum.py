import numpy as np
import umap.umap_ as umap
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import calinski_harabasz_score


def approximate_clusters(features: np.ndarray, min_n: int, max_n: int, verbose=False) -> np.ndarray:
    """
    Fast approximation of clusters with the use of umap and k-means
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
        predictions : np.ndarray
            array of predictions with the shape of (n_samples)
    """
    if verbose:
        print("Reducing dimensionality...")
    # Reduction of dimension of embedding vectors
    umap_features = umap.UMAP(n_components=20, n_neighbors=6, metric="cosine", verbose=verbose).fit_transform(features)

    if verbose:
        print("Started training...")

    best_predictions = np.zeros(len(features))
    best_score = -1

    # Iterating over the number of clusters
    for n_clusters in range(min_n, max_n + 1):
        if verbose:
            print("Testing n_clusters =", n_clusters)

        clusters = BisectingKMeans(n_clusters=n_clusters)

        predictions = clusters.fit(umap_features).labels_

        score = calinski_harabasz_score(umap_features, predictions)
        if verbose:
            print("Calinski-harabasz score = ", score)
        if score > best_score:
            best_score = score
            best_predictions = predictions
    return best_predictions


def approximate_n_clusters(features: np.ndarray, min_n: int = 2, max_n: int = 30) -> int:
    """
    Calculating the approximate number of clusters.
    """
    return len(np.unique(approximate_clusters(features, min_n, max_n)))
