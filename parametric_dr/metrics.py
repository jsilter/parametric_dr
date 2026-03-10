"""Dimensionality reduction quality metrics.

All functions are pure numpy and accept raw arrays, so they work
with any DR method (not just parametric t-SNE).
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def _distance_matrix(
    X: np.ndarray, metric: str = "euclidean", **metric_kwargs,
) -> np.ndarray:
    """Compute an NxN distance matrix.

    Parameters
    ----------
    X : 2-d array (N, D)
    metric : str or callable
        Any metric accepted by scipy.spatial.distance.pdist, including
        "euclidean", "cosine", "cityblock", "correlation", "minkowski",
        "chebyshev", "mahalanobis", or a custom callable.
    **metric_kwargs
        Extra kwargs forwarded to pdist (e.g. ``p=3`` for minkowski,
        ``VI=inv_cov`` for mahalanobis).

    Returns
    -------
    D : 2-d array (N, N)
        Symmetric distance matrix with zero diagonal.
    """
    return squareform(pdist(X, metric=metric, **metric_kwargs))


def _knn_indices_from_D(D: np.ndarray, k: int) -> np.ndarray:
    """Return indices of k nearest neighbors from a precomputed distance matrix.

    Parameters
    ----------
    D : 2-d array (N, N)
        Pairwise distance matrix.
    k : int

    Returns
    -------
    indices : 2-d array (N, k)
        Row i contains the indices of point i's k nearest neighbors
        (excluding itself), sorted by distance.
    """
    D = D.astype(float, copy=True)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]


def trustworthiness(
    X_high: np.ndarray, X_low: np.ndarray, k: int = 10,
    metric: str = "euclidean", **metric_kwargs,
) -> float:
    """Trustworthiness: are low-dim neighbors also neighbors in high-dim?

    Penalizes "false neighbors" that appear close in the embedding
    but were far apart in the original space.

    Parameters
    ----------
    X_high : 2-d array (N, D_high)
    X_low : 2-d array (N, D_low)
    k : int
        Number of neighbors to consider.
    metric : str or callable
        Distance metric for X_high. Accepts any metric supported by
        scipy.spatial.distance.pdist, including "euclidean", "cosine",
        "cityblock", "correlation", "minkowski", "chebyshev",
        "mahalanobis", or a custom callable. X_low always uses Euclidean.
    **metric_kwargs
        Extra kwargs forwarded to pdist (e.g. ``p=3`` for minkowski,
        ``VI=inv_cov`` for mahalanobis).

    Returns
    -------
    T : float in (0, 1]
        1.0 means all low-dim neighbors were also high-dim neighbors.

    References
    ----------
    Venna & Kaski (2006). Local multidimensional scaling.
    """
    n = X_high.shape[0]
    k = min(k, n - 1)

    D_high = _distance_matrix(X_high, metric=metric, **metric_kwargs)
    D_low = _distance_matrix(X_low, "euclidean")

    nn_high = _knn_indices_from_D(D_high, k)
    nn_low = _knn_indices_from_D(D_low, k)

    ranks_high = np.argsort(np.argsort(D_high, axis=1), axis=1)

    penalty = 0.0
    for i in range(n):
        high_set = set(nn_high[i])
        for j in nn_low[i]:
            if j not in high_set:
                # rank is 1-indexed for the formula
                penalty += ranks_high[i, j] - k

    normalization = n * k * (2 * n - 3 * k - 1)
    if normalization == 0:
        return 1.0
    return 1.0 - (2.0 / normalization) * penalty


def continuity(
    X_high: np.ndarray, X_low: np.ndarray, k: int = 10,
    metric: str = "euclidean", **metric_kwargs,
) -> float:
    """Continuity: are high-dim neighbors preserved in the low-dim embedding?

    Penalizes "missing neighbors" that were close in the original space
    but ended up far apart in the embedding.

    Parameters
    ----------
    X_high : 2-d array (N, D_high)
    X_low : 2-d array (N, D_low)
    k : int
        Number of neighbors to consider.
    metric : str or callable
        Distance metric for X_high. Accepts any metric supported by
        scipy.spatial.distance.pdist, including "euclidean", "cosine",
        "cityblock", "correlation", "minkowski", "chebyshev",
        "mahalanobis", or a custom callable. X_low always uses Euclidean.
    **metric_kwargs
        Extra kwargs forwarded to pdist (e.g. ``p=3`` for minkowski,
        ``VI=inv_cov`` for mahalanobis).

    Returns
    -------
    C : float in (0, 1]
        1.0 means all high-dim neighbors are also low-dim neighbors.

    References
    ----------
    Venna & Kaski (2006). Local multidimensional scaling.
    """
    n = X_high.shape[0]
    k = min(k, n - 1)

    D_high = _distance_matrix(X_high, metric=metric, **metric_kwargs)
    D_low = _distance_matrix(X_low, "euclidean")

    nn_high = _knn_indices_from_D(D_high, k)
    nn_low = _knn_indices_from_D(D_low, k)

    ranks_low = np.argsort(np.argsort(D_low, axis=1), axis=1)

    penalty = 0.0
    for i in range(n):
        low_set = set(nn_low[i])
        for j in nn_high[i]:
            if j not in low_set:
                penalty += ranks_low[i, j] - k

    normalization = n * k * (2 * n - 3 * k - 1)
    if normalization == 0:
        return 1.0
    return 1.0 - (2.0 / normalization) * penalty


def neighborhood_preservation(
    X_high: np.ndarray, X_low: np.ndarray, k: int = 10,
    metric: str = "euclidean", **metric_kwargs,
) -> float:
    """Fraction of k-nearest neighbors preserved between spaces.

    Parameters
    ----------
    X_high : 2-d array (N, D_high)
    X_low : 2-d array (N, D_low)
    k : int
    metric : str or callable
        Distance metric for X_high. Accepts any metric supported by
        scipy.spatial.distance.pdist, including "euclidean", "cosine",
        "cityblock", "correlation", "minkowski", "chebyshev",
        "mahalanobis", or a custom callable. X_low always uses Euclidean.
    **metric_kwargs
        Extra kwargs forwarded to pdist (e.g. ``p=3`` for minkowski,
        ``VI=inv_cov`` for mahalanobis).

    Returns
    -------
    score : float in [0, 1]
        Mean fraction of k-NN overlap across all points.
    """
    n = X_high.shape[0]
    k = min(k, n - 1)

    D_high = _distance_matrix(X_high, metric=metric, **metric_kwargs)
    D_low = _distance_matrix(X_low, "euclidean")

    nn_high = _knn_indices_from_D(D_high, k)
    nn_low = _knn_indices_from_D(D_low, k)

    overlap = 0.0
    for i in range(n):
        overlap += len(set(nn_high[i]) & set(nn_low[i]))

    return overlap / (n * k)


def shepard_correlation(
    X_high: np.ndarray, X_low: np.ndarray,
    metric: str = "euclidean", **metric_kwargs,
) -> float:
    """Pearson correlation between pairwise distances (Shepard diagram).

    Measures global structure preservation. High correlation means
    pairwise distance relationships are well preserved.

    Parameters
    ----------
    X_high : 2-d array (N, D_high)
    X_low : 2-d array (N, D_low)
    metric : str or callable
        Distance metric for X_high. Accepts any metric supported by
        scipy.spatial.distance.pdist, including "euclidean", "cosine",
        "cityblock", "correlation", "minkowski", "chebyshev",
        "mahalanobis", or a custom callable. X_low always uses Euclidean.
    **metric_kwargs
        Extra kwargs forwarded to pdist (e.g. ``p=3`` for minkowski,
        ``VI=inv_cov`` for mahalanobis).

    Returns
    -------
    r : float in [-1, 1]
        Pearson correlation coefficient. 1.0 is perfect preservation.
    """
    d_high = pdist(X_high, metric=metric, **metric_kwargs)
    d_low = pdist(X_low)
    return float(np.corrcoef(d_high, d_low)[0, 1])
