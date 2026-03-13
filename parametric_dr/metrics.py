"""Dimensionality reduction quality metrics.

All functions are pure numpy and accept raw arrays, so they work
with any DR method (not just parametric t-SNE). All internal
computation uses float32 to reduce memory usage.
"""

from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform

_FLOAT = np.float32
_INT = np.int32


def _as_float32(X: np.ndarray) -> np.ndarray:
    """Ensure array is float32 (no-op if already correct dtype)."""
    return np.asarray(X, dtype=_FLOAT)


def _pdist32(X: np.ndarray, metric: str = "euclidean", **kw) -> np.ndarray:
    """pdist returning float32."""
    return pdist(_as_float32(X), metric=metric, **kw).astype(_FLOAT)


def _ranks(D: np.ndarray) -> np.ndarray:
    """Rank matrix (int32) from a distance matrix."""
    return np.argsort(np.argsort(D, axis=1), axis=1).astype(_INT)


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
    D : 2-d array (N, N), float32
        Symmetric distance matrix with zero diagonal.
    """
    return squareform(_pdist32(X, metric=metric, **metric_kwargs))


def _knn_indices_from_D(D: np.ndarray, k: int) -> np.ndarray:
    """Return indices of k nearest neighbors from a precomputed distance matrix.

    Parameters
    ----------
    D : 2-d array (N, N)
        Pairwise distance matrix.
    k : int

    Returns
    -------
    indices : 2-d array (N, k), int32
        Row i contains the indices of point i's k nearest neighbors
        (excluding itself), sorted by distance.
    """
    D = D.astype(_FLOAT, copy=True)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k].astype(_INT)


def trustworthiness(
    X_high: np.ndarray, X_low: np.ndarray, k: int = 10,
    metric: str = "euclidean",
    D_high: Optional[np.ndarray] = None,
    D_low: Optional[np.ndarray] = None,
    **metric_kwargs,
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
    D_high : 2-d array (N, N), optional
        Precomputed high-dim distance matrix. Skips recomputation when
        provided (``metric`` and ``metric_kwargs`` are ignored).
    D_low : 2-d array (N, N), optional
        Precomputed low-dim (Euclidean) distance matrix.
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

    if D_high is None:
        D_high = _distance_matrix(X_high, metric=metric, **metric_kwargs)
    if D_low is None:
        D_low = _distance_matrix(X_low, "euclidean")

    nn_high = _knn_indices_from_D(D_high, k)
    nn_low = _knn_indices_from_D(D_low, k)

    ranks_high = _ranks(D_high)

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
    metric: str = "euclidean",
    D_high: Optional[np.ndarray] = None,
    D_low: Optional[np.ndarray] = None,
    **metric_kwargs,
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
    D_high : 2-d array (N, N), optional
        Precomputed high-dim distance matrix. Skips recomputation when
        provided (``metric`` and ``metric_kwargs`` are ignored).
    D_low : 2-d array (N, N), optional
        Precomputed low-dim (Euclidean) distance matrix.
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

    if D_high is None:
        D_high = _distance_matrix(X_high, metric=metric, **metric_kwargs)
    if D_low is None:
        D_low = _distance_matrix(X_low, "euclidean")

    nn_high = _knn_indices_from_D(D_high, k)
    nn_low = _knn_indices_from_D(D_low, k)

    ranks_low = _ranks(D_low)

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
    metric: str = "euclidean",
    D_high: Optional[np.ndarray] = None,
    D_low: Optional[np.ndarray] = None,
    **metric_kwargs,
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
    D_high : 2-d array (N, N), optional
        Precomputed high-dim distance matrix. Skips recomputation when
        provided (``metric`` and ``metric_kwargs`` are ignored).
    D_low : 2-d array (N, N), optional
        Precomputed low-dim (Euclidean) distance matrix.
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

    if D_high is None:
        D_high = _distance_matrix(X_high, metric=metric, **metric_kwargs)
    if D_low is None:
        D_low = _distance_matrix(X_low, "euclidean")

    nn_high = _knn_indices_from_D(D_high, k)
    nn_low = _knn_indices_from_D(D_low, k)

    overlap = 0.0
    for i in range(n):
        overlap += len(set(nn_high[i]) & set(nn_low[i]))

    return overlap / (n * k)


def shepard_correlation(
    X_high: np.ndarray, X_low: np.ndarray,
    metric: str = "euclidean",
    D_high: Optional[np.ndarray] = None,
    D_low: Optional[np.ndarray] = None,
    **metric_kwargs,
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
    D_high : 2-d array (N, N), optional
        Precomputed high-dim distance matrix. Skips recomputation when
        provided (``metric`` and ``metric_kwargs`` are ignored).
    D_low : 2-d array (N, N), optional
        Precomputed low-dim (Euclidean) distance matrix.
    **metric_kwargs
        Extra kwargs forwarded to pdist (e.g. ``p=3`` for minkowski,
        ``VI=inv_cov`` for mahalanobis).

    Returns
    -------
    r : float in [-1, 1]
        Pearson correlation coefficient. 1.0 is perfect preservation.
    """
    # Shepard needs condensed (vector) form; extract from square if provided
    if D_high is not None:
        d_high = _as_float32(squareform(D_high, checks=False))
    else:
        d_high = _pdist32(X_high, metric=metric, **metric_kwargs)

    if D_low is not None:
        d_low = _as_float32(squareform(D_low, checks=False))
    else:
        d_low = _pdist32(X_low)

    return float(np.corrcoef(d_high, d_low)[0, 1])


def trajectory_smoothness(
    X_low: np.ndarray,
    normalize: bool = True,
    subsample: int = 500,
    seed: int = 0,
) -> float:
    """Mean squared step size between consecutive embedding points.

    Measures how smooth a trajectory is in the embedding space.
    Points are assumed to be in temporal order (row 0 is the first
    timepoint, row 1 the second, etc.).

    Parameters
    ----------
    X_low : 2-d array (N, D)
        Embedding coordinates in temporal order.
    normalize : bool
        If True, divide by the mean squared pairwise distance so that
        the metric is scale-invariant and comparable across methods.
        If False, return the raw mean squared step size.
    subsample : int
        Number of points to subsample when estimating the mean pairwise
        distance for normalization. Ignored when ``normalize=False``.
    seed : int
        Random seed for the subsample.

    Returns
    -------
    score : float >= 0
        Lower is smoother. When normalized, the value is the ratio of
        the mean step size to the mean pairwise distance.
    """
    X_low = _as_float32(X_low)
    diffs = np.diff(X_low, axis=0)
    mean_step = float(np.mean(np.sum(diffs ** 2, axis=1)))

    if not normalize:
        return mean_step

    n = len(X_low)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(subsample, n), replace=False)
    subset = X_low[idx]
    sq_dists = _pdist32(subset) ** 2
    mean_pdist = float(np.mean(sq_dists))

    if mean_pdist == 0:
        return 0.0
    return mean_step / mean_pdist


def compute_metrics(
    X_high: np.ndarray, X_low: np.ndarray, k: int = 10,
    metric: str = "euclidean", **metric_kwargs,
) -> dict:
    """Compute all quality metrics in one pass, sharing distance matrices.

    Equivalent to calling trustworthiness, continuity,
    neighborhood_preservation, and shepard_correlation individually,
    but computes each distance matrix only once.

    Parameters
    ----------
    X_high : 2-d array (N, D_high)
    X_low : 2-d array (N, D_low)
    k : int
        Number of neighbors for T, C, and neighborhood preservation.
    metric : str or callable
        Distance metric for X_high (see trustworthiness for details).
        X_low always uses Euclidean.
    **metric_kwargs
        Extra kwargs forwarded to pdist.

    Returns
    -------
    results : dict
        Keys: ``"trustworthiness"``, ``"continuity"``,
        ``"neighborhood_preservation"``, ``"shepard_correlation"``.
    """
    n = X_high.shape[0]
    k = min(k, n - 1)

    d_high_condensed = _pdist32(X_high, metric=metric, **metric_kwargs)
    d_low_condensed = _pdist32(X_low)

    D_high = squareform(d_high_condensed)
    D_low = squareform(d_low_condensed)

    nn_high = _knn_indices_from_D(D_high, k)
    nn_low = _knn_indices_from_D(D_low, k)

    ranks_high = _ranks(D_high)
    ranks_low = _ranks(D_low)

    t_penalty = 0.0
    c_penalty = 0.0
    overlap = 0.0
    for i in range(n):
        high_set = set(nn_high[i])
        low_set = set(nn_low[i])

        for j in nn_low[i]:
            if j not in high_set:
                t_penalty += ranks_high[i, j] - k

        for j in nn_high[i]:
            if j not in low_set:
                c_penalty += ranks_low[i, j] - k

        overlap += len(high_set & low_set)

    normalization = n * k * (2 * n - 3 * k - 1)
    if normalization == 0:
        T = C = 1.0
    else:
        T = 1.0 - (2.0 / normalization) * t_penalty
        C = 1.0 - (2.0 / normalization) * c_penalty

    N = overlap / (n * k)
    S = float(np.corrcoef(d_high_condensed, d_low_condensed)[0, 1])

    return {
        "trustworthiness": T,
        "continuity": C,
        "neighborhood_preservation": N,
        "shepard_correlation": S,
    }
