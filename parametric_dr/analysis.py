"""Dimensionality selection and quality metric analysis.

Provides tools for estimating intrinsic dimensionality and sweeping across
different target dimensionalities to find optimal embeddings.
"""

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from .metrics import trustworthiness, continuity, shepard_correlation
from .utils import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def estimate_intrinsic_dim(X: np.ndarray, method: str = "twonn") -> Dict[str, Any]:
    """Estimate data manifold intrinsic dimensionality.

    Wraps skdim (optional dependency) to estimate the dimensionality of the
    underlying manifold. Provides a principled upper bound for useful
    target dimensionalities.

    Parameters
    ----------
    X : 2-d array (N, D)
        Input data matrix.
    method : str
        Estimation method. Currently only "twonn" (TwoNN) is supported.

    Returns
    -------
    dict with keys:
        "estimate": float
            Estimated intrinsic dimensionality.
        "method": str
            The method used.

    Raises
    ------
    ImportError
        If skdim is not installed.

    References
    ----------
    Facco et al. (2017). Estimating the intrinsic dimension of datasets by
    a minimal neighborhood information.
    """
    try:
        from skdim.id import TwoNN
    except ImportError:
        raise ImportError(
            "skdim is required for estimate_intrinsic_dim but not installed.\n"
            "Install with: pip install scikit-dimension"
        )

    if method != "twonn":
        raise ValueError(f"Unsupported method: {method}. Only 'twonn' is supported.")

    estimator = TwoNN()
    estimate = estimator.fit(X).dimension_
    return {"estimate": estimate, "method": method}


def _elbow_detection(metric_curve: np.ndarray) -> int:
    """Detect elbow point using perpendicular distance method.

    Finds the point with maximum perpendicular distance from the line
    connecting the first and last points of the curve.

    Parameters
    ----------
    metric_curve : 1-d array
        Metric values across dimensionalities.

    Returns
    -------
    int
        Index of the elbow point.
    """
    n = len(metric_curve)
    if n <= 2:
        return n - 1

    # Start and end points define the reference line
    p1 = np.array([0, metric_curve[0]])
    p2 = np.array([n - 1, metric_curve[-1]])

    # Vector along the line
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-8:
        # Flat curve; return the last point
        return n - 1

    # For each point, compute perpendicular distance to the line
    distances = np.zeros(n)
    for i in range(n):
        p = np.array([i, metric_curve[i]])
        # Vector from p1 to p
        p1_to_p = p - p1
        # Project onto line
        proj_len = np.dot(p1_to_p, line_vec) / line_len
        proj_len = np.clip(proj_len, 0, line_len)
        proj_point = p1 + (proj_len / line_len) * line_vec
        # Perpendicular distance
        distances[i] = np.linalg.norm(p - proj_point)

    return np.argmax(distances)


def sweep_dims(
    X: np.ndarray,
    model_class,
    dims: Tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10),
    k: int = 10,
    n_subsample: Optional[int] = 500,
    metric: str = "euclidean",
    metric_kwargs: Optional[Dict[str, Any]] = None,
    plot: bool = True,
    **model_kwargs,
) -> Dict[str, Any]:
    """Train DR models at multiple dimensionalities and evaluate quality.

    Trains the given model class at each dimensionality in `dims`, evaluates
    quality metrics (trustworthiness, continuity, Shepard correlation), and
    detects the elbow point in the trustworthiness curve.

    Parameters
    ----------
    X : 2-d array (N, D_high)
        Input high-dimensional data.
    model_class : class
        Parametric DR model class (e.g., Parametric_UMAP).
        Must inherit from ParametricDR and implement fit().
    dims : tuple of int
        Target dimensionalities to evaluate. Default: (2, 3, 4, 5, 6, 8, 10).
    k : int
        Number of neighbors for trustworthiness and continuity metrics.
    n_subsample : int or None
        Number of points to subsample for metric computation (O(N²) cost).
        If None, use all points. Default: 500 (PLOS CompBio 2024 validated).
    metric : str or callable
        Distance metric for the high-dimensional space. Accepts any metric
        supported by scipy.spatial.distance.pdist, including "euclidean",
        "cosine", "cityblock", "correlation", "minkowski", "chebyshev",
        "mahalanobis", or a custom callable. Default: "euclidean".
    metric_kwargs : dict or None
        Extra kwargs forwarded to pdist (e.g. ``{"p": 3}`` for minkowski,
        ``{"VI": inv_cov}`` for mahalanobis).
    plot : bool
        If True, return a matplotlib Figure with T/C/Shepard vs dim plots.
    **model_kwargs
        Keyword arguments to pass to model_class constructor.
        Must include num_inputs and any other required hyperparameters.

    Returns
    -------
    dict with keys:
        "results": pd.DataFrame
            Columns: dim, trustworthiness, continuity, shepard_correlation.
        "elbow_dim": int
            Dimensionality detected as the elbow point.
        "figure": matplotlib Figure or None
            Visualization of metrics vs dimensionality (if plot=True).

    Examples
    --------
    >>> from parametric_dr import Parametric_UMAP, sweep_dims
    >>> from sklearn.datasets import load_digits
    >>> X, _ = load_digits(return_X_y=True)
    >>> results = sweep_dims(
    ...     X, Parametric_UMAP,
    ...     dims=[2, 3, 4, 5],
    ...     num_inputs=64,
    ...     num_epochs=20,
    ...     plot=True
    ... )
    >>> print(results["results"])
    >>> print("Elbow dim:", results["elbow_dim"])
    """
    dims = tuple(dims) if not isinstance(dims, tuple) else dims
    N = X.shape[0]

    # Subsample strategy
    if n_subsample is not None and n_subsample < N:
        subsample_indices = np.random.choice(N, n_subsample, replace=False)
        X_subsample = X[subsample_indices]
    else:
        subsample_indices = None
        X_subsample = X

    results_list = []

    for d in dims:
        logger.debug("Training at dim=%d...", d)

        # Instantiate and train model
        model = model_class(num_outputs=d, **model_kwargs)
        model.fit(X)

        # Transform
        embedding = model.transform(X)

        # Subsample if needed
        if subsample_indices is not None:
            X_subsample_high = X[subsample_indices]
            embedding_subsample = embedding[subsample_indices]
        else:
            X_subsample_high = X
            embedding_subsample = embedding

        # Compute metrics
        mkw = metric_kwargs or {}
        t = trustworthiness(X_subsample_high, embedding_subsample, k=k, metric=metric, **mkw)
        c = continuity(X_subsample_high, embedding_subsample, k=k, metric=metric, **mkw)
        s = shepard_correlation(X_subsample_high, embedding_subsample, metric=metric, **mkw)

        logger.info("T=%.4f, C=%.4f, S=%.4f", t, c, s)

        results_list.append(
            {
                "dim": d,
                "trustworthiness": t,
                "continuity": c,
                "shepard_correlation": s,
            }
        )

    results_df = pd.DataFrame(results_list)

    # Detect elbow
    trust_curve = results_df["trustworthiness"].values
    elbow_idx = _elbow_detection(trust_curve)
    elbow_dim = int(results_df.iloc[elbow_idx]["dim"])

    logger.info("Elbow detected at dim=%d", elbow_dim)

    # Optional plot
    figure = None
    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skipping plot.")
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # Trustworthiness
            axes[0].plot(results_df["dim"], results_df["trustworthiness"], "o-")
            axes[0].axvline(elbow_dim, color="red", linestyle="--", label=f"Elbow (d={elbow_dim})")
            axes[0].set_xlabel("Dimensionality")
            axes[0].set_ylabel("Trustworthiness")
            axes[0].set_title("Trustworthiness vs Dim")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Continuity
            axes[1].plot(results_df["dim"], results_df["continuity"], "o-", color="green")
            axes[1].axvline(elbow_dim, color="red", linestyle="--", label=f"Elbow (d={elbow_dim})")
            axes[1].set_xlabel("Dimensionality")
            axes[1].set_ylabel("Continuity")
            axes[1].set_title("Continuity vs Dim")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Shepard correlation
            axes[2].plot(results_df["dim"], results_df["shepard_correlation"], "o-", color="orange")
            axes[2].axvline(elbow_dim, color="red", linestyle="--", label=f"Elbow (d={elbow_dim})")
            axes[2].set_xlabel("Dimensionality")
            axes[2].set_ylabel("Shepard Correlation")
            axes[2].set_title("Shepard Correlation vs Dim")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            figure = fig

    return {
        "results": results_df,
        "elbow_dim": elbow_dim,
        "figure": figure,
    }
