"""Numpy utility functions for t-SNE: beta calculation, distances, perplexity."""

import numpy as np

LOGGER_NAME = "parametric_dr"


def Hbeta_vec(distances: np.ndarray, betas: np.ndarray):
    """Vectorized Gaussian kernel entropy computation.

    Parameters
    ----------
    distances : 2-d array (N, N)
        Square matrix of squared Euclidean distances.
    betas : 1-d array (N,)
        Precisions of the Gaussian kernel. beta = 1 / (2 * sigma^2).

    Returns
    -------
    H : 1-d array (N,)
        Entropy of each point.
    p_matr : 2-d array (N, N)
        Probability matrix.
    """
    beta_matr = betas[:, np.newaxis] * np.ones_like(distances)
    p_matr = np.exp(-distances * beta_matr)
    sumP = np.sum(p_matr, axis=1)
    H = np.log(sumP) + (betas * np.sum(distances * p_matr, axis=1)) / sumP
    p_matr = p_matr / (sumP[:, np.newaxis] * np.ones_like(p_matr))
    return H, p_matr


def Hbeta_scalar(distances: np.ndarray, beta: float):
    """Scalar Gaussian kernel entropy for a single point.

    Parameters
    ----------
    distances : 1-d array (N,)
        Squared distances from the current point to all others.
    beta : float
        Precision of the Gaussian kernel.

    Returns
    -------
    H : float
        Entropy.
    p_matr : 1-d array (N,)
        Probability vector.
    """
    p_matr = np.exp(-distances * beta)
    sumP = np.sum(p_matr)
    H = np.log(sumP) + (beta * np.sum(distances * p_matr)) / sumP
    p_matr = p_matr / sumP
    return H, p_matr


def get_squared_cross_diff_np(x: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances.

    Z_ij = ||x_i - x_j||^2

    Parameters
    ----------
    x : 2-d array (N, D)

    Returns
    -------
    Z_ij : 2-d array (N, N)
    """
    batch_size = x.shape[0]
    expanded = np.expand_dims(x, 1)
    tiled = np.tile(expanded, np.stack([1, batch_size, 1]))
    tiled_trans = np.transpose(tiled, axes=[1, 0, 2])
    diffs = tiled - tiled_trans
    return np.sum(np.square(diffs), axis=2)


def get_Lmax(num_points: int) -> int:
    """Max scale level for multiscale perplexities."""
    return int(np.floor(np.log2(num_points / 4.0)))


def get_multiscale_perplexities(num_points: int) -> np.ndarray:
    """Generate perplexity range from data size.

    Parameters
    ----------
    num_points : int

    Returns
    -------
    perplexities : 1-d array

    References
    ----------
    Lee, Peluffo-Ordonez, & Verleysen (2015).
    Multiscale stochastic neighbor embedding: Towards parameter-free
    dimensionality reduction.
    """
    Lmax = get_Lmax(num_points)
    l_vals = np.arange(2, Lmax)
    return 2.0 ** l_vals


def compute_knn_graph(X: np.ndarray, n_neighbors: int):
    """Compute k-nearest neighbor graph using sklearn.

    Parameters
    ----------
    X : 2-d array (N, D)
    n_neighbors : int

    Returns
    -------
    indices : 2-d array (N, n_neighbors)
        KNN indices (excluding self).
    distances : 2-d array (N, n_neighbors)
        KNN distances (excluding self).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    # Exclude self (first column, distance 0)
    return indices[:, 1:], distances[:, 1:]


def calc_betas_loop(
    indata: np.ndarray, perplexity: float, tol: float = 1e-4, max_tries: int = 50
):
    """Binary search for Gaussian kernel widths matching desired perplexity.

    Parameters
    ----------
    indata : 2-d array (N, D)
    perplexity : float
    tol : float
        Absolute tolerance for entropy convergence.
    max_tries : int

    Returns
    -------
    betas : 1-d array (N,)
    Hs : 1-d array (N,)
        Final entropy at each point.
    p_matr : 2-d array (N, N)
        Probability matrix.
    """
    logPx = np.log(perplexity)
    num_samps = indata.shape[0]

    beta_init = np.ones([num_samps], dtype=float)
    betas = beta_init.copy()
    p_matr = np.zeros([num_samps, num_samps])
    Hs = beta_init.copy()

    in_sq_diffs = get_squared_cross_diff_np(indata)

    for ss in range(num_samps):
        betamin = -np.inf
        betamax = np.inf

        Di = in_sq_diffs[ss, :]
        H, thisPx = Hbeta_scalar(Di, betas[ss])
        Hdiff = 100 * tol

        tries = 0
        while abs(Hdiff) > tol and tries < max_tries:
            H, thisPx = Hbeta_scalar(Di, betas[ss])
            Hdiff = H - logPx
            tries += 1

            if Hdiff > 0.0:
                betamin = betas[ss]
                if np.isinf(betamax):
                    betas[ss] = betas[ss] * 2.0
                else:
                    betas[ss] = (betas[ss] + betamax) / 2.0
            else:
                betamax = betas[ss]
                if np.isinf(betamin):
                    betas[ss] = betas[ss] / 2.0
                else:
                    betas[ss] = (betas[ss] + betamin) / 2.0

        p_matr[ss, :] = thisPx
        Hs[ss] = H

    return betas, Hs, p_matr
