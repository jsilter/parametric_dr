"""Parametric TriMap: dimensionality reduction via triplet constraints.

Learns a neural network embedding by optimizing weighted triplet
losses (anchor, near, far).

Reference:
Amid & Warmuth (2019). TriMap: Large-scale Dimensionality Reduction
Using Triplets. arXiv:1910.00204.
"""

import logging

import numpy as np
import torch

from ._base import ParametricDR
from .utils import LOGGER_NAME, compute_knn_graph

logger = logging.getLogger(LOGGER_NAME)


def trimap_loss(anchor_emb, near_emb, far_emb, weights):
    """TriMap triplet loss.

    s(a, b) = 1 / (1 + ||y_a - y_b||^2)   (Student-t similarity)
    L = sum w_ijk * s(anchor, far) / (s(anchor, near) + s(anchor, far))

    Parameters
    ----------
    anchor_emb : tensor (T, D)
    near_emb : tensor (T, D)
    far_emb : tensor (T, D)
    weights : tensor (T,)

    Returns
    -------
    loss : scalar tensor
    """
    d_near = torch.sum((anchor_emb - near_emb) ** 2, dim=1)
    d_far = torch.sum((anchor_emb - far_emb) ** 2, dim=1)

    s_near = 1.0 / (1.0 + d_near)
    s_far = 1.0 / (1.0 + d_far)

    loss = weights * s_far / (s_near + s_far + 1e-7)
    return loss.mean()


def _generate_triplets(X, knn_indices, knn_distances, n_outliers, n_random, seed=0):
    """Generate weighted triplets for TriMap.

    For each point, generates n_outliers * n_random triplets. Near neighbors
    come from the KNN graph; far points are sampled randomly. Weights are
    derived from the ratio of far to near distances in the input space.

    Parameters
    ----------
    X : 2-d array (N, D)
    knn_indices : 2-d array (N, K)
    knn_distances : 2-d array (N, K)
    n_outliers : int
        Number of near neighbors to use per point.
    n_random : int
        Number of far points to sample per near neighbor.
    seed : int

    Returns
    -------
    triplets : 2-d array (T, 3)
        Columns: [anchor, near, far].
    weights : 1-d array (T,)
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    n_near = min(n_outliers, knn_indices.shape[1])

    anchors, nears, fars, ws = [], [], [], []
    for i in range(n):
        for j_pos in range(n_near):
            j = knn_indices[i, j_pos]
            d_near_sq = max(knn_distances[i, j_pos] ** 2, 1e-10)

            for _ in range(n_random):
                k = rng.randint(0, n)
                while k == i or k == j:
                    k = rng.randint(0, n)
                d_far_sq = float(np.sum((X[i] - X[k]) ** 2))
                w = d_far_sq / (d_near_sq + 1e-10)

                anchors.append(i)
                nears.append(j)
                fars.append(k)
                ws.append(w)

    triplets = np.column_stack([anchors, nears, fars])
    weights = np.array(ws, dtype=np.float32)
    # Clip weights to prevent extreme values
    weights = np.clip(weights, 1e-4, 1e4)
    return triplets, weights


class Parametric_TriMap(ParametricDR):
    """Parametric TriMap dimensionality reduction.

    Parameters
    ----------
    num_inputs : int
    num_outputs : int
    n_neighbors : int
        Number of neighbors for the KNN graph.
    n_outliers : int
        Number of near neighbors to use in triplet generation.
    n_random : int
        Number of random far points per near neighbor.
    learning_rate : float
    encoder : nn.Module or None
    seed : int
    batch_size : int
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        n_neighbors: int = 10,
        n_outliers: int = 5,
        n_random: int = 3,
        n_pca: int | None = 50,
        learning_rate: float = 1e-3,
        encoder=None,
        seed: int = 0,
        batch_size: int = 64,
    ):
        super().__init__(
            num_inputs, num_outputs, n_pca, learning_rate, encoder, seed, batch_size
        )
        self.n_neighbors = n_neighbors
        self.n_outliers = n_outliers
        self.n_random = n_random

    def fit(self, X, y=None, epochs=100):
        """Train the parametric TriMap model.

        Parameters
        ----------
        X : 2-d array (N, num_inputs)
        y : ignored
        epochs : int

        Returns
        -------
        self
        """
        X = self._process_training_data(X)
        n = X.shape[0]

        k = min(self.n_neighbors, n - 1)
        knn_indices, knn_distances = compute_knn_graph(X, k)

        triplets, weights = _generate_triplets(
            X, knn_indices, knn_distances, self.n_outliers, self.n_random, self.seed
        )
        n_triplets = len(triplets)

        X_t = torch.tensor(X, dtype=torch.float32)
        weights_t = torch.tensor(weights, dtype=torch.float32)
        opt = self._setup_training()

        for epoch in range(epochs):
            perm = np.random.permutation(n_triplets)
            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, n_triplets, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n_triplets)
                batch_idx = perm[batch_start:batch_end]

                t = triplets[batch_idx]
                w = weights_t[batch_idx]

                anchor_emb = self.encoder(X_t[t[:, 0]])
                near_emb = self.encoder(X_t[t[:, 1]])
                far_emb = self.encoder(X_t[t[:, 2]])

                opt.zero_grad()
                loss = trimap_loss(anchor_emb, near_emb, far_emb, w)
                loss = loss + self._extra_loss(anchor_emb, t[:, 0])
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

            if n_batches > 0:
                avg = epoch_loss / n_batches
                logger.debug("Epoch %d/%d, loss=%.4f", epoch + 1, epochs, avg)

        self._is_fitted = True
        return self
