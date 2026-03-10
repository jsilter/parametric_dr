"""Parametric PaCMAP: Pairwise Controlled Manifold Approximation.

Learns a neural network embedding by optimizing three types of pair
relationships (near, mid-near, further) with a phased weight schedule.

Reference:
Wang, Huang, Ruber, Liang, Xu (2021). Understanding How Dimension
Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP,
TriMap, and PaCMAP for Data Visualization. JMLR.
"""

import logging

import numpy as np
import torch

from ._base import ParametricDR
from .utils import LOGGER_NAME, compute_knn_graph

logger = logging.getLogger(LOGGER_NAME)


def pacmap_loss(anchor_emb, near_emb, mn_emb, fp_emb, w_NB, w_MN, w_FP):
    """PaCMAP loss over pre-sampled pairs.

    Parameters
    ----------
    anchor_emb : tensor (B, D)
    near_emb : tensor (B, K_near, D)
    mn_emb : tensor (B, K_mn, D)
    fp_emb : tensor (B, K_fp, D)
    w_NB, w_MN, w_FP : float
        Phase-dependent weights.

    Returns
    -------
    loss : scalar tensor
    """
    # Near-neighbor attractive loss
    d_near = torch.sum((anchor_emb.unsqueeze(1) - near_emb) ** 2, dim=2) + 1.0
    L_NB = torch.mean(d_near / (10.0 + d_near))

    # Mid-near attractive loss
    loss = w_NB * L_NB
    if w_MN > 0 and mn_emb.shape[1] > 0:
        d_mn = torch.sum((anchor_emb.unsqueeze(1) - mn_emb) ** 2, dim=2) + 1.0
        L_MN = torch.mean(d_mn / (10000.0 + d_mn))
        loss = loss + w_MN * L_MN

    # Further-pair repulsive loss
    d_fp = torch.sum((anchor_emb.unsqueeze(1) - fp_emb) ** 2, dim=2) + 1.0
    L_FP = torch.mean(1.0 / (1.0 + d_fp))
    loss = loss + w_FP * L_FP

    return loss


class Parametric_PaCMAP(ParametricDR):
    """Parametric PaCMAP dimensionality reduction.

    Parameters
    ----------
    num_inputs : int
    num_outputs : int
    n_neighbors : int
        Number of near neighbors per point.
    n_MN_ratio : float
        Ratio of mid-near pairs to near neighbors.
    n_FP_ratio : float
        Ratio of further pairs to near neighbors.
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
        n_MN_ratio: float = 0.5,
        n_FP_ratio: float = 2.0,
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
        self.n_MN_ratio = n_MN_ratio
        self.n_FP_ratio = n_FP_ratio

    def fit(self, X, y=None, epochs=100):
        """Train the parametric PaCMAP model.

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

        n_MN = max(1, int(self.n_MN_ratio * self.n_neighbors))
        n_FP = max(1, int(self.n_FP_ratio * self.n_neighbors))

        # Build KNN graph (need up to 6*n_neighbors for mid-near sampling)
        knn_k = min(6 * self.n_neighbors, n - 1)
        knn_indices, _ = compute_knn_graph(X, knn_k)

        # Near pairs: k-nearest neighbors
        near_pairs = knn_indices[:, : self.n_neighbors]

        # Mid-near pairs: sample from ranks n_neighbors to 6*n_neighbors
        mn_candidates = knn_indices[:, self.n_neighbors :]
        mn_pairs = np.zeros((n, n_MN), dtype=int)
        for i in range(n):
            n_cand = mn_candidates.shape[1]
            if n_cand == 0:
                mn_pairs[i] = np.random.randint(0, n, size=n_MN)
            else:
                chosen = np.random.choice(
                    n_cand, size=n_MN, replace=(n_cand < n_MN)
                )
                mn_pairs[i] = mn_candidates[i, chosen]

        # Further pairs: random
        fp_pairs = np.random.randint(0, n, size=(n, n_FP))

        X_t = torch.tensor(X, dtype=torch.float32)
        num_batches = n // self.batch_size
        opt = self._setup_training()

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0

            # Phase-dependent weights
            progress = epoch / max(epochs, 1)
            if progress < 0.22:
                frac = progress / 0.22
                w_NB = 2.0
                w_MN = 1000.0 - (1000.0 - 3.0) * frac
                w_FP = 1.0
            elif progress < 0.44:
                w_NB, w_MN, w_FP = 3.0, 3.0, 1.0
            else:
                w_NB, w_MN, w_FP = 1.0, 0.0, 1.0

            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                idx = perm[start : start + self.batch_size]
                bs = len(idx)

                anchor_emb = self.encoder(X_t[idx])

                near_idx = near_pairs[idx]
                near_emb = self.encoder(X_t[near_idx.ravel()]).reshape(
                    bs, self.n_neighbors, self.num_outputs
                )

                mn_idx = mn_pairs[idx]
                mn_emb = self.encoder(X_t[mn_idx.ravel()]).reshape(
                    bs, n_MN, self.num_outputs
                )

                fp_idx = fp_pairs[idx]
                fp_emb = self.encoder(X_t[fp_idx.ravel()]).reshape(
                    bs, n_FP, self.num_outputs
                )

                opt.zero_grad()
                loss = pacmap_loss(anchor_emb, near_emb, mn_emb, fp_emb, w_NB, w_MN, w_FP)
                loss = loss + self._extra_loss(anchor_emb, idx)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

            if num_batches > 0:
                avg = epoch_loss / num_batches
                logger.debug("Epoch %d/%d, loss=%.4f", epoch + 1, epochs, avg)

        self._is_fitted = True
        return self
