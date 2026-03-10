"""Parametric CEBRA: Contrastive Embedding for Behavioral and neural Recording Analysis.

Learns a neural network embedding using InfoNCE contrastive loss
with temporal structure. Points that are temporally close are pulled
together; random points are pushed apart.

Reference:
Schneider, Lee, Mathis (2023). Learnable latent embeddings for joint
behavioural and neural analysis. Nature.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

from ._base import ParametricDR
from .utils import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def cebra_loss(anchor_emb, pos_emb, neg_emb, temperature=1.0):
    """InfoNCE contrastive loss.

    Parameters
    ----------
    anchor_emb : tensor (B, D)
    pos_emb : tensor (B, D)
        Positive (temporally close) embeddings.
    neg_emb : tensor (B, K, D)
        Negative (random) embeddings.
    temperature : float

    Returns
    -------
    loss : scalar tensor
    """
    # L2 normalize
    anchor_norm = F.normalize(anchor_emb, dim=1)
    pos_norm = F.normalize(pos_emb, dim=1)
    neg_norm = F.normalize(neg_emb, dim=2)

    # Positive similarity: (B,)
    pos_sim = (anchor_norm * pos_norm).sum(dim=1) / temperature

    # Negative similarity: (B, K)
    neg_sim = torch.bmm(neg_norm, anchor_norm.unsqueeze(2)).squeeze(2) / temperature

    # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(anchor_emb.shape[0], dtype=torch.long)
    return F.cross_entropy(logits, labels)


class Parametric_CEBRA(ParametricDR):
    """Parametric CEBRA: contrastive temporal dimensionality reduction.

    Parameters
    ----------
    num_inputs : int
    num_outputs : int
    n_neighbors_time : int
        Temporal window: positive samples are within this many time steps.
    temperature : float
        InfoNCE temperature parameter.
    negative_sample_rate : int
        Number of negative samples per anchor.
    learning_rate : float
    encoder : nn.Module or None
    seed : int
    batch_size : int
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        n_neighbors_time: int = 5,
        temperature: float = 1.0,
        negative_sample_rate: int = 10,
        n_pca: int | None = 50,
        learning_rate: float = 1e-3,
        encoder=None,
        seed: int = 0,
        batch_size: int = 64,
    ):
        super().__init__(
            num_inputs, num_outputs, n_pca, learning_rate, encoder, seed, batch_size
        )
        self.n_neighbors_time = n_neighbors_time
        self.temperature = temperature
        self.negative_sample_rate = negative_sample_rate

    def fit(self, X, y=None, time_indices=None, epochs=100):
        """Train the parametric CEBRA model.

        Parameters
        ----------
        X : 2-d array (N, num_inputs)
        y : ignored
        time_indices : 1-d array (N,), optional
            Temporal index for each sample. If None, assumes sequential
            ordering (0, 1, 2, ...).
        epochs : int

        Returns
        -------
        self
        """
        X = self._process_training_data(X)
        n = X.shape[0]

        if time_indices is None:
            time_indices = np.arange(n, dtype=np.float64)
        time_indices = np.asarray(time_indices, dtype=np.float64)

        # Build temporal neighbor lookup: for each point, which others
        # are within n_neighbors_time steps?
        temporal_neighbors = []
        for i in range(n):
            mask = np.abs(time_indices - time_indices[i]) <= self.n_neighbors_time
            mask[i] = False
            neighbors = np.where(mask)[0]
            if len(neighbors) == 0:
                # Fallback: use nearest point by time
                diffs = np.abs(time_indices - time_indices[i])
                diffs[i] = np.inf
                neighbors = np.array([np.argmin(diffs)])
            temporal_neighbors.append(neighbors)

        X_t = torch.tensor(X, dtype=torch.float32)
        num_batches = n // self.batch_size
        opt = self._setup_training()

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches_done = 0

            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                idx = perm[start : start + self.batch_size]
                bs = len(idx)

                # Sample one positive per anchor (random temporal neighbor)
                pos_idx = np.array(
                    [np.random.choice(temporal_neighbors[i]) for i in idx]
                )

                # Sample negatives (random points)
                neg_idx = np.random.randint(0, n, size=(bs, self.negative_sample_rate))

                anchor_emb = self.encoder(X_t[idx])
                pos_emb = self.encoder(X_t[pos_idx])
                neg_emb = self.encoder(X_t[neg_idx.ravel()]).reshape(
                    bs, self.negative_sample_rate, self.num_outputs
                )

                opt.zero_grad()
                loss = cebra_loss(anchor_emb, pos_emb, neg_emb, self.temperature)
                loss = loss + self._extra_loss(anchor_emb, idx)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches_done += 1

            if n_batches_done > 0:
                avg = epoch_loss / n_batches_done
                logger.debug("Epoch %d/%d, loss=%.4f", epoch + 1, epochs, avg)

        self._is_fitted = True
        return self
