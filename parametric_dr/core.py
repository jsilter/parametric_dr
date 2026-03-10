"""Parametric t-SNE: dimensionality reduction via KL divergence minimization.

Trains an encoder network by minimizing KL divergence between Gaussian
similarities (P) in the high-dimensional input space and Student-t
similarities (Q) in the low-dimensional output space.

Main reference:
van der Maaten, L. (2009). Learning a parametric embedding by preserving
local structure. RBM, 500(500), 26.
"""

import logging
from typing import List, Union, Optional

import numpy as np
import torch

from ._base import ParametricDR
from .loss import kl_loss
from .utils import LOGGER_NAME, calc_betas_loop, get_squared_cross_diff_np

logger = logging.getLogger(LOGGER_NAME)

DEFAULT_EPS = 1e-7


def _make_P_ji(data: np.ndarray, betas: np.ndarray, in_sq_diffs: np.ndarray = None):
    """Compute asymmetric similarity probabilities.

    Parameters
    ----------
    data : 2-d array (N, D)
    betas : 2-d array (N, P)
    in_sq_diffs : 2-d array (N, N), optional

    Returns
    -------
    P_ji : 3-d array (N, N, P)
    """
    if in_sq_diffs is None:
        in_sq_diffs = get_squared_cross_diff_np(data)
    tmp = in_sq_diffs[:, :, np.newaxis] * betas[np.newaxis, :, :]
    return np.exp(-1.0 * tmp)


def _get_normed_sym_np(x: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Normalize and symmetrize probability matrix (numpy)."""
    batch_size = x.shape[0]
    zero_diags = 1.0 - np.identity(batch_size)
    P = x * zero_diags
    norm_facs = np.sum(x, axis=0, keepdims=True)
    P = P / (norm_facs + eps)
    P = 0.5 * (P + np.transpose(P))
    return P


def _make_P_np(data: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Compute symmetric P matrix for each perplexity."""
    P_ji = _make_P_ji(data, betas)
    P_3 = np.zeros_like(P_ji)
    for zz in range(P_3.shape[2]):
        P_3[:, :, zz] = _get_normed_sym_np(P_ji[:, :, zz])
    return P_3


def _compute_batch_P(data: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Compute the concatenated P matrix for a single batch.

    Returns
    -------
    P_concat : 2-d array (batch_size, batch_size * num_perplexities)
    """
    P_3d = _make_P_np(data, betas)
    P_arrays = [P_3d[:, :, pp] for pp in range(P_3d.shape[2])]
    return np.concatenate(P_arrays, axis=1)


class Parametric_tSNE(ParametricDR):
    """Parametric t-SNE via neural network encoder.

    Implements the sklearn estimator interface (fit / transform / fit_transform).

    Parameters
    ----------
    num_inputs : int
        Dimensionality of input data.
    num_outputs : int
        Dimensionality of embedding (typically 2).
    perplexities : float, list of float, or None
        Perplexity value(s). Can be a list for multi-scale t-SNE.
        None is allowed only when providing training_betas to fit().
    alpha : float
        Degrees of freedom for Student-t kernel. Default 1.0.
    learning_rate : float
        Learning rate for the optimizer.
    encoder : nn.Module or None
        Custom encoder network. If None, uses the default architecture.
    seed : int
        Random seed.
    batch_size : int
        Training batch size.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        perplexities: Union[float, List[float], None] = None,
        alpha: float = 1.0,
        n_pca: Optional[int] = 50,
        learning_rate: float = 1e-3,
        encoder=None,
        seed: int = 0,
        batch_size: int = 64,
    ):
        super().__init__(
            num_inputs, num_outputs, n_pca, learning_rate, encoder, seed, batch_size
        )
        if perplexities is not None and not isinstance(
            perplexities, (list, tuple, np.ndarray)
        ):
            perplexities = [perplexities]
        self.perplexities = perplexities
        self.alpha = alpha

    @property
    def num_perplexities(self) -> Optional[int]:
        if self.perplexities is None:
            return None
        return len(self.perplexities)

    @staticmethod
    def _calc_training_betas(
        training_data: np.ndarray,
        perplexities: Union[float, List[float], np.ndarray],
        beta_batch_size: int = 1000,
    ) -> np.ndarray:
        """Compute beta (kernel width) values via binary search on perplexity.

        Processes data in chunks of beta_batch_size for memory efficiency.
        """
        assert perplexities is not None, (
            "Must provide desired perplexities if training beta values"
        )
        num_pts = len(training_data)
        if not isinstance(perplexities, (list, tuple, np.ndarray)):
            perplexities = np.array([perplexities])
        else:
            perplexities = np.asarray(perplexities)
        num_perplexities = len(perplexities)
        training_betas = np.zeros([num_pts, num_perplexities])

        cur_start = 0
        cur_end = min(cur_start + beta_batch_size, num_pts)
        while cur_start < num_pts:
            cur_training_data = training_data[cur_start:cur_end, :]
            for pind, curperp in enumerate(perplexities):
                cur_training_betas, cur_Hs, cur_P = calc_betas_loop(
                    cur_training_data, curperp
                )
                training_betas[cur_start:cur_end, pind] = cur_training_betas
            cur_start += beta_batch_size
            cur_end = min(cur_start + beta_batch_size, num_pts)

        return training_betas

    def fit(
        self,
        X: np.ndarray,
        y=None,
        training_betas: Optional[np.ndarray] = None,
        epochs: int = 10,
    ):
        """Train the parametric t-SNE model.

        Parameters
        ----------
        X : 2-d array (N, num_inputs)
        y : ignored
        training_betas : 2-d array (N, P), optional
            Precomputed beta values. If None, computed from perplexities.
        epochs : int

        Returns
        -------
        self
        """
        X = self._process_training_data(X)
        X = X.astype(np.float64)

        if training_betas is None:
            assert self.perplexities is not None, (
                "Must provide perplexities or training_betas"
            )
            logger.debug("Computing training betas...")
            training_betas = self._calc_training_betas(X, self.perplexities)
        num_perplexities = training_betas.shape[1]

        n = X.shape[0]
        num_batches = n // self.batch_size
        opt = self._setup_training()

        logger.debug("Training for %d epochs, %d batches/epoch", epochs, num_batches)

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                idx = perm[start : start + self.batch_size]

                batch_data = X[idx]
                batch_betas = training_betas[idx]
                batch_P = _compute_batch_P(batch_data, batch_betas)

                batch_data_t = torch.tensor(batch_data, dtype=torch.float32)
                batch_P_t = torch.tensor(batch_P, dtype=torch.float32)

                opt.zero_grad()
                output = self.encoder(batch_data_t)
                loss = kl_loss(
                    batch_P_t,
                    output,
                    alpha=self.alpha,
                    num_perplexities=num_perplexities,
                )
                loss = loss + self._extra_loss(output, idx)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.debug("Epoch %d/%d, loss=%.4f", epoch + 1, epochs, avg_loss)

        logger.debug("Training complete")

        self._is_fitted = True
        return self
