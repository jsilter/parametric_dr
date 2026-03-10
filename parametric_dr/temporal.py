"""TemporalMixin: adds temporal smoothness penalty to any ParametricDR subclass.

Composable mixin that penalizes temporally adjacent embeddings from
being far apart. Use with multiple inheritance:

    class Temporal_tSNE(TemporalMixin, Parametric_tSNE):
        pass

    model = Temporal_tSNE(
        num_inputs=10, num_outputs=2, perplexities=10.0,
        temporal_weight=0.1,
    )
    model.fit(X, time_indices=np.arange(len(X)))
"""

import numpy as np
import torch

from ._base import ParametricDR


class TemporalMixin:
    """Mixin that adds temporal smoothness to any ParametricDR subclass.

    Adds a loss term: temporal_weight * mean(||f(x_t) - f(x_{t+1})||^2)
    for temporally consecutive points.

    Parameters
    ----------
    temporal_weight : float
        Weight for the temporal smoothness penalty.
    """

    def __init__(self, *args, temporal_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_weight = temporal_weight
        self._temporal_pairs = None
        self._X_temporal = None

    def fit(self, X, y=None, time_indices=None, **kwargs):
        """Wraps parent fit() to capture time_indices and set up temporal pairs.

        Parameters
        ----------
        X : 2-d array (N, D)
        y : ignored
        time_indices : 1-d array (N,), optional
            Temporal index for each sample. Consecutive integer values
            (e.g., 0, 1, 2, ...) define temporal adjacency. If None,
            no temporal penalty is applied.
        **kwargs : passed to parent fit()
        """
        # Process training data (fits PCA if enabled) so that _X_temporal
        # stores encoder-compatible data
        X_processed = self._process_training_data(X)

        if time_indices is not None:
            time_indices = np.asarray(time_indices)
            # Find pairs of data indices that are temporally consecutive
            order = np.argsort(time_indices)
            times_sorted = time_indices[order]
            consecutive = np.where(np.diff(times_sorted) == 1)[0]
            if len(consecutive) > 0:
                self._temporal_pairs = np.column_stack(
                    [order[consecutive], order[consecutive + 1]]
                )
            else:
                self._temporal_pairs = np.empty((0, 2), dtype=int)
            self._X_temporal = torch.tensor(X_processed, dtype=torch.float32)
        else:
            self._temporal_pairs = None
            self._X_temporal = None

        # Pass processed data to parent; parent's _process_training_data
        # will detect PCA is already fitted/applied and pass through
        return super().fit(X_processed, y=y, **kwargs)

    def _extra_loss(self, output, batch_indices):
        """Add temporal smoothness penalty.

        Samples a subset of temporal pairs each step and penalizes
        the squared distance between their embeddings.
        """
        base_extra = super()._extra_loss(output, batch_indices)

        if self._temporal_pairs is None or len(self._temporal_pairs) == 0:
            return base_extra

        # Sample temporal pairs (up to batch_size pairs per step)
        n_pairs = min(len(self._temporal_pairs), self.batch_size)
        pair_idx = np.random.choice(
            len(self._temporal_pairs), size=n_pairs, replace=False
        )
        pairs = self._temporal_pairs[pair_idx]

        # Encode both endpoints
        emb_a = self.encoder(self._X_temporal[pairs[:, 0]])
        emb_b = self.encoder(self._X_temporal[pairs[:, 1]])

        temporal_loss = torch.mean(torch.sum((emb_a - emb_b) ** 2, dim=1))
        return base_extra + self.temporal_weight * temporal_loss
