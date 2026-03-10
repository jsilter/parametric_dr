import numpy as np
import pytest
import torch.nn as nn

from parametric_dr import Parametric_tSNE, Parametric_PaCMAP, TemporalMixin
from parametric_dr._base import ParametricDR


class Temporal_tSNE(TemporalMixin, Parametric_tSNE):
    pass


class Temporal_PaCMAP(TemporalMixin, Parametric_PaCMAP):
    pass


@pytest.fixture
def temporal_data():
    np.random.seed(42)
    N, D = 128, 10
    X = np.random.randn(N, D).astype(np.float32)
    time_indices = np.arange(N)
    return X, time_indices


class TestTemporalMixin:
    def test_compose_with_tsne(self, temporal_data):
        X, t = temporal_data
        encoder = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
        model = Temporal_tSNE(
            num_inputs=10, num_outputs=2, perplexities=10.0,
            temporal_weight=0.1, encoder=encoder, batch_size=32, seed=42,
        )
        result = model.fit_transform(X, time_indices=t, epochs=2)
        assert result.shape == (128, 2)

    def test_compose_with_pacmap(self, temporal_data):
        X, t = temporal_data
        encoder = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
        model = Temporal_PaCMAP(
            num_inputs=10, num_outputs=2, n_neighbors=5,
            temporal_weight=0.1, encoder=encoder, batch_size=32, seed=42,
        )
        result = model.fit_transform(X, time_indices=t, epochs=3)
        assert result.shape == (128, 2)

    def test_no_time_indices_works(self, temporal_data):
        """Without time_indices, temporal penalty is skipped."""
        X, _ = temporal_data
        encoder = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
        model = Temporal_tSNE(
            num_inputs=10, num_outputs=2, perplexities=10.0,
            temporal_weight=0.1, encoder=encoder, batch_size=32, seed=42,
        )
        result = model.fit_transform(X, epochs=2)
        assert result.shape == (128, 2)

    def test_temporal_weight_stored(self):
        model = Temporal_tSNE(
            num_inputs=5, num_outputs=2, perplexities=5.0,
            temporal_weight=0.5,
        )
        assert model.temporal_weight == 0.5

    def test_inherits_parametric_dr(self):
        model = Temporal_tSNE(
            num_inputs=5, num_outputs=2, perplexities=5.0,
        )
        assert isinstance(model, ParametricDR)
        assert isinstance(model, Parametric_tSNE)
        assert isinstance(model, TemporalMixin)

    def test_temporal_extra_loss_nonzero(self):
        """Temporal mixin produces non-zero extra loss when pairs exist."""
        import torch

        np.random.seed(42)
        X = np.random.randn(64, 10).astype(np.float32)
        time_idx = np.arange(64)

        encoder = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
        model = Temporal_tSNE(
            num_inputs=10, num_outputs=2, perplexities=10.0,
            temporal_weight=1.0, encoder=encoder, batch_size=32, seed=42,
        )

        # Manually set up temporal state (normally done in fit())
        model._X_temporal = torch.tensor(X, dtype=torch.float32)
        order = np.argsort(time_idx)
        times_sorted = time_idx[order]
        consecutive = np.where(np.diff(times_sorted) == 1)[0]
        model._temporal_pairs = np.column_stack(
            [order[consecutive], order[consecutive + 1]]
        )

        output = model.encoder(torch.tensor(X[:32], dtype=torch.float32))
        extra = model._extra_loss(output, np.arange(32))

        assert extra.item() > 0, "Temporal extra loss should be non-zero"
        assert torch.isfinite(extra)

    def test_temporal_extra_loss_zero_without_pairs(self):
        """Without time_indices, extra loss should be zero."""
        import torch

        encoder = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
        model = Temporal_tSNE(
            num_inputs=10, num_outputs=2, perplexities=10.0,
            temporal_weight=1.0, encoder=encoder, batch_size=32, seed=42,
        )
        # _temporal_pairs is None by default
        output = torch.randn(8, 2)
        extra = model._extra_loss(output, np.arange(8))
        assert extra.item() == 0.0
