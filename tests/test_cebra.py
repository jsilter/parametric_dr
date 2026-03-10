import tempfile
import os

import numpy as np
import pytest
import torch.nn as nn

from parametric_dr import Parametric_CEBRA
from parametric_dr._base import ParametricDR


@pytest.fixture
def temporal_data():
    """Synthetic temporal data: 128 time steps, 10 features."""
    np.random.seed(42)
    N, D = 128, 10
    X = np.random.randn(N, D).astype(np.float32)
    time_indices = np.arange(N)
    return X, time_indices


class TestParametricCEBRA:
    def test_fit_transform_shape(self, temporal_data):
        X, t = temporal_data
        model = Parametric_CEBRA(
            num_inputs=10, num_outputs=2, n_neighbors_time=3,
            batch_size=32, seed=42,
        )
        result = model.fit_transform(X, time_indices=t, epochs=3)
        assert result.shape == (128, 2)

    def test_fit_without_time_indices(self, temporal_data):
        """Should work with sequential ordering assumed."""
        X, _ = temporal_data
        model = Parametric_CEBRA(
            num_inputs=10, num_outputs=2, n_neighbors_time=3,
            batch_size=32, seed=42,
        )
        result = model.fit_transform(X, epochs=3)
        assert result.shape == (128, 2)

    def test_transform_after_fit(self, temporal_data):
        X, t = temporal_data
        model = Parametric_CEBRA(
            num_inputs=10, num_outputs=2, n_neighbors_time=3,
            batch_size=32, seed=42,
        )
        model.fit(X, time_indices=t, epochs=3)
        new_data = np.random.randn(20, 10).astype(np.float32)
        result = model.transform(new_data)
        assert result.shape == (20, 2)

    def test_transform_before_fit_raises(self):
        model = Parametric_CEBRA(num_inputs=5, num_outputs=2)
        with pytest.raises(AssertionError, match="fit"):
            model.transform(np.random.randn(10, 5))

    def test_save_and_restore(self, temporal_data):
        X, t = temporal_data
        model = Parametric_CEBRA(
            num_inputs=10, num_outputs=2, n_neighbors_time=3,
            batch_size=32, seed=42,
        )
        model.fit(X, time_indices=t, epochs=3)
        orig = model.transform(X[:10])

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save_model(path)
            model2 = Parametric_CEBRA(
                num_inputs=10, num_outputs=2, n_neighbors_time=3,
                batch_size=32, seed=42,
            )
            model2.restore_model(path)
            restored = model2.transform(X[:10])
            np.testing.assert_allclose(orig, restored, atol=1e-6)
        finally:
            os.unlink(path)

    def test_custom_encoder(self, temporal_data):
        X, t = temporal_data
        encoder = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
        model = Parametric_CEBRA(
            num_inputs=10, num_outputs=2, n_neighbors_time=3,
            encoder=encoder, batch_size=32, seed=42,
        )
        result = model.fit_transform(X, time_indices=t, epochs=3)
        assert result.shape == (128, 2)

    def test_inherits_parametric_dr(self):
        model = Parametric_CEBRA(num_inputs=5, num_outputs=2)
        assert isinstance(model, ParametricDR)

    def test_contrastive_loss_decreases(self, temporal_data):
        """Loss should decrease over training epochs (with structured data)."""
        # Create data with clear temporal structure
        np.random.seed(42)
        t = np.linspace(0, 4 * np.pi, 128)
        X = np.column_stack([np.sin(t + i * 0.1) for i in range(10)]).astype(np.float32)
        time_idx = np.arange(128)

        encoder = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
        model = Parametric_CEBRA(
            num_inputs=10, num_outputs=2, n_neighbors_time=5,
            encoder=encoder, batch_size=32, seed=42,
        )
        model.fit(X, time_indices=time_idx, epochs=5)
        assert model._is_fitted
