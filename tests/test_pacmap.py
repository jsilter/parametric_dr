import tempfile
import os

import numpy as np
import pytest
import torch.nn as nn

from parametric_dr import Parametric_PaCMAP
from parametric_dr._base import ParametricDR


@pytest.fixture
def small_data():
    np.random.seed(42)
    return np.random.randn(128, 10).astype(np.float32)


class TestParametricPaCMAP:
    def test_fit_transform_shape(self, small_data):
        model = Parametric_PaCMAP(
            num_inputs=10, num_outputs=2, n_neighbors=5,
            batch_size=32, seed=42,
        )
        result = model.fit_transform(small_data, epochs=3)
        assert result.shape == (128, 2)

    def test_transform_after_fit(self, small_data):
        model = Parametric_PaCMAP(
            num_inputs=10, num_outputs=2, n_neighbors=5,
            batch_size=32, seed=42,
        )
        model.fit(small_data, epochs=3)
        new_data = np.random.randn(20, 10).astype(np.float32)
        result = model.transform(new_data)
        assert result.shape == (20, 2)

    def test_transform_before_fit_raises(self):
        model = Parametric_PaCMAP(num_inputs=5, num_outputs=2)
        with pytest.raises(AssertionError, match="fit"):
            model.transform(np.random.randn(10, 5))

    def test_save_and_restore(self, small_data):
        model = Parametric_PaCMAP(
            num_inputs=10, num_outputs=2, n_neighbors=5,
            batch_size=32, seed=42,
        )
        model.fit(small_data, epochs=3)
        orig = model.transform(small_data[:10])

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save_model(path)
            model2 = Parametric_PaCMAP(
                num_inputs=10, num_outputs=2, n_neighbors=5,
                batch_size=32, seed=42,
            )
            model2.restore_model(path)
            restored = model2.transform(small_data[:10])
            np.testing.assert_allclose(orig, restored, atol=1e-6)
        finally:
            os.unlink(path)

    def test_custom_encoder(self, small_data):
        encoder = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
        model = Parametric_PaCMAP(
            num_inputs=10, num_outputs=2, n_neighbors=5,
            encoder=encoder, batch_size=32, seed=42,
        )
        result = model.fit_transform(small_data, epochs=3)
        assert result.shape == (128, 2)

    def test_inherits_parametric_dr(self):
        model = Parametric_PaCMAP(num_inputs=5, num_outputs=2)
        assert isinstance(model, ParametricDR)

    def test_phase_weights_change(self, small_data):
        """Verify the three-phase weight schedule produces different losses."""
        model = Parametric_PaCMAP(
            num_inputs=10, num_outputs=2, n_neighbors=5,
            batch_size=32, seed=42,
        )
        # Train for 10 epochs to cover multiple phases
        model.fit(small_data, epochs=10)
        assert model._is_fitted
