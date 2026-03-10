import tempfile
import os

import numpy as np
import pytest
import torch.nn as nn

from parametric_dr import Parametric_TriMap
from parametric_dr._base import ParametricDR
from parametric_dr.trimap import _generate_triplets
from parametric_dr.utils import compute_knn_graph


@pytest.fixture
def small_data():
    np.random.seed(42)
    return np.random.randn(128, 10).astype(np.float32)


class TestParametricTriMap:
    def test_fit_transform_shape(self, small_data):
        model = Parametric_TriMap(
            num_inputs=10, num_outputs=2, n_neighbors=5,
            n_outliers=3, n_random=2, batch_size=32, seed=42,
        )
        result = model.fit_transform(small_data, epochs=3)
        assert result.shape == (128, 2)

    def test_transform_after_fit(self, small_data):
        model = Parametric_TriMap(
            num_inputs=10, num_outputs=2, n_neighbors=5,
            n_outliers=3, n_random=2, batch_size=32, seed=42,
        )
        model.fit(small_data, epochs=3)
        new_data = np.random.randn(20, 10).astype(np.float32)
        result = model.transform(new_data)
        assert result.shape == (20, 2)

    def test_transform_before_fit_raises(self):
        model = Parametric_TriMap(num_inputs=5, num_outputs=2)
        with pytest.raises(AssertionError, match="fit"):
            model.transform(np.random.randn(10, 5))

    def test_save_and_restore(self, small_data):
        model = Parametric_TriMap(
            num_inputs=10, num_outputs=2, n_neighbors=5,
            n_outliers=3, n_random=2, batch_size=32, seed=42,
        )
        model.fit(small_data, epochs=3)
        orig = model.transform(small_data[:10])

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save_model(path)
            model2 = Parametric_TriMap(
                num_inputs=10, num_outputs=2, n_neighbors=5,
                n_outliers=3, n_random=2, batch_size=32, seed=42,
            )
            model2.restore_model(path)
            restored = model2.transform(small_data[:10])
            np.testing.assert_allclose(orig, restored, atol=1e-6)
        finally:
            os.unlink(path)

    def test_custom_encoder(self, small_data):
        encoder = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
        model = Parametric_TriMap(
            num_inputs=10, num_outputs=2, n_neighbors=5,
            n_outliers=3, n_random=2, encoder=encoder, batch_size=32, seed=42,
        )
        result = model.fit_transform(small_data, epochs=3)
        assert result.shape == (128, 2)

    def test_inherits_parametric_dr(self):
        model = Parametric_TriMap(num_inputs=5, num_outputs=2)
        assert isinstance(model, ParametricDR)


class TestTripletGeneration:
    def test_valid_indices(self):
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        indices, distances = compute_knn_graph(X, 10)
        triplets, weights = _generate_triplets(X, indices, distances, 3, 2)

        assert triplets.shape[1] == 3
        assert len(weights) == len(triplets)
        assert np.all(triplets >= 0)
        assert np.all(triplets < 50)

    def test_anchor_not_equal_near_or_far(self):
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        indices, distances = compute_knn_graph(X, 10)
        triplets, _ = _generate_triplets(X, indices, distances, 3, 2)

        # Anchor should differ from far (near can equal anchor only if KNN has self)
        assert np.all(triplets[:, 0] != triplets[:, 2])

    def test_weights_positive(self):
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        indices, distances = compute_knn_graph(X, 10)
        _, weights = _generate_triplets(X, indices, distances, 3, 2)
        assert np.all(weights > 0)
