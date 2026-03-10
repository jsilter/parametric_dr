import tempfile
import os

import numpy as np
import pytest
import torch.nn as nn

from parametric_dr import Parametric_UMAP
from parametric_dr._base import ParametricDR
from parametric_dr.umap import _fit_ab_params, _compute_fuzzy_simplicial_set
from parametric_dr.utils import compute_knn_graph


@pytest.fixture
def small_data():
    np.random.seed(42)
    return np.random.randn(128, 10).astype(np.float32)


class TestParametricUMAP:
    def test_fit_transform_shape(self, small_data):
        model = Parametric_UMAP(
            num_inputs=10, num_outputs=2, n_neighbors=10,
            batch_size=32, seed=42,
        )
        result = model.fit_transform(small_data, epochs=3)
        assert result.shape == (128, 2)

    def test_transform_after_fit(self, small_data):
        model = Parametric_UMAP(
            num_inputs=10, num_outputs=2, n_neighbors=10,
            batch_size=32, seed=42,
        )
        model.fit(small_data, epochs=3)
        new_data = np.random.randn(20, 10).astype(np.float32)
        result = model.transform(new_data)
        assert result.shape == (20, 2)

    def test_transform_before_fit_raises(self):
        model = Parametric_UMAP(num_inputs=5, num_outputs=2)
        with pytest.raises(AssertionError, match="fit"):
            model.transform(np.random.randn(10, 5))

    def test_save_and_restore(self, small_data):
        model = Parametric_UMAP(
            num_inputs=10, num_outputs=2, n_neighbors=10,
            batch_size=32, seed=42,
        )
        model.fit(small_data, epochs=3)
        orig = model.transform(small_data[:10])

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save_model(path)
            model2 = Parametric_UMAP(
                num_inputs=10, num_outputs=2, n_neighbors=10,
                batch_size=32, seed=42,
            )
            model2.restore_model(path)
            restored = model2.transform(small_data[:10])
            np.testing.assert_allclose(orig, restored, atol=1e-6)
        finally:
            os.unlink(path)

    def test_custom_encoder(self, small_data):
        encoder = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
        model = Parametric_UMAP(
            num_inputs=10, num_outputs=2, n_neighbors=10,
            encoder=encoder, batch_size=32, seed=42,
        )
        result = model.fit_transform(small_data, epochs=3)
        assert result.shape == (128, 2)

    def test_inherits_parametric_dr(self):
        model = Parametric_UMAP(num_inputs=5, num_outputs=2)
        assert isinstance(model, ParametricDR)


class TestFitAbParams:
    def test_returns_positive(self):
        a, b = _fit_ab_params(0.1)
        assert a > 0
        assert b > 0

    def test_smaller_min_dist_larger_a(self):
        a_small, _ = _fit_ab_params(0.01)
        a_large, _ = _fit_ab_params(0.5)
        # Smaller min_dist should give a sharper kernel (larger a)
        assert a_small > a_large


class TestFuzzySimplicialSet:
    def test_symmetric(self):
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        indices, distances = compute_knn_graph(X, 10)
        graph = _compute_fuzzy_simplicial_set(indices, distances, 10)
        diff = abs(graph - graph.T).max()
        assert diff < 1e-10

    def test_nonnegative_weights(self):
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        indices, distances = compute_knn_graph(X, 10)
        graph = _compute_fuzzy_simplicial_set(indices, distances, 10)
        assert graph.min() >= 0
