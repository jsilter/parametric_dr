import tempfile
import os

import numpy as np
import pytest

from parametric_dr import Parametric_tSNE


@pytest.fixture
def small_data():
    np.random.seed(42)
    N, D = 128, 10
    X = np.random.randn(N, D)
    return X


class TestParametricTSNE:
    def test_fit_transform_shape(self, small_data):
        model = Parametric_tSNE(
            num_inputs=10,
            num_outputs=2,
            perplexities=10.0,
            batch_size=32,
            seed=42,
        )
        result = model.fit_transform(small_data, epochs=2)
        assert result.shape == (128, 2)

    def test_transform_after_fit(self, small_data):
        model = Parametric_tSNE(
            num_inputs=10,
            num_outputs=2,
            perplexities=10.0,
            batch_size=32,
            seed=42,
        )
        model.fit(small_data, epochs=2)
        new_data = np.random.randn(20, 10)
        result = model.transform(new_data)
        assert result.shape == (20, 2)

    def test_transform_before_fit_raises(self):
        model = Parametric_tSNE(
            num_inputs=5, num_outputs=2, perplexities=5.0
        )
        with pytest.raises(AssertionError, match="fit"):
            model.transform(np.random.randn(10, 5))

    def test_wrong_input_dim_raises(self, small_data):
        model = Parametric_tSNE(
            num_inputs=5, num_outputs=2, perplexities=5.0
        )
        with pytest.raises(AssertionError):
            model.fit(small_data)  # small_data has 10 dims, not 5

    def test_multi_perplexity(self, small_data):
        model = Parametric_tSNE(
            num_inputs=10,
            num_outputs=2,
            perplexities=[5.0, 20.0],
            batch_size=32,
            seed=42,
        )
        result = model.fit_transform(small_data, epochs=2)
        assert result.shape == (128, 2)

    def test_save_and_restore(self, small_data):
        model = Parametric_tSNE(
            num_inputs=10,
            num_outputs=2,
            perplexities=10.0,
            batch_size=32,
            seed=42,
        )
        model.fit(small_data, epochs=2)
        orig_output = model.transform(small_data[:10])

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            model_path = f.name

        try:
            model.save_model(model_path)

            model2 = Parametric_tSNE(
                num_inputs=10,
                num_outputs=2,
                perplexities=10.0,
                batch_size=32,
                seed=42,
            )
            model2.restore_model(model_path)
            restored_output = model2.transform(small_data[:10])

            np.testing.assert_allclose(orig_output, restored_output, atol=1e-6)
        finally:
            os.unlink(model_path)

    def test_custom_encoder(self, small_data):
        import torch.nn as nn

        encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        model = Parametric_tSNE(
            num_inputs=10,
            num_outputs=2,
            perplexities=10.0,
            encoder=encoder,
            batch_size=32,
            seed=42,
        )
        result = model.fit_transform(small_data, epochs=2)
        assert result.shape == (128, 2)

    def test_sklearn_get_params(self):
        model = Parametric_tSNE(
            num_inputs=10, num_outputs=2, perplexities=10.0
        )
        params = model.get_params()
        assert params["num_inputs"] == 10
        assert params["num_outputs"] == 2
        assert params["alpha"] == 1.0

    def test_inherits_parametric_dr(self):
        from parametric_dr._base import ParametricDR

        model = Parametric_tSNE(num_inputs=5, num_outputs=2, perplexities=5.0)
        assert isinstance(model, ParametricDR)
