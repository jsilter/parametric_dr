import numpy as np
import pandas as pd
import pytest

from parametric_dr import Parametric_PaCMAP
from parametric_dr.analysis import _elbow_detection, estimate_intrinsic_dim, sweep_dims


# ---------------------------------------------------------------------------
# _elbow_detection
# ---------------------------------------------------------------------------


class TestElbowDetection:
    def test_classic_elbow_shape(self):
        """Sharp rise then plateau; elbow should be at the knee (index 2)."""
        curve = np.array([0.5, 0.8, 0.95, 0.96, 0.97])
        assert _elbow_detection(curve) == 2

    def test_concave_diminishing_returns(self):
        """Log-like growth curve; elbow should be near the steep-to-flat transition."""
        curve = np.array([0.0, 0.6, 0.85, 0.93, 0.96, 0.98, 0.99])
        idx = _elbow_detection(curve)
        # The maximum curvature is around index 1-2
        assert idx in (1, 2)

    def test_convex_then_concave(self):
        """Slow start then jump: the elbow (max perpendicular distance from
        the endpoint line) is where the curve departs most from linear."""
        curve = np.array([0.1, 0.12, 0.15, 0.5, 0.9, 0.92])
        idx = _elbow_detection(curve)
        # Point at index 2 (0.15) is farthest below the line from (0,0.1)
        # to (5,0.92), which is the correct perpendicular-distance elbow.
        assert idx == 2

    def test_single_element(self):
        assert _elbow_detection(np.array([0.9])) == 0

    def test_two_elements(self):
        assert _elbow_detection(np.array([0.5, 0.9])) == 1

    def test_flat_curve(self):
        """All points lie on the endpoint line; all perpendicular distances
        are zero, so argmax returns 0."""
        curve = np.array([0.9, 0.9, 0.9, 0.9])
        assert _elbow_detection(curve) == 0


# ---------------------------------------------------------------------------
# estimate_intrinsic_dim
# ---------------------------------------------------------------------------


class TestEstimateIntrinsicDim:
    def test_unsupported_method_raises(self):
        pytest.importorskip("skdim")
        with pytest.raises(ValueError, match="Unsupported method"):
            estimate_intrinsic_dim(np.random.randn(50, 5), method="bogus")

    def test_twonn_estimates_near_true_dim(self):
        pytest.importorskip("skdim")
        np.random.seed(0)
        # 3D manifold embedded in 10D; estimate should be near 3
        X_3d = np.random.randn(500, 3)
        W = np.random.randn(3, 10)
        X = X_3d @ W
        result = estimate_intrinsic_dim(X, method="twonn")
        assert result["method"] == "twonn"
        assert 2.0 <= result["estimate"] <= 5.0

    def test_skdim_missing_raises_import_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "skdim.id" or name == "skdim":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="skdim"):
            estimate_intrinsic_dim(np.random.randn(50, 5))


# ---------------------------------------------------------------------------
# sweep_dims
# ---------------------------------------------------------------------------


class _FastPaCMAP(Parametric_PaCMAP):
    """PaCMAP that trains for only 1 epoch (for fast tests)."""

    def fit(self, X, y=None, epochs=1):
        return super().fit(X, y=y, epochs=epochs)


class TestSweepDims:
    @pytest.fixture
    def clustered_data(self):
        """Two well-separated clusters in 10D. Even a weak model should
        achieve decent trustworthiness on this."""
        np.random.seed(42)
        c0 = np.random.randn(60, 10).astype(np.float32) + 5
        c1 = np.random.randn(60, 10).astype(np.float32) - 5
        return np.vstack([c0, c1])

    def _run_sweep(self, data, dims=(2, 3), plot=False, **extra):
        defaults = dict(
            k=5, n_subsample=None, plot=plot,
            num_inputs=10, n_pca=None, batch_size=32, seed=42, n_neighbors=5,
        )
        defaults.update(extra)
        return sweep_dims(data, _FastPaCMAP, dims=dims, **defaults)

    def test_dataframe_structure_and_metrics(self, clustered_data):
        """Verify returned DataFrame has one row per dim with all metric columns,
        and that well-separated clusters produce trustworthiness > 0.7."""
        dims = (2, 3, 4)
        result = self._run_sweep(clustered_data, dims=dims)

        df = result["results"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df["dim"]) == [2, 3, 4]

        for _, row in df.iterrows():
            assert 0.0 < row["trustworthiness"] <= 1.0
            assert 0.0 < row["continuity"] <= 1.0
            # Well-separated clusters should be easy to embed
            assert row["trustworthiness"] > 0.7

        assert result["elbow_dim"] in dims

    def test_subsample_uses_fewer_points(self, clustered_data):
        """With n_subsample < N, metrics should still be computed (on the subset).
        Run both with and without subsampling; both should produce valid results
        but may differ in value."""
        np.random.seed(99)
        r_full = self._run_sweep(clustered_data, dims=(2,), n_subsample=None)
        np.random.seed(99)
        r_sub = self._run_sweep(clustered_data, dims=(2,), n_subsample=30)

        t_full = r_full["results"].iloc[0]["trustworthiness"]
        t_sub = r_sub["results"].iloc[0]["trustworthiness"]

        # Both should be valid
        assert 0.0 < t_full <= 1.0
        assert 0.0 < t_sub <= 1.0
        # With different sample sizes the exact values will differ
        # (unless by coincidence); just verify both ran without error.

    def test_plot_produces_figure_with_axes(self, clustered_data):
        matplotlib = pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        result = self._run_sweep(clustered_data, dims=(2, 3), plot=True)
        fig = result["figure"]
        assert isinstance(fig, matplotlib.figure.Figure)
        # Should have 3 subplots: trustworthiness, continuity, shepard
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_plot_false_returns_no_figure(self, clustered_data):
        result = self._run_sweep(clustered_data, dims=(2,), plot=False)
        assert result["figure"] is None
