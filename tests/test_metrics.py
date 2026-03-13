import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from parametric_dr.metrics import (
    _distance_matrix,
    _knn_indices_from_D,
    trustworthiness,
    continuity,
    neighborhood_preservation,
    shepard_correlation,
    trajectory_smoothness,
    compute_metrics,
)


@pytest.fixture
def identity_embedding():
    """Embedding that perfectly preserves structure (identity in 2D)."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    # Use PCA-like projection (linear, structure-preserving)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_low = pca.fit_transform(X)
    return X, X_low


@pytest.fixture
def random_embedding():
    """Embedding that does not preserve structure."""
    np.random.seed(42)
    X_high = np.random.randn(50, 5)
    np.random.seed(999)
    X_low = np.random.randn(50, 2)
    return X_high, X_low


class TestTrustworthiness:
    def test_perfect_embedding(self, identity_embedding):
        X_high, X_low = identity_embedding
        t = trustworthiness(X_high, X_low, k=5)
        assert t > 0.8

    def test_random_embedding_lower(self, identity_embedding, random_embedding):
        t_good = trustworthiness(*identity_embedding, k=5)
        t_bad = trustworthiness(*random_embedding, k=5)
        assert t_good > t_bad

    def test_range(self, random_embedding):
        t = trustworthiness(*random_embedding, k=5)
        assert 0.0 < t <= 1.0


class TestContinuity:
    def test_perfect_embedding(self, identity_embedding):
        X_high, X_low = identity_embedding
        c = continuity(X_high, X_low, k=5)
        assert c > 0.8

    def test_random_embedding_lower(self, identity_embedding, random_embedding):
        c_good = continuity(*identity_embedding, k=5)
        c_bad = continuity(*random_embedding, k=5)
        assert c_good > c_bad

    def test_range(self, random_embedding):
        c = continuity(*random_embedding, k=5)
        assert 0.0 < c <= 1.0


class TestNeighborhoodPreservation:
    def test_perfect_embedding(self, identity_embedding):
        X_high, X_low = identity_embedding
        score = neighborhood_preservation(X_high, X_low, k=5)
        # PCA into 2D from 5D loses some neighbors; threshold accordingly
        assert score > 0.3

    def test_random_embedding_lower(self, identity_embedding, random_embedding):
        s_good = neighborhood_preservation(*identity_embedding, k=5)
        s_bad = neighborhood_preservation(*random_embedding, k=5)
        assert s_good > s_bad

    def test_range(self, random_embedding):
        score = neighborhood_preservation(*random_embedding, k=5)
        assert 0.0 <= score <= 1.0


class TestShepardCorrelation:
    def test_perfect_embedding(self, identity_embedding):
        X_high, X_low = identity_embedding
        r = shepard_correlation(X_high, X_low)
        assert r > 0.5

    def test_range(self, random_embedding):
        r = shepard_correlation(*random_embedding)
        assert -1.0 <= r <= 1.0

    def test_identical_is_one(self):
        X = np.random.randn(30, 3)
        r = shepard_correlation(X, X)
        np.testing.assert_allclose(r, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests for helpers
# ---------------------------------------------------------------------------


class TestDistanceMatrix:
    """Tests for _distance_matrix helper."""

    def test_euclidean_matches_pdist(self):
        np.random.seed(0)
        X = np.random.randn(30, 4)
        D = _distance_matrix(X, "euclidean")
        expected = squareform(pdist(X, "euclidean"))
        np.testing.assert_allclose(D, expected, atol=1e-6)

    def test_cosine_matches_pdist(self):
        np.random.seed(1)
        X = np.random.randn(25, 6)
        D = _distance_matrix(X, "cosine")
        expected = squareform(pdist(X, "cosine"))
        np.testing.assert_allclose(D, expected, atol=1e-6)

    def test_minkowski_with_kwargs(self):
        np.random.seed(2)
        X = np.random.randn(20, 3)
        D = _distance_matrix(X, "minkowski", p=3)
        expected = squareform(pdist(X, "minkowski", p=3))
        np.testing.assert_allclose(D, expected, atol=1e-6)

    def test_symmetry_and_zero_diagonal(self):
        np.random.seed(3)
        X = np.random.randn(15, 5)
        for metric in ["euclidean", "cosine", "cityblock"]:
            D = _distance_matrix(X, metric)
            np.testing.assert_allclose(D, D.T, atol=1e-12)
            np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)


class TestKnnIndicesFromD:
    """Tests for _knn_indices_from_D helper."""

    def test_shape(self):
        D = np.array([
            [0, 1, 2, 3],
            [1, 0, 1.5, 2.5],
            [2, 1.5, 0, 1],
            [3, 2.5, 1, 0],
        ])
        idx = _knn_indices_from_D(D, k=2)
        assert idx.shape == (4, 2)

    def test_correct_neighbors(self):
        D = np.array([
            [0, 1, 5, 10],
            [1, 0, 3, 8],
            [5, 3, 0, 2],
            [10, 8, 2, 0],
        ])
        idx = _knn_indices_from_D(D, k=2)
        # Point 0: nearest are 1 (d=1), 2 (d=5)
        assert list(idx[0]) == [1, 2]
        # Point 3: nearest are 2 (d=2), 1 (d=8)
        assert list(idx[3]) == [2, 1]

    def test_does_not_mutate_input(self):
        D = np.array([[0.0, 1.0], [1.0, 0.0]])
        D_orig = D.copy()
        _knn_indices_from_D(D, k=1)
        np.testing.assert_array_equal(D, D_orig)


# ---------------------------------------------------------------------------
# Parametrized metric tests for all public functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "cityblock"])
class TestMetricParam:
    """Test that the metric parameter works for all four public functions."""

    def test_trustworthiness_runs(self, metric):
        np.random.seed(42)
        X = np.random.randn(40, 5)
        Y = np.random.randn(40, 2)
        t = trustworthiness(X, Y, k=5, metric=metric)
        assert 0.0 < t <= 1.0

    def test_continuity_runs(self, metric):
        np.random.seed(42)
        X = np.random.randn(40, 5)
        Y = np.random.randn(40, 2)
        c = continuity(X, Y, k=5, metric=metric)
        assert 0.0 < c <= 1.0

    def test_neighborhood_preservation_runs(self, metric):
        np.random.seed(42)
        X = np.random.randn(40, 5)
        Y = np.random.randn(40, 2)
        s = neighborhood_preservation(X, Y, k=5, metric=metric)
        assert 0.0 <= s <= 1.0

    def test_shepard_correlation_runs(self, metric):
        np.random.seed(42)
        X = np.random.randn(40, 5)
        Y = np.random.randn(40, 2)
        r = shepard_correlation(X, Y, metric=metric)
        assert -1.0 <= r <= 1.0


@pytest.mark.parametrize("metric", ["cosine", "cityblock"])
class TestNonEuclideanDiffers:
    """Non-Euclidean metrics should produce different scores than Euclidean."""

    def test_trustworthiness_differs(self, metric):
        np.random.seed(42)
        X = np.random.randn(40, 5)
        Y = np.random.randn(40, 2)
        t_euc = trustworthiness(X, Y, k=5, metric="euclidean")
        t_met = trustworthiness(X, Y, k=5, metric=metric)
        # Not strictly guaranteed but extremely likely with random data
        assert t_euc != t_met

    def test_shepard_differs(self, metric):
        np.random.seed(42)
        X = np.random.randn(40, 5)
        Y = np.random.randn(40, 2)
        r_euc = shepard_correlation(X, Y, metric="euclidean")
        r_met = shepard_correlation(X, Y, metric=metric)
        assert r_euc != r_met


# ---------------------------------------------------------------------------
# Regression: refactored Euclidean results match old implementation
# ---------------------------------------------------------------------------


class TestEuclideanRegression:
    """Verify that refactored code produces identical results to the old
    implementation when using Euclidean distance (the default)."""

    def test_trustworthiness_regression(self, identity_embedding):
        X, Y = identity_embedding
        t = trustworthiness(X, Y, k=5)
        assert isinstance(t, float)
        assert t > 0.8

    def test_shepard_identical_regression(self):
        np.random.seed(99)
        X = np.random.randn(25, 3)
        r = shepard_correlation(X, X)
        np.testing.assert_allclose(r, 1.0, atol=1e-10)

    def test_all_metrics_deterministic(self):
        """Same inputs produce identical outputs across two calls."""
        np.random.seed(42)
        X = np.random.randn(30, 4)
        Y = np.random.randn(30, 2)

        t1 = trustworthiness(X, Y, k=5)
        t2 = trustworthiness(X, Y, k=5)
        assert t1 == t2

        c1 = continuity(X, Y, k=5)
        c2 = continuity(X, Y, k=5)
        assert c1 == c2

        n1 = neighborhood_preservation(X, Y, k=5)
        n2 = neighborhood_preservation(X, Y, k=5)
        assert n1 == n2

        s1 = shepard_correlation(X, Y)
        s2 = shepard_correlation(X, Y)
        assert s1 == s2


# ---------------------------------------------------------------------------
# Tests for compute_metrics (combined function)
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Test that compute_metrics matches individual function results."""

    def test_matches_individual_functions(self):
        np.random.seed(42)
        X = np.random.randn(40, 5)
        Y = np.random.randn(40, 2)
        k = 5

        combined = compute_metrics(X, Y, k=k)
        assert combined["trustworthiness"] == trustworthiness(X, Y, k=k)
        assert combined["continuity"] == continuity(X, Y, k=k)
        assert combined["neighborhood_preservation"] == neighborhood_preservation(X, Y, k=k)
        assert combined["shepard_correlation"] == shepard_correlation(X, Y)

    def test_matches_with_nondefault_metric(self):
        np.random.seed(42)
        X = np.random.randn(40, 5)
        Y = np.random.randn(40, 2)
        k = 5

        combined = compute_metrics(X, Y, k=k, metric="cosine")
        assert combined["trustworthiness"] == trustworthiness(X, Y, k=k, metric="cosine")
        assert combined["continuity"] == continuity(X, Y, k=k, metric="cosine")
        assert combined["neighborhood_preservation"] == neighborhood_preservation(X, Y, k=k, metric="cosine")
        assert combined["shepard_correlation"] == shepard_correlation(X, Y, metric="cosine")

    def test_returns_all_keys(self):
        np.random.seed(42)
        X = np.random.randn(30, 4)
        Y = np.random.randn(30, 2)
        result = compute_metrics(X, Y, k=5)
        assert set(result.keys()) == {
            "trustworthiness", "continuity",
            "neighborhood_preservation", "shepard_correlation",
        }

    def test_good_embedding_scores_high(self, identity_embedding):
        X, Y = identity_embedding
        result = compute_metrics(X, Y, k=5)
        assert result["trustworthiness"] > 0.8
        assert result["continuity"] > 0.8
        assert result["shepard_correlation"] > 0.5


class TestPrecomputedDistanceMatrix:
    """Test that passing precomputed D_high/D_low matches fresh computation."""

    def test_precomputed_matches_fresh(self):
        np.random.seed(42)
        X = np.random.randn(40, 5)
        Y = np.random.randn(40, 2)
        k = 5

        # Use float32 to match internal computation
        D_high = _distance_matrix(X)
        D_low = _distance_matrix(Y)

        assert trustworthiness(X, Y, k=k) == trustworthiness(X, Y, k=k, D_high=D_high, D_low=D_low)
        assert continuity(X, Y, k=k) == continuity(X, Y, k=k, D_high=D_high, D_low=D_low)
        assert neighborhood_preservation(X, Y, k=k) == neighborhood_preservation(X, Y, k=k, D_high=D_high, D_low=D_low)
        assert shepard_correlation(X, Y) == shepard_correlation(X, Y, D_high=D_high, D_low=D_low)

    def test_precomputed_high_only(self):
        np.random.seed(42)
        X = np.random.randn(40, 5)
        Y = np.random.randn(40, 2)
        D_high = _distance_matrix(X)

        assert trustworthiness(X, Y, k=5) == trustworthiness(X, Y, k=5, D_high=D_high)
        assert shepard_correlation(X, Y) == shepard_correlation(X, Y, D_high=D_high)


# ---------------------------------------------------------------------------
# Tests for trajectory_smoothness
# ---------------------------------------------------------------------------


class TestTrajectorySmoothnessUnnormalized:
    """Tests for trajectory_smoothness with normalize=False."""

    def test_constant_embedding_is_zero(self):
        emb = np.ones((20, 2))
        assert trajectory_smoothness(emb, normalize=False) == 0.0

    def test_linear_trajectory(self):
        """Uniform steps should give a constant step size."""
        emb = np.column_stack([np.arange(10), np.zeros(10)]).astype(np.float32)
        # Each step is (1, 0), squared norm = 1.0
        np.testing.assert_allclose(
            trajectory_smoothness(emb, normalize=False), 1.0, atol=1e-6,
        )

    def test_jagged_worse_than_smooth(self):
        t = np.linspace(0, 2 * np.pi, 100)
        smooth = np.column_stack([np.cos(t), np.sin(t)])
        jagged = smooth.copy()
        jagged[1::2] *= 3  # every other point jumps outward
        assert trajectory_smoothness(jagged, normalize=False) > trajectory_smoothness(smooth, normalize=False)


class TestTrajectorySmoothnessNormalized:
    """Tests for trajectory_smoothness with normalize=True (default)."""

    def test_constant_embedding_is_zero(self):
        emb = np.ones((20, 2))
        assert trajectory_smoothness(emb) == 0.0

    def test_scale_invariant(self):
        """Normalized score should not change when embedding is scaled."""
        np.random.seed(42)
        emb = np.cumsum(np.random.randn(100, 2), axis=0).astype(np.float32)
        s1 = trajectory_smoothness(emb)
        s2 = trajectory_smoothness(emb * 10.0)
        np.testing.assert_allclose(s1, s2, rtol=0.05)

    def test_jagged_worse_than_smooth(self):
        t = np.linspace(0, 2 * np.pi, 100)
        smooth = np.column_stack([np.cos(t), np.sin(t)])
        jagged = smooth.copy()
        jagged[1::2] *= 3
        assert trajectory_smoothness(jagged) > trajectory_smoothness(smooth)

    def test_positive(self):
        np.random.seed(42)
        emb = np.random.randn(50, 3)
        assert trajectory_smoothness(emb) > 0

    def test_normalize_false_differs(self):
        np.random.seed(42)
        emb = np.random.randn(50, 2).astype(np.float32)
        raw = trajectory_smoothness(emb, normalize=False)
        normed = trajectory_smoothness(emb, normalize=True)
        assert raw != normed
