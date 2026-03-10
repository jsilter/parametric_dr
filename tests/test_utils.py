import numpy as np
import pytest

from parametric_dr.utils import (
    calc_betas_loop,
    get_squared_cross_diff_np,
    get_multiscale_perplexities,
    get_Lmax,
    Hbeta_scalar,
    Hbeta_vec,
    compute_knn_graph,
)


class TestGetSquaredCrossDiff:
    def test_shape(self):
        x = np.random.randn(10, 5)
        result = get_squared_cross_diff_np(x)
        assert result.shape == (10, 10)

    def test_symmetric(self):
        x = np.random.randn(8, 3)
        result = get_squared_cross_diff_np(x)
        np.testing.assert_allclose(result, result.T)

    def test_zero_diagonal(self):
        x = np.random.randn(6, 4)
        result = get_squared_cross_diff_np(x)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-10)

    def test_known_values(self):
        x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        result = get_squared_cross_diff_np(x)
        expected = np.array([[0, 1, 1], [1, 0, 2], [1, 2, 0]], dtype=float)
        np.testing.assert_allclose(result, expected)


class TestHbetaScalar:
    def test_probabilities_sum_to_one(self):
        distances = np.array([0.0, 1.0, 2.0, 3.0])
        beta = 1.0
        H, p = Hbeta_scalar(distances, beta)
        np.testing.assert_allclose(np.sum(p), 1.0)

    def test_higher_beta_concentrates(self):
        distances = np.array([0.0, 1.0, 4.0, 9.0])
        _, p_low = Hbeta_scalar(distances, 0.1)
        _, p_high = Hbeta_scalar(distances, 10.0)
        # Higher beta should give more weight to nearest neighbor
        assert p_high[0] > p_low[0]


class TestCalcBetasLoop:
    def test_returns_correct_shapes(self):
        data = np.random.randn(20, 5)
        betas, Hs, p_matr = calc_betas_loop(data, perplexity=5.0)
        assert betas.shape == (20,)
        assert Hs.shape == (20,)
        assert p_matr.shape == (20, 20)

    def test_betas_positive(self):
        data = np.random.randn(15, 3)
        betas, _, _ = calc_betas_loop(data, perplexity=5.0)
        assert np.all(betas > 0)

    def test_p_matr_rows_sum_to_one(self):
        data = np.random.randn(20, 4)
        _, _, p_matr = calc_betas_loop(data, perplexity=5.0)
        row_sums = np.sum(p_matr, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


class TestComputeKnnGraph:
    def test_shape(self):
        X = np.random.randn(50, 5).astype(np.float32)
        indices, distances = compute_knn_graph(X, 10)
        assert indices.shape == (50, 10)
        assert distances.shape == (50, 10)

    def test_excludes_self(self):
        X = np.random.randn(30, 3).astype(np.float32)
        indices, distances = compute_knn_graph(X, 5)
        # No point should be its own neighbor
        for i in range(30):
            assert i not in indices[i]

    def test_distances_sorted(self):
        X = np.random.randn(40, 4).astype(np.float32)
        _, distances = compute_knn_graph(X, 8)
        for i in range(40):
            assert np.all(np.diff(distances[i]) >= -1e-10)

    def test_distances_nonnegative(self):
        X = np.random.randn(20, 3).astype(np.float32)
        _, distances = compute_knn_graph(X, 5)
        assert np.all(distances >= 0)


class TestMultiscalePerplexities:
    def test_returns_powers_of_two(self):
        perps = get_multiscale_perplexities(1000)
        for p in perps:
            assert p == int(p)
            assert (np.log2(p) % 1) == 0

    def test_larger_data_more_perplexities(self):
        perps_small = get_multiscale_perplexities(100)
        perps_large = get_multiscale_perplexities(10000)
        assert len(perps_large) > len(perps_small)

    def test_get_Lmax(self):
        assert get_Lmax(16) == 2
        assert get_Lmax(64) == 4
