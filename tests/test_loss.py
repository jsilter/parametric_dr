import numpy as np
import torch
import pytest

from parametric_dr.loss import (
    kl_loss,
    _make_Q,
    _get_squared_cross_diff_torch,
    _get_normed_sym_torch,
    torch_set_diag,
)


class TestTorchSetDiag:
    def test_sets_diagonal(self):
        x = torch.ones(3, 3)
        result = torch_set_diag(x, 0.0)
        expected = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_does_not_modify_original(self):
        x = torch.ones(3, 3)
        _ = torch_set_diag(x, 0.0)
        assert x[0, 0] == 1.0


class TestSquaredCrossDiff:
    def test_shape(self):
        x = torch.randn(10, 5)
        result = _get_squared_cross_diff_torch(x)
        assert result.shape == (10, 10)

    def test_symmetric(self):
        x = torch.randn(8, 3)
        result = _get_squared_cross_diff_torch(x)
        torch.testing.assert_close(result, result.T)

    def test_zero_diagonal(self):
        x = torch.randn(6, 4)
        result = _get_squared_cross_diff_torch(x)
        torch.testing.assert_close(
            torch.diag(result), torch.zeros(6), atol=1e-6, rtol=0
        )

    def test_matches_numpy(self):
        from parametric_dr.utils import get_squared_cross_diff_np

        data = np.random.randn(12, 5).astype(np.float32)
        np_result = get_squared_cross_diff_np(data)
        torch_result = _get_squared_cross_diff_torch(torch.tensor(data)).numpy()
        np.testing.assert_allclose(np_result, torch_result, atol=1e-5)


class TestNormedSym:
    def test_symmetric(self):
        x = torch.rand(6, 6)
        result = _get_normed_sym_torch(x)
        torch.testing.assert_close(result, result.T)

    def test_zero_diagonal(self):
        x = torch.rand(5, 5)
        result = _get_normed_sym_torch(x)
        torch.testing.assert_close(
            torch.diag(result), torch.zeros(5), atol=1e-7, rtol=0
        )


class TestMakeQ:
    def test_symmetric(self):
        output = torch.randn(10, 2)
        Q = _make_Q(output, alpha=1.0)
        torch.testing.assert_close(Q, Q.T)

    def test_zero_diagonal(self):
        output = torch.randn(10, 2)
        Q = _make_Q(output, alpha=1.0)
        torch.testing.assert_close(
            torch.diag(Q), torch.zeros(10), atol=1e-7, rtol=0
        )

    def test_nonnegative(self):
        output = torch.randn(10, 2)
        Q = _make_Q(output, alpha=1.0)
        assert torch.all(Q >= 0)


class TestKLLoss:
    def test_zero_for_identical_distributions(self):
        """KL(P || P) should be approximately zero."""
        output = torch.randn(8, 2)
        Q = _make_Q(output, alpha=1.0)
        # Use Q as both P and compute loss; should be ~0
        loss = kl_loss(Q, output, alpha=1.0, num_perplexities=1)
        assert loss.item() < 1e-4

    def test_nonnegative_with_valid_P(self):
        """KL should be non-negative when P is a proper distribution from _make_Q."""
        batch_size = 16
        # Use a proper Q-style distribution as P (normalized, symmetric, zero diag)
        from parametric_dr.loss import _make_Q
        source = torch.randn(batch_size, 5)
        P = _make_Q(source, alpha=1.0).detach()
        output = torch.randn(batch_size, 2)
        loss = kl_loss(P, output, alpha=1.0, num_perplexities=1)
        assert loss.item() >= -1e-4

    def test_multi_perplexity(self):
        batch_size = 10
        num_perp = 3
        P = torch.rand(batch_size, batch_size * num_perp)
        output = torch.randn(batch_size, 2)
        loss = kl_loss(P, output, alpha=1.0, num_perplexities=num_perp)
        assert torch.isfinite(loss)

    def test_gradient_flows(self):
        batch_size = 8
        P = torch.rand(batch_size, batch_size)
        output = torch.randn(batch_size, 2, requires_grad=True)
        loss = kl_loss(P, output, alpha=1.0, num_perplexities=1)
        loss.backward()
        assert output.grad is not None
        assert torch.all(torch.isfinite(output.grad))
