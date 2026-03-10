"""Loss functions and probability distribution utilities for parametric t-SNE.

Contains the KL divergence loss and helper functions for computing
Q (Student-t) distributions in the low-dimensional embedding space.
"""

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required but not installed. Install CPU-only (~200MB) with:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "Or with CUDA support:\n"
        "  pip install torch"
    )

DEFAULT_EPS = 1e-7


def torch_set_diag(x: torch.Tensor, val: float) -> torch.Tensor:
    """Set diagonal of a 2D tensor to a given value.

    Parameters
    ----------
    x : 2-d tensor (N, M)
    val : float

    Returns
    -------
    result : 2-d tensor (N, M)
        Copy of x with diagonal set to val.
    """
    result = x.clone()
    n = min(result.shape[0], result.shape[1])
    result[torch.arange(n), torch.arange(n)] = val
    return result


def _get_squared_cross_diff_torch(x: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances.

    Z_ij = ||x_i - x_j||^2

    Parameters
    ----------
    x : 2-d tensor (N, D)

    Returns
    -------
    Z_ij : 2-d tensor (N, N)
    """
    diffs = x.unsqueeze(1) - x.unsqueeze(0)
    return torch.sum(diffs ** 2, dim=2)


def _get_normed_sym_torch(x: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Normalize and symmetrize a probability matrix.

    Parameters
    ----------
    x : 2-d tensor (N, N)
        Asymmetric probabilities.
    eps : float
        Prevents division by zero.

    Returns
    -------
    P : 2-d tensor (N, N)
        Symmetric probabilities with zero diagonal.
    """
    P = torch_set_diag(x, 0.0)
    norm_facs = torch.sum(P, dim=0, keepdim=True)
    P = P / (norm_facs + eps)
    P = 0.5 * (P + P.t())
    return P


def _make_Q(output: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute the Q probability distribution using the Student-t kernel.

    Parameters
    ----------
    output : 2-d tensor (N, output_dims)
        Low-dimensional embedding from the neural network.
    alpha : float
        Degrees of freedom parameter. Recommend output_dims - 1.

    Returns
    -------
    Q : 2-d tensor (N, N)
        Symmetric Q distribution.
    """
    out_sq_diffs = _get_squared_cross_diff_torch(output)
    Q = torch.pow((1 + out_sq_diffs / alpha), -(alpha + 1) / 2)
    Q = _get_normed_sym_torch(Q)
    return Q


def kl_loss(
    P: torch.Tensor,
    y_pred: torch.Tensor,
    alpha: float = 1.0,
    num_perplexities: int = 1,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """KL divergence loss between P (high-dim) and Q (low-dim) distributions.

    Parameters
    ----------
    P : 2-d tensor (N, N*num_perplexities)
        Gaussian similarity matrix from high-dimensional data.
        Multiple perplexity P matrices are concatenated along dim 1.
    y_pred : 2-d tensor (N, output_dims)
        Neural network output (low-dimensional embedding).
    alpha : float
        Student-t distribution parameter.
    num_perplexities : int
        Number of perplexity values stacked along axis 1 of P.
    eps : float
        Prevents log(0).

    Returns
    -------
    loss : scalar tensor
        Sum of KL divergences across all perplexities.
    """
    Q = _make_Q(y_pred, alpha)
    batch_size = P.shape[0]

    kls = []
    split_size = batch_size
    components = torch.split(P, split_size, dim=1)
    for cur_P in components:
        kl_matr = cur_P * (torch.log(cur_P + eps) - torch.log(Q + eps))
        kl_matr = torch_set_diag(kl_matr, 0.0)
        kls.append(torch.sum(kl_matr))

    return torch.sum(torch.stack(kls))
