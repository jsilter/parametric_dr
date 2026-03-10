"""Parametric UMAP: Uniform Manifold Approximation and Projection.

Fully self-contained (no umap-learn dependency). Learns a neural network
embedding by optimizing a cross-entropy loss over fuzzy simplicial complex
edge weights.

Reference:
McInnes, Healy, Melville (2018). UMAP: Uniform Manifold Approximation
and Projection for Dimension Reduction. arXiv:1802.03426.
"""

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix

from ._base import ParametricDR
from .utils import compute_knn_graph

DEFAULT_EPS = 1e-4


def _fit_ab_params(min_dist: float, spread: float = 1.0):
    """Fit a, b parameters for the UMAP distance kernel.

    The kernel is phi(d) = 1 / (1 + a * d^(2b)), fitted to approximate
    the piecewise target: 1.0 if d <= min_dist, else exp(-(d - min_dist) / spread).

    Returns
    -------
    a, b : float
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros_like(xv)
    yv[xv <= min_dist] = 1.0
    yv[xv > min_dist] = np.exp(-(xv[xv > min_dist] - min_dist) / spread)

    (a, b), _ = curve_fit(curve, xv, yv, p0=(1.0, 1.0), maxfev=5000)
    return float(a), float(b)


def _compute_fuzzy_simplicial_set(knn_indices, knn_distances, n_neighbors):
    """Compute the fuzzy simplicial set (symmetric edge weights).

    For each point i:
    - rho_i = distance to nearest neighbor
    - sigma_i found via binary search so that
      sum(exp(-(d_ij - rho_i) / sigma_i)) = log2(n_neighbors)
    - Edge weight w_ij = exp(-(d_ij - rho_i) / sigma_i)
    - Symmetrize: w_sym = w + w^T - w * w^T

    Parameters
    ----------
    knn_indices : 2-d array (N, K)
    knn_distances : 2-d array (N, K)
    n_neighbors : int

    Returns
    -------
    graph : scipy.sparse.csr_matrix (N, N)
        Symmetric fuzzy simplicial set.
    """
    n = knn_indices.shape[0]
    target = np.log2(n_neighbors)

    # rho: distance to nearest neighbor
    rho = knn_distances[:, 0].copy()
    rho = np.maximum(rho, 1e-8)

    # Binary search for sigma per point
    sigmas = np.ones(n)
    for i in range(n):
        lo, hi = 1e-8, 1000.0
        for _ in range(64):
            sigma = (lo + hi) / 2.0
            d_adj = np.maximum(knn_distances[i] - rho[i], 0.0)
            weights = np.exp(-d_adj / sigma)
            # Exclude self-contribution (distance 0 maps to weight 1)
            s = weights.sum() - 1.0 if knn_distances[i, 0] == 0.0 else weights.sum()
            if s > target:
                hi = sigma
            else:
                lo = sigma
            if abs(s - target) < 1e-5:
                break
        sigmas[i] = (lo + hi) / 2.0

    # Build sparse edge weight matrix
    rows, cols, vals = [], [], []
    for i in range(n):
        for j_pos in range(knn_indices.shape[1]):
            j = knn_indices[i, j_pos]
            if j == i:
                continue
            d_adj = max(knn_distances[i, j_pos] - rho[i], 0.0)
            w = np.exp(-d_adj / sigmas[i])
            rows.append(i)
            cols.append(j)
            vals.append(w)

    graph = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    # Symmetrize: w_sym = w + w^T - w * w^T
    transpose = graph.T.tocsr()
    sym = graph + transpose - graph.multiply(transpose)
    sym.eliminate_zeros()
    return sym


def umap_loss(pos_emb_a, pos_emb_b, neg_emb_a, neg_emb_b, a, b, eps=DEFAULT_EPS):
    """UMAP cross-entropy loss for positive and negative edges.

    Parameters
    ----------
    pos_emb_a, pos_emb_b : tensor (B, D)
        Embeddings of positive edge endpoints.
    neg_emb_a, neg_emb_b : tensor (B * neg_rate, D)
        Embeddings of negative edge endpoints.
    a, b : float
        Kernel parameters.
    eps : float

    Returns
    -------
    loss : scalar tensor
    """
    # Positive: attract (push phi toward 1, so -log(phi) is minimized)
    pos_d_sq = torch.sum((pos_emb_a - pos_emb_b) ** 2, dim=1).clamp(min=1e-10)
    phi_pos = 1.0 / (1.0 + a * pos_d_sq.pow(b))
    loss_attract = -torch.mean(torch.log(phi_pos.clamp(min=eps)))

    # Negative: repel (push phi toward 0, so -log(1-phi) is minimized)
    neg_d_sq = torch.sum((neg_emb_a - neg_emb_b) ** 2, dim=1).clamp(min=1e-10)
    phi_neg = 1.0 / (1.0 + a * neg_d_sq.pow(b))
    loss_repel = -torch.mean(torch.log((1.0 - phi_neg).clamp(min=eps)))

    return loss_attract + loss_repel


class Parametric_UMAP(ParametricDR):
    """Parametric UMAP dimensionality reduction.

    Fully self-contained; does not depend on umap-learn.

    Parameters
    ----------
    num_inputs : int
    num_outputs : int
    n_neighbors : int
        Number of neighbors for the fuzzy simplicial complex.
    min_dist : float
        Minimum distance in the embedding (controls tightness of clusters).
    spread : float
        Scale of the embedding.
    negative_sample_rate : int
        Number of negative samples per positive edge.
    learning_rate : float
    encoder : nn.Module or None
    seed : int
    batch_size : int
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        spread: float = 1.0,
        negative_sample_rate: int = 5,
        n_pca: int | None = 50,
        learning_rate: float = 1e-3,
        encoder=None,
        seed: int = 0,
        batch_size: int = 64,
    ):
        super().__init__(
            num_inputs, num_outputs, n_pca, learning_rate, encoder, seed, batch_size
        )
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.negative_sample_rate = negative_sample_rate

    def fit(self, X, y=None, epochs=100, verbose=0):
        """Train the parametric UMAP model.

        Parameters
        ----------
        X : 2-d array (N, num_inputs)
        y : ignored
        epochs : int
        verbose : int

        Returns
        -------
        self
        """
        X = self._process_training_data(X)
        n = X.shape[0]

        # Build KNN graph
        k = min(self.n_neighbors, n - 1)
        knn_indices, knn_distances = compute_knn_graph(X, k)

        # Compute fuzzy simplicial complex
        graph = _compute_fuzzy_simplicial_set(knn_indices, knn_distances, k)

        # Fit a, b kernel parameters
        a, b = _fit_ab_params(self.min_dist, self.spread)

        # Convert to edge list for sampling
        graph_coo = graph.tocoo()
        edges = np.column_stack([graph_coo.row, graph_coo.col])
        edge_weights = graph_coo.data.copy()
        edge_weights = edge_weights / edge_weights.sum()

        X_t = torch.tensor(X, dtype=torch.float32)
        opt = self._setup_training()
        n_edges = len(edges)
        edges_per_epoch = max(n_edges, n)

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Sample positive edges proportional to weight
            pos_idx = np.random.choice(
                n_edges, size=edges_per_epoch, p=edge_weights
            )

            for batch_start in range(0, edges_per_epoch, self.batch_size):
                batch_end = min(batch_start + self.batch_size, edges_per_epoch)
                batch_edge_idx = pos_idx[batch_start:batch_end]
                bs = len(batch_edge_idx)

                pos_edges = edges[batch_edge_idx]
                pos_a_emb = self.encoder(X_t[pos_edges[:, 0]])
                pos_b_emb = self.encoder(X_t[pos_edges[:, 1]])

                # Sample negatives
                neg_a_idx = np.repeat(pos_edges[:, 0], self.negative_sample_rate)
                neg_b_idx = np.random.randint(0, n, size=bs * self.negative_sample_rate)
                neg_a_emb = self.encoder(X_t[neg_a_idx])
                neg_b_emb = self.encoder(X_t[neg_b_idx])

                opt.zero_grad()
                loss = umap_loss(pos_a_emb, pos_b_emb, neg_a_emb, neg_b_emb, a, b)
                # _extra_loss uses anchor indices from positive edges
                loss = loss + self._extra_loss(pos_a_emb, pos_edges[:, 0])
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

            if verbose and n_batches > 0:
                avg = epoch_loss / n_batches
                print(f"  Epoch {epoch + 1}/{epochs}, loss={avg:.4f}")

        self._is_fitted = True
        return self
