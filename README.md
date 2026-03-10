# Overview
Multi-method parametric dimensionality reduction using PyTorch. Train a neural network to learn a low-dimensional embedding, then reuse the trained model to transform new data instantly.

Supported methods: **t-SNE**, **UMAP**, **PaCMAP**, **TriMap**, and **CEBRA**. All methods share a common API (sklearn-compatible fit/transform) and support custom encoder architectures, optional PCA preprocessing, and model serialization. A **TemporalMixin** adds temporal smoothness regularization to any method.

By default the encoder is a dense network with layers [input_dim, 500, 500, 2000, output_dim], following van der Maaten 2009<sup>1</sup>.

# Installation

```commandline
git clone git@github.com:jsilter/parametric_dr.git
cd parametric_dr
pip install -e .
```

# Usage

Simple example usage may be:

```python
from parametric_dr import Parametric_tSNE

train_data = load_my_training_data()

high_dims = train_data.shape[1]
num_outputs = 2
perplexity = 30
ptSNE = Parametric_tSNE(high_dims, num_outputs, perplexity)
ptSNE.fit(train_data)
output_res = ptSNE.transform(train_data)
```

`output_res` will be `N x num_outputs`, the transformation of each point.
At this point, `ptSNE` will be a trained model, so we can quickly transform other data:

```python
test_data = load_my_test_data()
test_res = ptSNE.transform(test_data)
```

See the [examples/](./examples/) directory for complete scripts comparing methods on various datasets:

- **example_synthetic_clusters.py** - Synthetic clustered data in 14 dimensions; compares all methods including temporal variants
- **example_digits.py** - sklearn handwritten digits (64 features, 10 classes)
- **example_olivetti_faces.py** - Olivetti face images (4096 features, 40 individuals)
- **example_lorenz.py** - Lorenz attractor projected to 20 dimensions; static vs temporal methods
- **example_neural_timecourse.py** - Simulated place cell population on a circular track
- **example_hematopoiesis.py** - Paul et al. 2015 scRNA-seq differentiation data (requires scanpy)

Each example trains multiple methods, computes quality metrics (trustworthiness, continuity, neighborhood preservation, Shepard correlation), reports fit times, and generates a multi-page PDF with visualizations.

To use a custom encoder architecture, pass a PyTorch `nn.Module`:

```python
from parametric_dr import Parametric_tSNE
import torch.nn as nn

train_data = load_my_training_data()
high_dims = train_data.shape[1]
num_outputs = 2
perplexity = 30

encoder = nn.Sequential(
    nn.Linear(high_dims, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, num_outputs),
)
ptSNE = Parametric_tSNE(high_dims, num_outputs, perplexity, encoder=encoder)
```

If the dimensionality is large (>100), it is recommended to apply PCA first (see the `n_pca` parameter).

## t-SNE notes

The `perplexity` parameter can also be a list (e.g. `[10, 20, 30, 50, 100, 200]`), in which case the total loss is a sum over each perplexity value. This multiscale approach is inspired by Lee et al. 2014. Initialization time scales linearly with the number of perplexity values, though inference speed is unaffected.

The default output layer is linear rather than ReLU (as in van der Maaten 2009<sup>1</sup>); ReLU occasionally produced degenerate embeddings with all-zero dimensions.

# References

**t-SNE**

van der Maaten, L. (2009). Learning a parametric embedding by preserving local structure. RBM, 500(500), 26.

van der Maaten, L.J.P. and Hinton, G.E. (2008). Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research, 9(Nov), 2579-2605.

Lee, J.A., Peluffo-Ordonez, D.H., and Verleysen, M. (2014). Multiscale stochastic neighbor embedding: Towards parameter-free dimensionality reduction. ESANN 2014.

**UMAP**

McInnes, L., Healy, J., and Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.

**PaCMAP**

Wang, Y., Huang, H., Ruber, C., and Liang, Y. (2021). Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization. Journal of Machine Learning Research, 22(201), 1-73.

**TriMap**

Amid, E. and Warmuth, M.K. (2019). TriMap: Large-scale Dimensionality Reduction Using Triplets. arXiv:1910.00204.

**CEBRA**

Schneider, S., Lee, J.H., and Mathis, M.W. (2023). Learnable latent embeddings for joint behavioural and neural analysis. Nature, 617, 360-368.

