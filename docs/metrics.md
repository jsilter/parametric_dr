# Quality Metrics

All metrics are implemented in `parametric_dr/metrics.py` and work with any DR method (not just the parametric ones in this package). The convenience function `compute_metrics()` computes all four in a single pass, sharing distance matrices.

## Trustworthiness (T)

**Question:** Are the neighbors in the embedding actually neighbors in the original space?

Trustworthiness penalizes "false neighbors": points that appear close in the low-dimensional embedding but were far apart in the high-dimensional input. For each point, it checks whether its $k$ nearest neighbors in the embedding were also among its $k$ nearest neighbors in the original space. Points that were not neighbors are penalized proportionally to their rank in the original space (farther away = larger penalty).

**Range:** $(0, 1]$. A score of 1.0 means every neighbor in the embedding was also a neighbor in the original space.

**Formula:**

$$T(k) = 1 - \frac{2}{Nk(2N - 3k - 1)} \sum_{i=1}^{N} \sum_{j \in U_k(i)} \bigl(r(i,j) - k\bigr)$$

where $U_k(i)$ is the set of points that are among $i$'s $k$ nearest neighbors in the embedding but **not** in the original space, and $r(i,j)$ is the rank of $j$ among $i$'s neighbors in the original space.

**Reference:** Venna & Kaski (2006). Local multidimensional scaling. *Neural Networks*.

## Continuity (C)

**Question:** Are the original neighbors still neighbors in the embedding?

Continuity is the complement of trustworthiness. It penalizes "missing neighbors": points that were close in the original space but ended up far apart in the embedding. The penalty is proportional to the rank in the embedding space.

**Range:** $(0, 1]$. A score of 1.0 means every neighbor in the original space is also a neighbor in the embedding.

**Formula:**

$$C(k) = 1 - \frac{2}{Nk(2N - 3k - 1)} \sum_{i=1}^{N} \sum_{j \in V_k(i)} \bigl(\hat{r}(i,j) - k\bigr)$$

where $V_k(i)$ is the set of points that are among $i$'s $k$ nearest neighbors in the original space but **not** in the embedding, and $\hat{r}(i,j)$ is the rank of $j$ among $i$'s neighbors in the embedding.

**Interpretation together with T:**
- High $T$, high $C$: the embedding faithfully preserves local neighborhoods.
- High $T$, low $C$: the embedding doesn't invent false neighbors, but it tears apart some true neighborhoods (points that were close get separated).
- Low $T$, high $C$: the embedding keeps all true neighbors close, but also pulls in points that shouldn't be there (crowding).
- Low $T$, low $C$: the local structure is poorly preserved in both directions.

**Reference:** Venna & Kaski (2006). Local multidimensional scaling. *Neural Networks*.

## Neighborhood Preservation (N)

**Question:** What fraction of $k$-nearest neighbors are shared between the two spaces?

A simpler metric than $T$ or $C$: for each point, compute the overlap between its $k$ nearest neighbors in the original space and the embedding, then average across all points.

**Range:** $[0, 1]$. A score of 1.0 means perfect $k$-NN overlap.

**Formula:**

$$N(k) = \frac{1}{Nk} \sum_{i=1}^{N} \bigl| \text{NN}_\text{high}(i, k) \cap \text{NN}_\text{low}(i, k) \bigr|$$

## Shepard Correlation (S)

**Question:** Are pairwise distances globally preserved?

Computes the Pearson correlation between all $\binom{N}{2}$ pairwise distances in the original space and the corresponding distances in the embedding. This measures global structure preservation (unlike $T$, $C$, and $N$ which focus on local neighborhoods).

**Range:** $[-1, 1]$. A score of 1.0 means pairwise distances are perfectly linearly related. Typical good embeddings score 0.4--0.8; values near 1.0 are rare for nonlinear methods that prioritize local structure.

**Note:** Methods that prioritize local structure (t-SNE, UMAP) tend to have lower Shepard correlation than methods that preserve global structure. This is expected and not necessarily a flaw.

## Choosing $k$

The parameter $k$ (number of neighbors) controls the scale of "local" in $T$, $C$, and $N$. Typical values are 5--20. Smaller $k$ focuses on very local structure; larger $k$ captures broader neighborhood relationships. The default in this package is $k = 10$.

## Usage

```python
from parametric_dr import compute_metrics

# After fitting and transforming:
metrics = compute_metrics(X_high, embedding, k=10)
print(metrics)
# {'trustworthiness': 0.95, 'continuity': 0.97,
#  'neighborhood_preservation': 0.42, 'shepard_correlation': 0.61}
```

Individual metrics can also be called separately:

```python
from parametric_dr.metrics import trustworthiness, continuity
from parametric_dr.metrics import neighborhood_preservation, shepard_correlation

T = trustworthiness(X_high, embedding, k=10)
C = continuity(X_high, embedding, k=10)
```

All metrics accept a `metric` parameter for the high-dimensional distance (default: `"euclidean"`). Any metric supported by `scipy.spatial.distance.pdist` works (e.g., `"cosine"`, `"correlation"`, `"cityblock"`). The embedding distance is always Euclidean.
