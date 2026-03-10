#!/usr/bin/python
"""Example: parametric DR on hematopoietic differentiation (Paul et al. 2015).

Uses scanpy's built-in Paul15 dataset: 2,730 cells with 3,451 gene
expression measurements from mouse bone marrow. Cells differentiate from
myeloid progenitors into erythrocytes, monocytes, neutrophils, and other
lineages; this trajectory structure makes it ideal for comparing static
vs temporal DR methods.

Requires: scanpy (pip install scanpy)

Usage:
    python example_hematopoiesis.py
"""

import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages

try:
    import scanpy as sc
except ImportError:
    print(
        "This example requires scanpy. Install it with:\n"
        "  pip install scanpy\n"
    )
    sys.exit(1)

from parametric_dr import (
    Parametric_tSNE,
    Parametric_PaCMAP,
    Parametric_UMAP,
    Parametric_TriMap,
    Parametric_CEBRA,
    TemporalMixin,
    compute_metrics,
)

plt.style.use("ggplot")


# ---------------------------------------------------------------------------
# Temporal mixin classes (module-level for pickling)
# ---------------------------------------------------------------------------


class Temporal_tSNE(TemporalMixin, Parametric_tSNE):
    pass


class Temporal_UMAP(TemporalMixin, Parametric_UMAP):
    pass


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------


def _load_paul15():
    """Load and preprocess the Paul15 hematopoiesis dataset.

    Preprocessing follows standard scRNA-seq practice:
    1. Filter to top highly variable genes (for speed)
    2. Log-normalize
    3. Compute diffusion pseudotime for temporal methods

    Returns
    -------
    X : array (n_cells, n_genes)
        Log-normalized expression matrix (top HVGs).
    cell_types : array of str (n_cells,)
        Cell type labels.
    cell_type_codes : array of int (n_cells,)
        Integer-encoded cell type labels.
    pseudotime : array (n_cells,)
        Diffusion pseudotime (integer-valued for temporal methods).
    lineage_labels : array of str (n_cells,)
        Coarse lineage labels (Ery, Mono, Neu, etc.).
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.datasets.paul15()

    # Standard preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Select top highly variable genes, then PCA to 50 dims
    # (standard scRNA pipeline; keeps pairwise distance matrices tractable)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50)

    # Compute diffusion pseudotime for temporal ordering
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
    sc.tl.diffmap(adata)

    # Pick root cell: use the earliest myeloid progenitor cluster (7MEP)
    mep_mask = adata.obs["paul15_clusters"] == "7MEP"
    if mep_mask.any():
        mep_indices = np.where(mep_mask)[0]
        # Pick the MEP cell closest to the center of the MEP cluster in diffmap
        mep_diffmap = adata.obsm["X_diffmap"][mep_indices]
        centroid = mep_diffmap.mean(axis=0)
        dists = np.sum((mep_diffmap - centroid) ** 2, axis=1)
        root_idx = mep_indices[np.argmin(dists)]
    else:
        root_idx = 0

    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata)

    X = np.asarray(adata.obsm["X_pca"], dtype=np.float32)

    cell_types = adata.obs["paul15_clusters"].values.astype(str)

    # Create coarse lineage labels from the cluster names
    lineage_map = {}
    for ct in np.unique(cell_types):
        # Cluster names are like "1Ery", "14Mo", "17Neu", etc.
        # Extract the suffix
        suffix = "".join(c for c in ct if c.isalpha())
        lineage_map[ct] = suffix
    lineage_labels = np.array([lineage_map[ct] for ct in cell_types])

    # Integer-encode cell types
    unique_types = sorted(set(cell_types))
    type_to_int = {t: i for i, t in enumerate(unique_types)}
    cell_type_codes = np.array([type_to_int[t] for t in cell_types])

    # Convert pseudotime to integer indices for TemporalMixin
    # (needs consecutive integers for temporal pair construction)
    dpt = adata.obs["dpt_pseudotime"].values.copy()
    dpt[np.isinf(dpt)] = np.nanmax(dpt[~np.isinf(dpt)])
    pseudotime_order = np.argsort(dpt)
    pseudotime = np.empty(len(dpt), dtype=int)
    pseudotime[pseudotime_order] = np.arange(len(dpt))

    return X, cell_types, cell_type_codes, pseudotime, lineage_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basic_configure_logging():
    logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z"
    )


def _make_encoder(num_inputs, num_outputs):
    """Encoder for scRNA expression data."""
    return nn.Sequential(
        nn.Linear(num_inputs, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, num_outputs),
    )


def _trajectory_smoothness(emb, pseudotime):
    """Mean squared distance between pseudotime-adjacent embedding points."""
    order = np.argsort(pseudotime)
    emb_ordered = emb[order]
    diffs = np.diff(emb_ordered, axis=0)
    return float(np.mean(np.sum(diffs ** 2, axis=1)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    _basic_configure_logging()

    print("Loading Paul15 hematopoiesis dataset...")
    X, cell_types, cell_type_codes, pseudotime, lineage_labels = _load_paul15()
    print(f"  {X.shape[0]} cells, {X.shape[1]} PCA components")
    print(f"  {len(np.unique(cell_types))} cell type clusters")
    print(f"  Coarse lineages: {sorted(set(lineage_labels))}")

    num_inputs = X.shape[1]
    num_outputs = 2
    alpha_ = num_outputs - 1.0

    epochs = 50
    batch_size = 128
    seed = 54321
    model_path_template = "example_hematopoiesis_{tag}.pt"
    pdf_path = "example_hematopoiesis.pdf"
    override = False

    # Color palettes
    unique_lineages = sorted(set(lineage_labels))
    lineage_palette = dict(zip(unique_lineages, sns.color_palette("Set2", len(unique_lineages))))
    lineage_colors = np.array([lineage_palette[l] for l in lineage_labels])

    unique_types = sorted(set(cell_types))
    type_palette = dict(zip(unique_types, sns.color_palette("tab20", len(unique_types))))

    # ------------------------------------------------------------------
    # Define methods
    # ------------------------------------------------------------------
    transformer_list = [
        # --- Static methods ---
        {
            "label": "t-SNE",
            "tag": "tSNE",
            "temporal": False,
            "model": Parametric_tSNE(
                num_inputs, num_outputs,
                perplexities=30, alpha=alpha_,
                batch_size=batch_size, seed=seed,
                encoder=_make_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
        },
        {
            "label": "PaCMAP",
            "tag": "PaCMAP",
            "temporal": False,
            "model": Parametric_PaCMAP(
                num_inputs, num_outputs,
                n_neighbors=10, batch_size=batch_size, seed=seed,
                encoder=_make_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
        },
        {
            "label": "UMAP",
            "tag": "UMAP",
            "temporal": False,
            "model": Parametric_UMAP(
                num_inputs, num_outputs,
                n_neighbors=15, min_dist=0.1,
                batch_size=batch_size, seed=seed,
                encoder=_make_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
        },
        {
            "label": "TriMap",
            "tag": "TriMap",
            "temporal": False,
            "model": Parametric_TriMap(
                num_inputs, num_outputs,
                n_neighbors=10, n_outliers=5, n_random=3,
                batch_size=batch_size, seed=seed,
                encoder=_make_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
        },
        # --- Temporal methods (using diffusion pseudotime) ---
        {
            "label": "CEBRA",
            "tag": "CEBRA",
            "temporal": True,
            "model": Parametric_CEBRA(
                num_inputs, num_outputs,
                n_neighbors_time=10, temperature=0.5,
                negative_sample_rate=10,
                batch_size=batch_size, seed=seed,
                encoder=_make_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {
                "epochs": epochs,
                "time_indices": pseudotime,
            },
        },
        {
            "label": "Temporal t-SNE",
            "tag": "Temporal_tSNE",
            "temporal": True,
            "model": Temporal_tSNE(
                num_inputs, num_outputs,
                perplexities=30, alpha=alpha_,
                temporal_weight=0.3,
                batch_size=batch_size, seed=seed,
                encoder=_make_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {
                "epochs": epochs,
                "time_indices": pseudotime,
            },
        },
        {
            "label": "Temporal UMAP",
            "tag": "Temporal_UMAP",
            "temporal": True,
            "model": Temporal_UMAP(
                num_inputs, num_outputs,
                n_neighbors=15, min_dist=0.1,
                temporal_weight=0.3,
                batch_size=batch_size, seed=seed,
                encoder=_make_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {
                "epochs": epochs,
                "time_indices": pseudotime,
            },
        },
    ]

    # ------------------------------------------------------------------
    # Train (or load) each method
    # ------------------------------------------------------------------
    for entry in transformer_list:
        model = entry["model"]
        model_path = model_path_template.format(tag=entry["tag"])

        if override or not os.path.exists(model_path):
            print(f"\n{'='*60}")
            print(f"Training: {entry['label']}")
            print(f"{'='*60}")
            t0 = time.perf_counter()
            model.fit(X, **entry["fit_kwargs"])
            entry["fit_time"] = time.perf_counter() - t0
            print(f"  Fit time: {entry['fit_time']:.2f}s")
            model.save_model(model_path)
        else:
            print(f"Loading {entry['label']} from {model_path}")
            model.restore_model(model_path)
            entry["fit_time"] = None

    # ------------------------------------------------------------------
    # Compute quality metrics
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("Quality Metrics")
    print(f"{'='*80}")
    print(
        f"{'Method':<20s}  {'Trust':>7s}  {'Cont':>7s}  {'NbrPres':>7s}  "
        f"{'Shepard':>7s}  {'TrajSmooth':>10s}  {'Time(s)':>8s}"
    )
    print("-" * 80)

    metrics_k = 10
    metrics_results = {}
    for entry in transformer_list:
        model = entry["model"]
        label = entry["label"]
        emb = model.transform(X)

        m = compute_metrics(X, emb, k=metrics_k)
        t, c, n, s = m["trustworthiness"], m["continuity"], m["neighborhood_preservation"], m["shepard_correlation"]
        ts = _trajectory_smoothness(emb, pseudotime)
        fit_time = entry.get("fit_time")

        metrics_results[label] = {"T": t, "C": c, "N": n, "S": s, "TS": ts, "time": fit_time}
        time_str = f"{fit_time:8.2f}" if fit_time is not None else "   (load)"
        print(f"{label:<20s}  {t:7.4f}  {c:7.4f}  {n:7.4f}  {s:7.4f}  {ts:10.4f}  {time_str}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    pdf_obj = PdfPages(pdf_path)

    # Per-method pages: 1x3 layout (cell type, lineage, pseudotime)
    for entry in transformer_list:
        model = entry["model"]
        label = entry["label"]
        is_temporal = entry["temporal"]
        m = metrics_results.get(label, {})
        emb = model.transform(X)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # Left: colored by cell type cluster
        for ct in unique_types:
            mask = cell_types == ct
            axes[0].scatter(
                emb[mask, 0], emb[mask, 1],
                c=[type_palette[ct]], s=5, alpha=0.5,
                label=ct, edgecolors="none",
            )
        axes[0].set_title("Cell type clusters")
        axes[0].legend(
            fontsize=5, ncol=2, loc="best", markerscale=2,
            handletextpad=0.3, columnspacing=0.5,
        )

        # Center: colored by coarse lineage
        for lin in unique_lineages:
            mask = lineage_labels == lin
            axes[1].scatter(
                emb[mask, 0], emb[mask, 1],
                c=[lineage_palette[lin]], s=5, alpha=0.5,
                label=lin, edgecolors="none",
            )
        axes[1].set_title("Lineage")
        axes[1].legend(fontsize=8, loc="best", markerscale=3)

        # Right: colored by pseudotime
        sc_pt = axes[2].scatter(
            emb[:, 0], emb[:, 1], c=pseudotime, cmap="viridis",
            s=5, alpha=0.5, edgecolors="none",
        )
        axes[2].set_title("Pseudotime")
        plt.colorbar(sc_pt, ax=axes[2], label="Diffusion pseudotime rank")

        if m:
            time_part = f"  Fit={m['time']:.1f}s" if m.get("time") is not None else ""
            metric_text = (
                f"Trust={m['T']:.3f}  Cont={m['C']:.3f}  "
                f"NbrPres={m['N']:.3f}  Shepard={m['S']:.3f}  "
                f"TrajSmooth={m['TS']:.3f}{time_part}"
            )
            fig.text(0.5, 0.01, metric_text, ha="center", fontsize=9, fontstyle="italic")

        tag = "[temporal]" if is_temporal else "[static]"
        fig.suptitle(
            f"{label} {tag} -- Paul15 hematopoiesis (2,730 cells)",
            fontsize=13,
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        plt.savefig(pdf_obj, format="pdf")
        plt.close(fig)

    # Summary page: metrics bar chart
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    method_names = list(metrics_results.keys())
    x_pos = np.arange(len(method_names))
    bar_colors = sns.color_palette("Set2", len(method_names))

    metric_specs = [
        ("T", "Trustworthiness"), ("C", "Continuity"),
        ("N", "Neighborhood Preservation"), ("S", "Shepard Correlation"),
        ("TS", "Trajectory Smoothness (lower = better)"),
        ("time", "Fit Time (seconds)"),
    ]
    for idx, (metric_key, metric_label) in enumerate(metric_specs):
        ax = axes.flat[idx]
        vals = [metrics_results[m][metric_key] or 0 for m in method_names]
        ax.barh(x_pos, vals, color=bar_colors)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(method_names, fontsize=8)
        ax.set_title(metric_label)
        if metric_key not in ("TS", "time"):
            ax.set_xlim(0, 1.05)
        ax.invert_yaxis()

    fig.suptitle(f"Quality Metrics Comparison (k={metrics_k})", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(pdf_obj, format="pdf")
    plt.close(fig)

    pdf_obj.close()
    print(f"\nSaved plots to {pdf_path}")


if __name__ == "__main__":
    main()
