#!/usr/bin/python
"""Example: compare parametric DR methods on Olivetti Faces.

400 grayscale face images (64x64 pixels = 4096 features) of 40 individuals
(10 images each). A high-dimensional real dataset with known ground-truth
identity labels.

Usage:
    python example_olivetti_faces.py
"""

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

from parametric_dr import (
    Parametric_tSNE,
    Parametric_PaCMAP,
    Parametric_UMAP,
    Parametric_TriMap,
    compute_metrics,
)

plt.style.use("ggplot")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basic_configure_logging():
    logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z"
    )


def _make_encoder(num_inputs, num_outputs):
    """Encoder for high-dimensional face data."""
    return nn.Sequential(
        nn.Linear(num_inputs, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, num_outputs),
    )


def _plot_scatter(emb, labels, palette, alpha=0.5, symbol="o", markersize=4):
    for ci in sorted(set(labels)):
        mask = labels == ci
        plt.plot(
            emb[mask, 0], emb[mask, 1], symbol,
            color=palette[ci % len(palette)], label=str(ci), alpha=alpha,
            markersize=markersize,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    _basic_configure_logging()

    # Load data
    faces = fetch_olivetti_faces()
    X_raw = faces.data.astype(np.float32)  # (400, 4096)
    y = faces.target  # 0-39

    # PCA to 100 dims: reduces memory for pairwise distance computations
    # (the raw 400x4096 cross-diff matrix would be ~2.5 GB)
    pca_pre = PCA(n_components=100)
    X = pca_pre.fit_transform(X_raw).astype(np.float32)

    num_inputs = X.shape[1]  # 100
    num_outputs = 2
    num_classes = len(set(y))  # 40
    alpha_ = num_outputs - 1.0

    # With only 400 samples and 40 classes (10 per class), do not split;
    # use the full dataset for training and evaluation.
    epochs = 50
    batch_size = 64
    seed = 54321
    model_path_template = "example_faces_{tag}.pt"
    pdf_path = "example_olivetti_faces.pdf"
    override = False

    # 40 classes; cycle a large qualitative palette
    color_palette = (
        sns.color_palette("tab20", 20) + sns.color_palette("tab20b", 20)
    )

    # ------------------------------------------------------------------
    # Define methods
    # ------------------------------------------------------------------
    transformer_list = [
        {
            "label": "t-SNE (perp=10)",
            "tag": "tSNE",
            "model": Parametric_tSNE(
                num_inputs, num_outputs,
                perplexities=10, alpha=alpha_,
                batch_size=batch_size, seed=seed,
                n_pca=None, encoder=_make_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
        },
        {
            "label": "PaCMAP",
            "tag": "PaCMAP",
            "model": Parametric_PaCMAP(
                num_inputs, num_outputs,
                n_neighbors=10, batch_size=batch_size, seed=seed,
                n_pca=None, encoder=_make_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
        },
        {
            "label": "UMAP",
            "tag": "UMAP",
            "model": Parametric_UMAP(
                num_inputs, num_outputs,
                n_neighbors=10, min_dist=0.1,
                batch_size=batch_size, seed=seed,
                n_pca=None, encoder=_make_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
        },
        {
            "label": "TriMap",
            "tag": "TriMap",
            "model": Parametric_TriMap(
                num_inputs, num_outputs,
                n_neighbors=10, n_outliers=5, n_random=3,
                batch_size=batch_size, seed=seed,
                n_pca=None, encoder=_make_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
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

    # Add PCA baseline
    pca = PCA(n_components=2)
    t0 = time.perf_counter()
    pca.fit(X)
    pca_time = time.perf_counter() - t0
    transformer_list.append({
        "label": "PCA",
        "tag": "PCA",
        "model": pca,
        "fit_kwargs": {},
        "fit_time": pca_time,
    })

    # ------------------------------------------------------------------
    # Compute quality metrics
    # ------------------------------------------------------------------
    print(f"\n{'='*66}")
    print("Quality Metrics")
    print(f"{'='*66}")
    print(
        f"{'Method':<20s}  {'Trust':>7s}  {'Cont':>7s}  {'NbrPres':>7s}  {'Shepard':>7s}  {'Time(s)':>8s}"
    )
    print("-" * 66)

    metrics_k = 7  # smaller k since only 10 images per person
    metrics_results = {}
    for entry in transformer_list:
        model = entry["model"]
        label = entry["label"]
        emb = model.transform(X)

        m = compute_metrics(X, emb, k=metrics_k)
        t, c, n, s = m["trustworthiness"], m["continuity"], m["neighborhood_preservation"], m["shepard_correlation"]
        fit_time = entry.get("fit_time")

        metrics_results[label] = {"T": t, "C": c, "N": n, "S": s, "time": fit_time}
        time_str = f"{fit_time:8.2f}" if fit_time is not None else "   (load)"
        print(f"{label:<20s}  {t:7.4f}  {c:7.4f}  {n:7.4f}  {s:7.4f}  {time_str}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    pdf_obj = PdfPages(pdf_path)

    # Sample face images page
    fig, axes = plt.subplots(4, 10, figsize=(14, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_raw[i].reshape(64, 64), cmap="gray")
        ax.set_title(f"id={y[i]}", fontsize=7)
        ax.axis("off")
    fig.suptitle("Olivetti Faces: sample images (40 individuals, 10 each)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(pdf_obj, format="pdf")
    plt.close(fig)

    # Per-method pages
    for entry in transformer_list:
        model = entry["model"]
        label = entry["label"]
        m = metrics_results.get(label, {})
        emb = model.transform(X)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plt.sca(ax)
        _plot_scatter(emb, y, color_palette, alpha=0.7, symbol="o", markersize=5)
        ax.set_title(f"{label}")

        if m:
            time_part = f"  Fit={m['time']:.1f}s" if m.get("time") is not None else ""
            metric_text = (
                f"Trust={m['T']:.3f}  Cont={m['C']:.3f}  "
                f"NbrPres={m['N']:.3f}  Shepard={m['S']:.3f}{time_part}"
            )
            fig.text(0.5, 0.01, metric_text, ha="center", fontsize=9, fontstyle="italic")

        fig.suptitle(
            f"{label} on Olivetti Faces (40 classes, PCA to {num_inputs}d)",
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
        ("time", "Fit Time (seconds)"),
    ]
    for idx, (metric_key, metric_label) in enumerate(metric_specs):
        ax = axes.flat[idx]
        vals = [metrics_results[m][metric_key] or 0 for m in method_names]
        ax.barh(x_pos, vals, color=bar_colors)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(method_names, fontsize=9)
        ax.set_title(metric_label)
        if metric_key not in ("time",):
            ax.set_xlim(0, 1.05)
        ax.invert_yaxis()

    axes.flat[-1].set_visible(False)

    fig.suptitle(f"Quality Metrics Comparison (k={metrics_k})", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(pdf_obj, format="pdf")
    plt.close(fig)

    pdf_obj.close()
    print(f"\nSaved plots to {pdf_path}")


if __name__ == "__main__":
    main()
