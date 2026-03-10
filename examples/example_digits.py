#!/usr/bin/python
"""Example: compare parametric DR methods on the sklearn digits dataset.

Uses 8x8 pixel images of handwritten digits (64 features, 10 classes,
1797 samples). Compares t-SNE, PaCMAP, UMAP, TriMap, and PCA on how
well each separates the digit clusters.

Usage:
    python example_digits.py
"""

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.datasets import load_digits
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
    """Slightly larger encoder for 64-dim digit data with 10 classes."""
    return nn.Sequential(
        nn.Linear(num_inputs, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, num_outputs),
    )


def _plot_scatter(emb, labels, palette, alpha=0.5, symbol="o"):
    for ci in sorted(set(labels)):
        mask = labels == ci
        plt.plot(
            emb[mask, 0], emb[mask, 1], symbol,
            color=palette[ci], label=str(ci), alpha=alpha, markersize=4,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    _basic_configure_logging()

    # Load data
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target
    num_inputs = X.shape[1]  # 64
    num_outputs = 2
    num_classes = 10
    alpha_ = num_outputs - 1.0

    # Train/test split (80/20)
    np.random.seed(42)
    perm = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = perm[:split], perm[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    epochs = 50
    batch_size = 128
    seed = 54321
    model_path_template = "example_digits_{tag}.pt"
    pdf_path = "example_digits.pdf"
    override = False

    color_palette = sns.color_palette("tab10", num_classes)

    # ------------------------------------------------------------------
    # Define methods
    # ------------------------------------------------------------------
    transformer_list = [
        {
            "label": "t-SNE (perp=30)",
            "tag": "tSNE_perp30",
            "model": Parametric_tSNE(
                num_inputs, num_outputs,
                perplexities=30, alpha=alpha_,
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
                n_neighbors=15, min_dist=0.1,
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
            model.fit(X_train, **entry["fit_kwargs"])
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
    pca.fit(X_train)
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
    print("Quality Metrics (computed on training data)")
    print(f"{'='*66}")
    print(
        f"{'Method':<20s}  {'Trust':>7s}  {'Cont':>7s}  {'NbrPres':>7s}  {'Shepard':>7s}  {'Time(s)':>8s}"
    )
    print("-" * 66)

    metrics_k = 10
    metrics_results = {}
    for entry in transformer_list:
        model = entry["model"]
        label = entry["label"]
        emb = model.transform(X_train)

        m = compute_metrics(X_train, emb, k=metrics_k)
        t, c, n, s = m["trustworthiness"], m["continuity"], m["neighborhood_preservation"], m["shepard_correlation"]
        fit_time = entry.get("fit_time")

        metrics_results[label] = {"T": t, "C": c, "N": n, "S": s, "time": fit_time}
        time_str = f"{fit_time:8.2f}" if fit_time is not None else "   (load)"
        print(f"{label:<20s}  {t:7.4f}  {c:7.4f}  {n:7.4f}  {s:7.4f}  {time_str}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    pdf_obj = PdfPages(pdf_path)

    for entry in transformer_list:
        model = entry["model"]
        label = entry["label"]
        m = metrics_results.get(label, {})

        train_emb = model.transform(X_train)
        test_emb = model.transform(X_test)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: training data
        plt.sca(axes[0])
        _plot_scatter(train_emb, y_train, color_palette, alpha=0.5, symbol=".")
        axes[0].set_title(f"{label} (train)")
        axes[0].legend(fontsize=7, ncol=2, loc="best", markerscale=3)

        # Right: test data (out-of-sample)
        plt.sca(axes[1])
        _plot_scatter(test_emb, y_test, color_palette, alpha=0.5, symbol="o")
        axes[1].set_title(f"{label} (test, out-of-sample)")
        axes[1].legend(fontsize=7, ncol=2, loc="best", markerscale=2)

        if m:
            time_part = f"  Fit={m['time']:.1f}s" if m.get("time") is not None else ""
            metric_text = (
                f"Trust={m['T']:.3f}  Cont={m['C']:.3f}  "
                f"NbrPres={m['N']:.3f}  Shepard={m['S']:.3f}{time_part}"
            )
            fig.text(0.5, 0.01, metric_text, ha="center", fontsize=9, fontstyle="italic")

        fig.suptitle(f"{label} on sklearn digits (10 classes, 64 features)", fontsize=13)
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
