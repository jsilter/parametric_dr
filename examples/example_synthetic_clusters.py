#!/usr/bin/python
"""Example: compare all parametric DR methods on synthetic data.

Generates clustered data in high (14) dimensions, trains each method,
transforms held-out test data through the trained models, and produces
a multi-page PDF with scatter + KDE plots and quality metrics.

Usage:
    python example_viz_parametric_dr.py [hollow|dense]
"""

import logging
import os
import sys
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages

from parametric_dr import (
    Parametric_tSNE,
    Parametric_PaCMAP,
    Parametric_UMAP,
    Parametric_TriMap,
    Parametric_CEBRA,
    TemporalMixin,
    compute_metrics,
)
from parametric_dr.utils import get_multiscale_perplexities

has_sklearn = False
try:
    from sklearn.decomposition import PCA

    has_sklearn = True
except Exception as ex:
    print("Error trying to import sklearn, will not plot PCA")
    print(ex)

plt.style.use("ggplot")


# ---------------------------------------------------------------------------
# Temporal mixin composition (defined at module level for pickling)
# ---------------------------------------------------------------------------


class Temporal_tSNE(TemporalMixin, Parametric_tSNE):
    pass


class Temporal_PaCMAP(TemporalMixin, Parametric_PaCMAP):
    pass


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def basic_configure_logging():
    logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z"
    )


def _gen_cluster_centers(num_clusters: int, top_cluster_size: int):
    cluster_centers = np.zeros([num_clusters, num_clusters])
    cluster_centers[0:top_cluster_size, 0:top_cluster_size] = 1.0
    cluster_centers[top_cluster_size::, top_cluster_size::] = 1.0
    cluster_centers[np.diag_indices(num_clusters)] *= -1
    cluster_centers *= top_cluster_size
    return cluster_centers


def _gen_hollow_spheres(num_clusters: int, num_samps: int, num_rand_points: int = 0):
    top_cluster_size = min([5, num_samps])
    cluster_centers = _gen_cluster_centers(num_clusters, top_cluster_size)
    cluster_assignments = np.arange(0, num_samps) % num_clusters

    per_samp_centers = cluster_centers[cluster_assignments, :]

    radii = 0.5 * np.ones([num_clusters])
    radii[top_cluster_size::] = 1.5

    cluster_radii = radii[cluster_assignments]
    cluster_radii += np.random.normal(loc=0.0, scale=0.05, size=num_samps)

    for xx in range(num_rand_points):
        rand_ind = np.random.randint(len(cluster_radii))
        cluster_radii[rand_ind] = np.random.uniform(low=0.05, high=10.0)
        per_samp_centers[rand_ind, :] += np.random.normal(
            loc=0.0, scale=10.0, size=cluster_centers.shape[1]
        )

    init_points = np.random.normal(loc=0.0, scale=1.0, size=[num_samps, num_clusters])
    min_rad = 1e-3
    init_radii = np.linalg.norm(init_points, axis=1)
    bad_points = np.where(init_radii < min_rad)[0]
    num_bad_points = len(bad_points)
    while num_bad_points >= 1:
        init_points[bad_points, :] = np.random.normal(
            loc=0.0, scale=1.0, size=[num_bad_points, num_clusters]
        )
        init_radii = np.linalg.norm(init_points, axis=1)
        bad_points = np.where(init_radii < min_rad)[0]
        num_bad_points = len(bad_points)

    init_points = init_points / init_radii[:, np.newaxis]
    final_points = init_points * cluster_radii[:, np.newaxis]
    final_points += per_samp_centers

    return final_points, cluster_assignments


def _gen_dense_spheres(num_clusters: int, num_samps: int, num_rand_points: int = 0):
    """Generate num_clusters sets of dense spheres of points."""
    top_cluster_size = min([5, num_samps])
    cluster_centers = _gen_cluster_centers(num_clusters, top_cluster_size)

    pick_rows = np.arange(0, num_samps) % num_clusters
    scales = 1.0 + 2 * (np.array(pick_rows, dtype=float) / num_clusters)

    test_data = cluster_centers[pick_rows, :]

    for xx in range(num_rand_points):
        rand_ind = np.random.randint(len(scales))
        scales[rand_ind] = 10.0

    for xx in range(num_samps):
        test_data[xx, :] += np.random.normal(
            loc=0.0, scale=scales[xx], size=num_clusters
        )

    return test_data, pick_rows


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_scatter(
    output_res: np.ndarray,
    pick_rows,
    color_palette,
    alpha: float = 0.5,
    symbol: str = "o",
):
    num_clusters = len(set(pick_rows))
    for ci in range(num_clusters):
        cur_plot_rows = pick_rows == ci
        cur_color = color_palette[ci]
        plt.plot(
            output_res[cur_plot_rows, 0],
            output_res[cur_plot_rows, 1],
            symbol,
            color=cur_color,
            label=ci,
            alpha=alpha,
        )


def _plot_kde(output_res: np.ndarray, pick_rows, color_palette, alpha: float = 0.5):
    num_clusters = len(set(pick_rows))
    for ci in range(num_clusters):
        cur_plot_rows = pick_rows == ci
        cur_cmap = sns.light_palette(color_palette[ci], as_cmap=True)
        sns.kdeplot(
            x=output_res[cur_plot_rows, 0],
            y=output_res[cur_plot_rows, 1],
            cmap=cur_cmap,
            fill=True,
            alpha=alpha,
            thresh=0.05,
            warn_singular=False,
        )
        centroid = output_res[cur_plot_rows, :].mean(axis=0)
        plt.annotate(
            "%s" % ci,
            xy=centroid,
            xycoords="data",
            alpha=0.5,
            horizontalalignment="center",
            verticalalignment="center",
        )


def _make_small_encoder(num_inputs, num_outputs):
    """Smaller encoder for faster example runs."""
    return nn.Sequential(
        nn.Linear(num_inputs, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, num_outputs),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    basic_configure_logging()

    num_clusters = 14
    model_path_template = "example_viz_{model_tag}_{test_data_tag}.pt"
    figure_template = "example_viz_all_methods_{test_data_tag}.pdf"
    override = False

    num_samps = 1000
    epochs = 20
    batch_size = 128
    plot_pca = has_sklearn
    color_palette = sns.color_palette("hls", num_clusters)

    test_data_tag = "hollow"
    if len(sys.argv) >= 2:
        test_data_tag = sys.argv[1]

    debug = False
    if debug:
        model_path_template = "example_viz_debug_{model_tag}_{test_data_tag}.pt"
        figure_template = "example_viz_debug_all_methods_{test_data_tag}.pdf"
        num_samps = 400
        epochs = 5
        plot_pca = False
        override = True

    num_rand_points = int(num_samps / num_clusters)
    num_outputs = 2
    alpha_ = num_outputs - 1.0

    if test_data_tag == "dense":
        _gen_test_data = _gen_dense_spheres
    elif test_data_tag == "hollow":
        _gen_test_data = _gen_hollow_spheres
    else:
        raise ValueError(f"Unknown test data tag {test_data_tag}")

    # ------------------------------------------------------------------
    # Generate data
    # ------------------------------------------------------------------
    np.random.seed(12345)
    train_data, train_cluster_assignments = _gen_test_data(
        num_clusters, num_samps, num_rand_points
    )
    np.random.seed(86131894)
    test_data, test_cluster_assignments = _gen_test_data(
        num_clusters, num_samps, num_rand_points
    )

    num_inputs = train_data.shape[1]

    # Synthetic time indices for temporal methods (sequential ordering)
    time_indices = np.arange(num_samps)

    # ------------------------------------------------------------------
    # Define all methods to compare
    # ------------------------------------------------------------------
    transformer_list = [
        # --- t-SNE variants ---
        {
            "label": "Multiscale t-SNE",
            "tag": "tSNE_multiscale",
            "model": Parametric_tSNE(
                num_inputs, num_outputs,
                perplexities=get_multiscale_perplexities(2 * num_samps),
                alpha=alpha_, batch_size=batch_size, seed=54321,
                encoder=_make_small_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
        },
        {
            "label": "t-SNE (perplexity=30)",
            "tag": "tSNE_perp30",
            "model": Parametric_tSNE(
                num_inputs, num_outputs,
                perplexities=30, alpha=alpha_,
                batch_size=batch_size, seed=54321,
                encoder=_make_small_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
        },
        # --- PaCMAP ---
        {
            "label": "PaCMAP (k=10)",
            "tag": "PaCMAP_k10",
            "model": Parametric_PaCMAP(
                num_inputs, num_outputs,
                n_neighbors=10, batch_size=batch_size, seed=54321,
                encoder=_make_small_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
        },
        {
            "label": "PaCMAP (k=30)",
            "tag": "PaCMAP_k30",
            "model": Parametric_PaCMAP(
                num_inputs, num_outputs,
                n_neighbors=30, batch_size=batch_size, seed=54321,
                encoder=_make_small_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
        },
        # --- UMAP ---
        {
            "label": "UMAP (k=15, min_dist=0.1)",
            "tag": "UMAP_k15_md01",
            "model": Parametric_UMAP(
                num_inputs, num_outputs,
                n_neighbors=15, min_dist=0.1,
                batch_size=batch_size, seed=54321,
                encoder=_make_small_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
        },
        {
            "label": "UMAP (k=15, min_dist=0.5)",
            "tag": "UMAP_k15_md05",
            "model": Parametric_UMAP(
                num_inputs, num_outputs,
                n_neighbors=15, min_dist=0.5,
                batch_size=batch_size, seed=54321,
                encoder=_make_small_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
        },
        # --- TriMap ---
        {
            "label": "TriMap (k=10)",
            "tag": "TriMap_k10",
            "model": Parametric_TriMap(
                num_inputs, num_outputs,
                n_neighbors=10, n_outliers=5, n_random=3,
                batch_size=batch_size, seed=54321,
                encoder=_make_small_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
        },
        # --- CEBRA ---
        {
            "label": "CEBRA (temporal)",
            "tag": "CEBRA_temporal",
            "model": Parametric_CEBRA(
                num_inputs, num_outputs,
                n_neighbors_time=10, temperature=0.5,
                negative_sample_rate=10,
                batch_size=batch_size, seed=54321,
                encoder=_make_small_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {
                "epochs": epochs,
                "time_indices": time_indices,
            },
        },
        # --- Temporal t-SNE (mixin) ---
        {
            "label": "Temporal t-SNE",
            "tag": "Temporal_tSNE",
            "model": Temporal_tSNE(
                num_inputs, num_outputs,
                perplexities=30, alpha=alpha_,
                temporal_weight=0.5,
                batch_size=batch_size, seed=54321,
                encoder=_make_small_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {
                "epochs": epochs,
                "time_indices": time_indices,
            },
        },
        # --- Temporal PaCMAP (mixin) ---
        {
            "label": "Temporal PaCMAP",
            "tag": "Temporal_PaCMAP",
            "model": Temporal_PaCMAP(
                num_inputs, num_outputs,
                n_neighbors=10,
                temporal_weight=0.5,
                batch_size=batch_size, seed=54321,
                encoder=_make_small_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {
                "epochs": epochs,
                "time_indices": time_indices,
            },
        },
    ]

    # ------------------------------------------------------------------
    # Train (or load) each method
    # ------------------------------------------------------------------
    for entry in transformer_list:
        model = entry["model"]
        tag = entry["tag"]
        model_path = model_path_template.format(
            model_tag=tag, test_data_tag=test_data_tag
        )

        if override or not os.path.exists(model_path):
            print(f"\n{'='*60}")
            print(f"Training: {entry['label']}")
            print(f"{'='*60}")
            t0 = time.perf_counter()
            model.fit(train_data, **entry["fit_kwargs"])
            entry["fit_time"] = time.perf_counter() - t0
            print(f"  Fit time: {entry['fit_time']:.2f}s")
            model.save_model(model_path)
        else:
            print(f"Loading {entry['label']} from {model_path}")
            model.restore_model(model_path)
            entry["fit_time"] = None

    # ------------------------------------------------------------------
    # Add PCA baseline
    # ------------------------------------------------------------------
    if plot_pca:
        pca = PCA(n_components=2)
        t0 = time.perf_counter()
        pca.fit(train_data)
        pca_time = time.perf_counter() - t0
        transformer_list.append(
            {
                "label": "PCA",
                "tag": "PCA",
                "model": pca,
                "fit_kwargs": {},
                "fit_time": pca_time,
            }
        )

    # ------------------------------------------------------------------
    # Compute quality metrics for each method
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Quality Metrics (computed on training data)")
    print(f"{'='*70}")
    print(
        f"{'Method':<25s}  {'Trust':>7s}  {'Cont':>7s}  {'NbrPres':>7s}  {'Shepard':>7s}  {'Time(s)':>8s}"
    )
    print("-" * 70)

    metrics_k = 10
    metrics_results = {}
    for entry in transformer_list:
        model = entry["model"]
        label = entry["label"]
        emb = model.transform(train_data)

        m = compute_metrics(train_data, emb, k=metrics_k)
        t, c, n, s = m["trustworthiness"], m["continuity"], m["neighborhood_preservation"], m["shepard_correlation"]
        fit_time = entry.get("fit_time")

        metrics_results[label] = {"T": t, "C": c, "N": n, "S": s, "time": fit_time}
        time_str = f"{fit_time:8.2f}" if fit_time is not None else "   (load)"
        print(f"{label:<25s}  {t:7.4f}  {c:7.4f}  {n:7.4f}  {s:7.4f}  {time_str}")

    # ------------------------------------------------------------------
    # Plot all methods
    # ------------------------------------------------------------------
    pdf_path = figure_template.format(test_data_tag=test_data_tag)
    pdf_obj = PdfPages(pdf_path)

    for entry in transformer_list:
        model = entry["model"]
        label = entry["label"]

        train_emb = model.transform(train_data)
        test_emb = model.transform(test_data)

        m = metrics_results.get(label, {})

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left panel: training data (KDE + scatter)
        plt.sca(axes[0])
        _plot_kde(train_emb, train_cluster_assignments, color_palette, 0.5)
        _plot_scatter(
            train_emb, train_cluster_assignments, color_palette, alpha=0.15, symbol="."
        )
        axes[0].set_title(f"{label} - Train")

        # Right panel: test data (scatter only)
        plt.sca(axes[1])
        _plot_scatter(
            test_emb, test_cluster_assignments, color_palette, alpha=0.4, symbol="o"
        )
        axes[1].set_title(f"{label} - Test (out-of-sample)")

        # Add metrics annotation
        if m:
            time_part = f"  Fit={m['time']:.1f}s" if m.get("time") is not None else ""
            metric_text = (
                f"Trust={m['T']:.3f}  Cont={m['C']:.3f}\n"
                f"NbrPres={m['N']:.3f}  Shepard={m['S']:.3f}{time_part}"
            )
            fig.text(
                0.5, 0.01, metric_text,
                ha="center", fontsize=9, fontstyle="italic",
            )

        fig.suptitle(
            f"{label} -- {num_clusters} clusters, {test_data_tag} data",
            fontsize=13,
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        plt.savefig(pdf_obj, format="pdf")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Summary page: metrics bar chart
    # ------------------------------------------------------------------
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
        ax.set_yticklabels(method_names, fontsize=8)
        ax.set_title(metric_label)
        if metric_key not in ("time",):
            ax.set_xlim(0, 1.05)
        ax.invert_yaxis()

    axes.flat[-1].set_visible(False)

    fig.suptitle(
        f"Quality Metrics Comparison (k={metrics_k}, {test_data_tag} data)",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(pdf_obj, format="pdf")
    plt.close(fig)

    pdf_obj.close()
    print(f"\nSaved plots to {pdf_path}")


if __name__ == "__main__":
    main()
