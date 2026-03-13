#!/usr/bin/python
"""Example: simulated place cell population with temporal structure.

An animal moves along a circular track; 50 neurons fire according to
their preferred position. The resulting (2000, 50) matrix has natural
time ordering, making it ideal for comparing static vs temporal DR methods.

Static methods (t-SNE, PaCMAP, UMAP) should recover a ring-like shape
but with a fragmented trajectory. Temporal methods (CEBRA, Temporal t-SNE,
Temporal UMAP) should produce a smooth ring with a clean temporal loop.

Usage:
    python example_neural_timecourse.py
"""

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages

from parametric_dr import (
    Parametric_tSNE,
    Parametric_PaCMAP,
    Parametric_UMAP,
    Parametric_CEBRA,
    TemporalMixin,
    compute_metrics,
    trajectory_smoothness,
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
# Data generation: simulated place cells on a circular track
# ---------------------------------------------------------------------------


def _generate_place_cell_data(
    n_timesteps=2000, n_neurons=50, sigma=0.3, baseline=0.2, noise_std=0.5, seed=42,
):
    """Simulate place cell population activity on a circular track.

    Parameters
    ----------
    n_timesteps : int
        Number of time steps.
    n_neurons : int
        Number of neurons, each with a uniformly spaced preferred position.
    sigma : float
        Tuning curve width (in radians).
    baseline : float
        Baseline firing rate.
    noise_std : float
        Gaussian noise standard deviation added to firing rates.
    seed : int

    Returns
    -------
    X : array (n_timesteps, n_neurons)
        Noisy firing rates.
    theta : array (n_timesteps,)
        Animal position on [0, 2*pi).
    time_indices : array (n_timesteps,)
        Sequential time indices (0, 1, 2, ...).
    """
    rng = np.random.RandomState(seed)

    # Animal position: random walk with drift on the circle
    d_theta = 0.05 + 0.03 * rng.randn(n_timesteps)
    theta = np.cumsum(d_theta) % (2 * np.pi)

    # Preferred positions uniformly around the circle
    phi = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)

    # Firing rates: von-Mises-like tuning (Gaussian on angular distance)
    # angular_distance shape: (n_timesteps, n_neurons)
    diff = theta[:, None] - phi[None, :]
    angular_dist = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
    rates = np.exp(-0.5 * (angular_dist / sigma) ** 2) + baseline

    # Add Gaussian noise
    X = rates + noise_std * rng.randn(n_timesteps, n_neurons)

    time_indices = np.arange(n_timesteps)
    return X.astype(np.float32), theta, time_indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basic_configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z",
    )


def _make_small_encoder(num_inputs, num_outputs):
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
    _basic_configure_logging()

    # Generate data
    X, theta, time_indices = _generate_place_cell_data()
    num_inputs = X.shape[1]  # 50
    num_outputs = 2
    alpha_ = num_outputs - 1.0

    epochs = 50
    batch_size = 128
    seed = 54321
    model_path_template = "example_neural_{tag}.pt"
    pdf_path = "example_neural_timecourse.pdf"
    override = False

    # ------------------------------------------------------------------
    # Define methods: static and temporal
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
                encoder=_make_small_encoder(num_inputs, num_outputs),
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
                encoder=_make_small_encoder(num_inputs, num_outputs),
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
                encoder=_make_small_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {"epochs": epochs},
        },
        # --- Temporal methods ---
        {
            "label": "CEBRA",
            "tag": "CEBRA",
            "temporal": True,
            "model": Parametric_CEBRA(
                num_inputs, num_outputs,
                n_neighbors_time=10, temperature=0.5,
                negative_sample_rate=10,
                batch_size=batch_size, seed=seed,
                encoder=_make_small_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {
                "epochs": epochs,
                "time_indices": time_indices,
            },
        },
        {
            "label": "Temporal t-SNE",
            "tag": "Temporal_tSNE",
            "temporal": True,
            "model": Temporal_tSNE(
                num_inputs, num_outputs,
                perplexities=30, alpha=alpha_,
                temporal_weight=0.5,
                batch_size=batch_size, seed=seed,
                encoder=_make_small_encoder(num_inputs, num_outputs),
            ),
            "fit_kwargs": {
                "epochs": epochs,
                "time_indices": time_indices,
            },
        },
        {
            "label": "Temporal UMAP",
            "tag": "Temporal_UMAP",
            "temporal": True,
            "model": Temporal_UMAP(
                num_inputs, num_outputs,
                n_neighbors=15, min_dist=0.1,
                temporal_weight=0.5,
                batch_size=batch_size, seed=seed,
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
        ts = trajectory_smoothness(emb)
        fit_time = entry.get("fit_time")

        metrics_results[label] = {"T": t, "C": c, "N": n, "S": s, "TS": ts, "time": fit_time}
        time_str = f"{fit_time:8.2f}" if fit_time is not None else "   (load)"
        print(f"{label:<20s}  {t:7.4f}  {c:7.4f}  {n:7.4f}  {s:7.4f}  {ts:10.4f}  {time_str}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    pdf_obj = PdfPages(pdf_path)

    for entry in transformer_list:
        model = entry["model"]
        label = entry["label"]
        is_temporal = entry["temporal"]
        m = metrics_results.get(label, {})
        emb = model.transform(X)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: colored by angular position (circular colormap)
        sc0 = axes[0].scatter(
            emb[:, 0], emb[:, 1], c=theta, cmap="hsv",
            s=8, alpha=0.6, edgecolors="none",
        )
        if is_temporal:
            axes[0].plot(emb[:, 0], emb[:, 1], "-", color="gray", alpha=0.1, linewidth=0.3)
        axes[0].set_title(f"{label} (color = position)")
        plt.colorbar(sc0, ax=axes[0], label="Position (rad)")

        # Right: colored by time (sequential colormap)
        sc1 = axes[1].scatter(
            emb[:, 0], emb[:, 1], c=time_indices, cmap="viridis",
            s=8, alpha=0.6, edgecolors="none",
        )
        if is_temporal:
            axes[1].plot(emb[:, 0], emb[:, 1], "-", color="gray", alpha=0.1, linewidth=0.3)
        axes[1].set_title(f"{label} (color = time)")
        plt.colorbar(sc1, ax=axes[1], label="Time step")

        if m:
            time_part = f"  Fit={m['time']:.1f}s" if m.get("time") is not None else ""
            metric_text = (
                f"Trust={m['T']:.3f}  Cont={m['C']:.3f}  "
                f"NbrPres={m['N']:.3f}  Shepard={m['S']:.3f}  "
                f"TrajSmooth={m['TS']:.3f}{time_part}"
            )
            fig.text(0.5, 0.01, metric_text, ha="center", fontsize=9, fontstyle="italic")

        tag = "[temporal]" if is_temporal else "[static]"
        fig.suptitle(f"{label} {tag} -- Place cell simulation", fontsize=13)
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        plt.savefig(pdf_obj, format="pdf")
        plt.close(fig)

    # Trajectory page: temporal methods only, connect consecutive points
    temporal_entries = [e for e in transformer_list if e["temporal"]]
    if temporal_entries:
        n_temporal = len(temporal_entries)
        fig, axes = plt.subplots(1, n_temporal, figsize=(6 * n_temporal, 5))
        if n_temporal == 1:
            axes = [axes]

        for ax, entry in zip(axes, temporal_entries):
            emb = entry["model"].transform(X)
            ax.plot(emb[:, 0], emb[:, 1], "-", color="steelblue", alpha=0.3, linewidth=0.5)
            sc = ax.scatter(
                emb[:, 0], emb[:, 1], c=theta, cmap="hsv",
                s=6, alpha=0.7, edgecolors="none", zorder=2,
            )
            ax.set_title(entry["label"])
            plt.colorbar(sc, ax=ax, label="Position (rad)")

        fig.suptitle("Temporal methods: trajectory overlay", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
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
