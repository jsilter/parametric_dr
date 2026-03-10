#!/usr/bin/python
"""Example: Lorenz attractor projected to high dimensions.

Integrates the classic Lorenz system (chaotic butterfly attractor),
projects the 3D trajectory into 20 dimensions via a random linear
map plus noise, then compares static vs temporal DR methods on
recovering the butterfly structure.

The butterfly shape is visually recognizable, so users can immediately
see which methods recover it. Time ordering is essential to the
structure, making this a strong case for temporal methods.

Usage:
    python example_lorenz.py
"""

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from scipy.integrate import solve_ivp

from parametric_dr import (
    Parametric_tSNE,
    Parametric_PaCMAP,
    Parametric_UMAP,
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
# Data generation: Lorenz attractor in high dimensions
# ---------------------------------------------------------------------------


def _lorenz_rhs(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def _generate_lorenz_data(n_steps=3000, obs_dim=20, noise_std=0.5, seed=42):
    """Integrate the Lorenz system and project into high dimensions.

    Parameters
    ----------
    n_steps : int
        Number of time steps to sample from the trajectory.
    obs_dim : int
        Dimensionality of the observed (projected) data.
    noise_std : float
        Standard deviation of Gaussian noise added after projection.
    seed : int

    Returns
    -------
    X_obs : array (n_steps, obs_dim)
        High-dimensional observations.
    X_3d : array (n_steps, 3)
        Ground truth 3D Lorenz trajectory.
    time_indices : array (n_steps,)
        Sequential time indices.
    """
    rng = np.random.RandomState(seed)

    # Integrate
    t_span = (0, 50)
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    sol = solve_ivp(
        _lorenz_rhs, t_span, y0=[1.0, 1.0, 1.0],
        t_eval=t_eval, method="RK45", max_step=0.01,
    )
    X_3d = sol.y.T.astype(np.float32)  # (n_steps, 3)

    # Random linear projection to high-D + noise
    W = rng.randn(3, obs_dim).astype(np.float32)
    noise = noise_std * rng.randn(n_steps, obs_dim).astype(np.float32)
    X_obs = X_3d @ W + noise

    time_indices = np.arange(n_steps)
    return X_obs, X_3d, time_indices


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


def _trajectory_smoothness(emb):
    """Mean squared distance between consecutive embedding points."""
    diffs = np.diff(emb, axis=0)
    return float(np.mean(np.sum(diffs ** 2, axis=1)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    _basic_configure_logging()

    # Generate data
    X_obs, X_3d, time_indices = _generate_lorenz_data()
    z_coord = X_3d[:, 2]  # z-coordinate distinguishes the two butterfly wings
    num_inputs = X_obs.shape[1]  # 20
    num_outputs = 2
    alpha_ = num_outputs - 1.0

    epochs = 50
    batch_size = 128
    seed = 54321
    model_path_template = "example_lorenz_{tag}.pt"
    pdf_path = "example_lorenz.pdf"
    override = False

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
            model.fit(X_obs, **entry["fit_kwargs"])
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
        emb = model.transform(X_obs)

        m = compute_metrics(X_obs, emb, k=metrics_k)
        t, c, n, s = m["trustworthiness"], m["continuity"], m["neighborhood_preservation"], m["shepard_correlation"]
        ts = _trajectory_smoothness(emb)
        fit_time = entry.get("fit_time")

        metrics_results[label] = {"T": t, "C": c, "N": n, "S": s, "TS": ts, "time": fit_time}
        time_str = f"{fit_time:8.2f}" if fit_time is not None else "   (load)"
        print(f"{label:<20s}  {t:7.4f}  {c:7.4f}  {n:7.4f}  {s:7.4f}  {ts:10.4f}  {time_str}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    pdf_obj = PdfPages(pdf_path)

    # Ground truth page: 3D Lorenz attractor
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
        c=time_indices, cmap="viridis", s=2, alpha=0.5,
    )
    ax.plot(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], "-", color="gray", alpha=0.1, linewidth=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(sc, ax=ax, label="Time step", shrink=0.6)
    fig.suptitle("Ground truth: Lorenz attractor (3D)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(pdf_obj, format="pdf")
    plt.close(fig)

    # Per-method pages
    for entry in transformer_list:
        model = entry["model"]
        label = entry["label"]
        is_temporal = entry["temporal"]
        m = metrics_results.get(label, {})
        emb = model.transform(X_obs)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: colored by time
        sc0 = axes[0].scatter(
            emb[:, 0], emb[:, 1], c=time_indices, cmap="viridis",
            s=6, alpha=0.6, edgecolors="none",
        )
        if is_temporal:
            axes[0].plot(emb[:, 0], emb[:, 1], "-", color="gray", alpha=0.1, linewidth=0.3)
        axes[0].set_title(f"{label} (color = time)")
        plt.colorbar(sc0, ax=axes[0], label="Time step")

        # Right: colored by z-coordinate (distinguishes butterfly wings)
        sc1 = axes[1].scatter(
            emb[:, 0], emb[:, 1], c=z_coord, cmap="coolwarm",
            s=6, alpha=0.6, edgecolors="none",
        )
        if is_temporal:
            axes[1].plot(emb[:, 0], emb[:, 1], "-", color="gray", alpha=0.1, linewidth=0.3)
        axes[1].set_title(f"{label} (color = Lorenz z)")
        plt.colorbar(sc1, ax=axes[1], label="z coordinate")

        if m:
            time_part = f"  Fit={m['time']:.1f}s" if m.get("time") is not None else ""
            metric_text = (
                f"Trust={m['T']:.3f}  Cont={m['C']:.3f}  "
                f"NbrPres={m['N']:.3f}  Shepard={m['S']:.3f}  "
                f"TrajSmooth={m['TS']:.3f}{time_part}"
            )
            fig.text(0.5, 0.01, metric_text, ha="center", fontsize=9, fontstyle="italic")

        tag = "[temporal]" if is_temporal else "[static]"
        fig.suptitle(f"{label} {tag} -- Lorenz attractor", fontsize=13)
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
