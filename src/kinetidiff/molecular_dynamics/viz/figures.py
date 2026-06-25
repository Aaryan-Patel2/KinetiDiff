"""Publication-quality figures for the KinetiDiff MD validation campaign.

All plotting functions:
- Accept DataFrames (not raw arrays) as input.
- Label axes in clear units (ns, Å, kcal/mol).
- Save to both PNG (300 dpi) and SVG (for vector editing).
- Return the Figure object so the caller can display it in the notebook.

Color scheme:
    WT     → #2196F3 (blue)
    R206H  → #E91E63 (pink/magenta)  — the FOP mutation
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

from kinetidiff._logging import get_logger

logger = get_logger(__name__)

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
})

COLORS = {
    "WT":    "#2196F3",
    "R206H": "#E91E63",
}

LIGAND_DISPLAY = {
    "L1": "L1 (Vina −11.05)",
    "L2": "L2 (Vina −10.62)",
    "L3": "L3 (Vina −10.59)",
    "L4": "L4 (Vina −10.55)",
    "L5": "L5 (Vina −10.51)",
}


def save_figure(fig: Any, name: str, fig_dir: Path) -> None:
    """Save *fig* as PNG (300 dpi) and SVG to *fig_dir*.

    Args:
        fig: Matplotlib Figure.
        name: Filename stem (no extension).
        fig_dir: Output directory.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "svg"):
        path = fig_dir / f"{name}.{fmt}"
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
    logger.info("Saved figure: %s (.png / .svg)", name)


def plot_rmsd_summary(
    analysis_results: dict[tuple, dict],
    ligand_ids: list[str],
    fig_dir: Path,
    n_replicas: int = 3,
) -> Any:
    """Plot backbone + ligand RMSD time series for all leads.

    Args:
        analysis_results: Mapping of ``(lig_id, receptor, replica_id)`` → dict
            with key ``"rmsd"`` containing a DataFrame from
            ``trajectory.run_rmsd_analysis``.
        ligand_ids: List of ligand IDs to include (order defines subplot order).
        fig_dir: Directory to save the figure.
        n_replicas: Number of replicas per condition.

    Returns:
        Matplotlib Figure.
    """
    n_lig = len(ligand_ids)
    fig, axes = plt.subplots(n_lig, 1, figsize=(10, 2.8 * n_lig), sharex=True)
    if n_lig == 1:
        axes = [axes]

    for ax, lig_id in zip(axes, ligand_ids):
        for rec_type, color in COLORS.items():
            all_bb, all_lig, all_time = [], [], None
            for rep_id in range(1, n_replicas + 1):
                data = analysis_results.get((lig_id, rec_type, rep_id), {}).get("rmsd", pd.DataFrame())
                if data.empty or "backbone_rmsd_A" not in data.columns:
                    continue
                all_bb.append(data["backbone_rmsd_A"].values)
                if "ligand_rmsd_A" in data.columns:
                    all_lig.append(data["ligand_rmsd_A"].values)
                if all_time is None:
                    all_time = data["time_ns"].values

            if not all_bb or all_time is None:
                continue

            min_len = min(len(x) for x in all_bb)
            arr = np.array([x[:min_len] for x in all_bb])
            t   = all_time[:min_len]
            mean = arr.mean(axis=0)
            sem  = arr.std(axis=0) / math.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(mean)
            ax.plot(t, mean, color=color, lw=1.5, label=f"{rec_type} Cα")
            ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.15)

            if all_lig:
                min_len_lig = min(len(x) for x in all_lig)
                arr_lig = np.array([x[:min_len_lig] for x in all_lig])
                t_lig   = all_time[:min_len_lig]
                mean_lig = arr_lig.mean(axis=0)
                ax.plot(t_lig, mean_lig, color=color, lw=1.0, ls="--", alpha=0.7, label=f"{rec_type} lig")

        ax.axhline(4.0, color="gray", lw=0.8, ls=":", alpha=0.6)  # instability threshold
        ax.set_ylabel("RMSD (Å)", fontsize=9)
        ax.set_title(LIGAND_DISPLAY.get(lig_id, lig_id), fontsize=10)
        ax.legend(fontsize=8, frameon=False, ncol=4)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time (ns)", fontsize=11)
    fig.suptitle("Backbone & Ligand RMSD — ACVR1 WT vs R206H", fontsize=12, y=1.01)
    plt.tight_layout()
    save_figure(fig, "Fig_RMSD_all", fig_dir)
    return fig


def plot_rmsf_comparison(
    analysis_results: dict[tuple, dict],
    ligand_ids: list[str],
    fig_dir: Path,
    n_replicas: int = 3,
) -> Any:
    """Per-residue RMSF overlay WT vs R206H for each lead.

    Args:
        analysis_results: Same structure as for ``plot_rmsd_summary``.
        ligand_ids: Ligand IDs to include.
        fig_dir: Directory to save the figure.
        n_replicas: Number of replicas per condition.

    Returns:
        Matplotlib Figure.
    """
    n = len(ligand_ids)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, lig_id in zip(axes, ligand_ids):
        for rec_type, color in COLORS.items():
            all_rmsf = []
            for rep_id in range(1, n_replicas + 1):
                df = analysis_results.get((lig_id, rec_type, rep_id), {}).get("rmsf", pd.DataFrame())
                if not df.empty and "rmsf_A" in df.columns:
                    all_rmsf.append(df["rmsf_A"].values)
            if not all_rmsf:
                continue
            min_len = min(len(x) for x in all_rmsf)
            arr  = np.array([x[:min_len] for x in all_rmsf])
            mean = arr.mean(axis=0)
            ax.plot(range(len(mean)), mean, color=color, lw=1.2, label=rec_type, alpha=0.85)

        ax.set_title(LIGAND_DISPLAY.get(lig_id, lig_id), fontsize=10)
        ax.set_xlabel("Residue index", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("RMSF (Å)", fontsize=11)
    fig.suptitle("Per-Residue RMSF — WT vs R206H", fontsize=12)
    plt.tight_layout()
    save_figure(fig, "Fig_RMSF_comparison", fig_dir)
    return fig


def plot_mmgbsa_summary(
    mmgbsa_df: pd.DataFrame,
    fig_dir: Path,
) -> Any:
    """Bar chart of mean MM-GBSA per (ligand, receptor) with replica SEM error bars.

    Args:
        mmgbsa_df: Output of ``mmgbsa.mmgbsa_table``.
        fig_dir: Directory to save the figure.

    Returns:
        Matplotlib Figure, or None if data is insufficient.
    """
    if mmgbsa_df.empty:
        logger.warning("No MM-GBSA data to plot.")
        return None

    grouped = (
        mmgbsa_df.groupby(["ligand_id", "receptor"])["mmgbsa_mean_kcal"]
        .agg(["mean", lambda x: x.std() / math.sqrt(len(x))])
        .rename(columns={"<lambda_0>": "sem"})
        .reset_index()
    )

    ligands  = grouped["ligand_id"].unique()
    receptors = list(COLORS.keys())
    x     = np.arange(len(ligands))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(ligands)), 5))
    for i, rec in enumerate(receptors):
        sub = grouped[grouped["receptor"] == rec].set_index("ligand_id").reindex(ligands)
        ax.bar(
            x + i * width,
            sub["mean"].fillna(0),
            width,
            yerr=sub["sem"].fillna(0),
            color=COLORS[rec],
            label=rec,
            alpha=0.85,
            capsize=4,
        )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(ligands, fontsize=10)
    ax.set_ylabel("Approx. ΔG (kcal/mol)", fontsize=11)
    ax.set_title("MM-GBSA: ACVR1 WT vs R206H (GBn2 approximate)", fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    plt.tight_layout()
    save_figure(fig, "Fig_MMGBSA_summary", fig_dir)
    return fig


def plot_selectivity_scatter(
    selectivity_df: pd.DataFrame,
    fig_dir: Path,
) -> Any:
    """ΔG_WT vs ΔG_R206H scatter annotated with ΔΔG selectivity.

    Points below the y=x diagonal bind R206H better (selective for FOP mutant).

    Args:
        selectivity_df: Output of ``selectivity.compute_selectivity``.
        fig_dir: Directory to save the figure.

    Returns:
        Matplotlib Figure, or None if data is insufficient.
    """
    if selectivity_df.empty or "dG_WT" not in selectivity_df.columns:
        logger.warning("No selectivity data to plot.")
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    for _, row in selectivity_df.iterrows():
        color = "#E91E63" if row.get("selective", False) else "#2196F3"
        ax.errorbar(
            row["dG_WT"], row["dG_R206H"],
            xerr=row.get("dG_WT_sem", 0), yerr=row.get("dG_R206H_sem", 0),
            fmt="o", color=color, markersize=9, capsize=4,
        )
        ax.annotate(
            row["ligand_id"],
            (row["dG_WT"], row["dG_R206H"]),
            textcoords="offset points", xytext=(6, 4), fontsize=9,
        )

    # y = x diagonal
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
           max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5, label="y = x (no selectivity)")
    ax.axhline(ax.get_ylim()[0], color="none")  # keep ylim stable

    ax.set_xlabel("ΔG_bind WT (kcal/mol)", fontsize=11)
    ax.set_ylabel("ΔG_bind R206H (kcal/mol)", fontsize=11)
    ax.set_title("WT vs R206H Selectivity (pink = ΔΔG < −0.5 kcal/mol)", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    save_figure(fig, "Fig_Selectivity_scatter", fig_dir)
    return fig


def plot_vina_vs_mmgbsa(
    mmgbsa_df: pd.DataFrame,
    fig_dir: Path,
) -> Any:
    """Correlation scatter: Vina score vs mean MM-GBSA per ligand.

    Why:
        Shows methodological orthogonality — if Vina and MM-GBSA agree on the
        rank ordering, the binding predictions are more credible.  If they
        diverge, it warrants discussion of why (force-field artefacts vs.
        Vina approximations).

    Args:
        mmgbsa_df: Output of ``mmgbsa.mmgbsa_table`` with ``vina_score`` column.
        fig_dir: Directory to save the figure.

    Returns:
        Matplotlib Figure, or None if data is insufficient.
    """
    if mmgbsa_df.empty or "vina_score" not in mmgbsa_df.columns:
        logger.warning("No Vina/MM-GBSA data to plot correlation.")
        return None

    per_lig = (
        mmgbsa_df.groupby("ligand_id")
        .agg(vina=("vina_score", "first"), mmgbsa_mean=("mmgbsa_mean_kcal", "mean"))
        .dropna()
        .reset_index()
    )
    if len(per_lig) < 2:
        logger.warning("Fewer than 2 ligands with valid data — skipping Vina vs MM-GBSA plot.")
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(per_lig["vina"], per_lig["mmgbsa_mean"], color="#4CAF50", s=80, zorder=3)
    for _, row in per_lig.iterrows():
        ax.annotate(row["ligand_id"], (row["vina"], row["mmgbsa_mean"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=9)

    # Spearman correlation
    from scipy.stats import spearmanr  # type: ignore[import]
    rho, pval = spearmanr(per_lig["vina"], per_lig["mmgbsa_mean"])
    ax.set_title(f"Vina vs MM-GBSA (ρ = {rho:.2f}, p = {pval:.3f})", fontsize=11)
    ax.set_xlabel("Vina score (kcal/mol)", fontsize=11)
    ax.set_ylabel("Mean MM-GBSA (kcal/mol)", fontsize=11)
    plt.tight_layout()
    save_figure(fig, "Fig_Vina_vs_MMGBSA", fig_dir)
    return fig


def plot_acvr1_structural_summary(
    analysis_results: dict[tuple, dict],
    ligand_ids: list[str],
    fig_dir: Path,
    n_replicas: int = 3,
) -> Any:
    """Three-panel ACVR1-specific structural summary per lead.

    Panel rows (one per ligand):
    - Left:   αC-helix In/Out percentage (stacked bar, WT vs R206H)
    - Middle: Hinge H-bond occupancy at His286 and Glu288 (grouped bar)
    - Right:  A-loop mean RMSF Å (WT vs R206H, with ± std across replicas)

    Why:
        These three readouts directly show whether a lead shifts the kinase
        toward the active (αC-in) or inactive conformation, engages the hinge
        correctly, and dampens the pathologically flexible A-loop in the R206H
        mutant — the three mechanistic criteria for a selective FOP inhibitor.

    Args:
        analysis_results: Mapping ``(lig_id, receptor, replica_id)`` → dict with
            key ``"acvr1"`` containing the output of
            ``trajectory.run_acvr1_structural_analysis``.
        ligand_ids: Ligands to include.
        fig_dir: Output directory.
        n_replicas: Number of replicas per condition.

    Returns:
        Matplotlib Figure.
    """
    n_lig = len(ligand_ids)
    fig, axes = plt.subplots(n_lig, 3, figsize=(13, 3.2 * n_lig), squeeze=False)

    for row_idx, lig_id in enumerate(ligand_ids):
        ax_ac, ax_hinge, ax_aloop = axes[row_idx]

        # ── Collect per-receptor summary values ───────────────────────────────
        ac_in:  dict[str, list[float]] = {"WT": [], "R206H": []}
        ac_out: dict[str, list[float]] = {"WT": [], "R206H": []}
        his286: dict[str, list[float]] = {"WT": [], "R206H": []}
        glu288: dict[str, list[float]] = {"WT": [], "R206H": []}
        aloop:  dict[str, list[float]] = {"WT": [], "R206H": []}

        for rec in ("WT", "R206H"):
            for rep in range(1, n_replicas + 1):
                d = analysis_results.get((lig_id, rec, rep), {}).get("acvr1", {})
                if not d or d.get("reason", "ok") != "ok":
                    continue
                ac_in[rec].append(float(d.get("alpha_c_in_pct", float("nan"))) * 100)
                ac_out[rec].append(float(d.get("alpha_c_out_pct", float("nan"))) * 100)
                his286[rec].append(float(d.get("hinge_his286_occupancy", float("nan"))) * 100)
                glu288[rec].append(float(d.get("hinge_glu288_occupancy", float("nan"))) * 100)
                aloop[rec].append(float(d.get("aloop_mean_rmsf_A", float("nan"))))

        # ── Left: αC-helix stacked bar ─────────────────────────────────────────
        xpos = np.array([0.0, 0.5])
        for xi, rec in zip(xpos, ("WT", "R206H")):
            in_mean  = float(np.nanmean(ac_in[rec]))  if ac_in[rec]  else 0.0
            out_mean = float(np.nanmean(ac_out[rec])) if ac_out[rec] else 0.0
            other    = max(0.0, 100.0 - in_mean - out_mean)
            ax_ac.bar(xi, in_mean,  width=0.35, color=COLORS[rec], alpha=0.85, label=f"{rec} In")
            ax_ac.bar(xi, out_mean, width=0.35, bottom=in_mean, color=COLORS[rec], alpha=0.4, label=f"{rec} Out")
            ax_ac.bar(xi, other,    width=0.35, bottom=in_mean + out_mean, color="lightgray", alpha=0.5)
        ax_ac.set_xticks(xpos)
        ax_ac.set_xticklabels(["WT", "R206H"], fontsize=9)
        ax_ac.set_ylabel("% frames", fontsize=9)
        ax_ac.set_ylim(0, 105)
        ax_ac.set_title("αC-helix In (solid) / Out (pale)", fontsize=9)

        # ── Middle: hinge H-bond occupancy ────────────────────────────────────
        groups = ["His286", "Glu288"]
        wt_vals  = [float(np.nanmean(his286["WT"]))  if his286["WT"]  else 0.0,
                    float(np.nanmean(glu288["WT"]))  if glu288["WT"]  else 0.0]
        rh_vals  = [float(np.nanmean(his286["R206H"])) if his286["R206H"] else 0.0,
                    float(np.nanmean(glu288["R206H"])) if glu288["R206H"] else 0.0]
        x2 = np.arange(len(groups))
        ax_hinge.bar(x2 - 0.18, wt_vals,  width=0.32, color=COLORS["WT"],    label="WT",    alpha=0.85)
        ax_hinge.bar(x2 + 0.18, rh_vals,  width=0.32, color=COLORS["R206H"], label="R206H", alpha=0.85)
        ax_hinge.axhline(30, color="gray", lw=0.7, ls=":", alpha=0.6)
        ax_hinge.set_xticks(x2)
        ax_hinge.set_xticklabels(groups, fontsize=9)
        ax_hinge.set_ylabel("Occupancy (%)", fontsize=9)
        ax_hinge.set_ylim(0, 105)
        ax_hinge.set_title("Hinge H-bond occupancy", fontsize=9)
        ax_hinge.legend(fontsize=8, frameon=False)

        # ── Right: A-loop RMSF ─────────────────────────────────────────────────
        for rec, color in COLORS.items():
            vals = aloop[rec]
            if not vals:
                continue
            mean_v = float(np.nanmean(vals))
            std_v  = float(np.nanstd(vals))
            xi_map = {"WT": 0.0, "R206H": 0.4}
            ax_aloop.bar(xi_map[rec], mean_v, width=0.32,
                         yerr=std_v, color=color, alpha=0.85,
                         capsize=4, label=rec)

        ax_aloop.set_xticks([0.0, 0.4])
        ax_aloop.set_xticklabels(["WT", "R206H"], fontsize=9)
        ax_aloop.set_ylabel("Mean RMSF (Å)", fontsize=9)
        ax_aloop.set_title("A-loop RMSF (354–390)", fontsize=9)

        # Row label
        axes[row_idx][0].set_ylabel(
            f"{LIGAND_DISPLAY.get(lig_id, lig_id)}\n% frames",
            fontsize=8,
        )

    fig.suptitle("ACVR1 Structural Monitors — αC-helix / Hinge / A-loop", fontsize=12)
    plt.tight_layout()
    save_figure(fig, "Fig_ACVR1_structural", fig_dir)
    return fig
