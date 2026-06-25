#!/usr/bin/env python3
"""Generate MD analysis figures for KinetiDiff ACVR1/ALK2 campaign.

Reads per-run CSVs/JSONs from ORCD scratch and writes publication-ready
figures to results/md/figures/.

Usage:
    ~/.conda/envs/kinetidiff-md/bin/python3 results/md/plot_md_results.py
"""

import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
ANALYSIS_BASE = os.path.expanduser(
    "~/orcd/scratch/kinetidiff-md/analysis"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

LIGANDS = ["L1", "L2", "L3"]
CONDITIONS = ["WT", "R206H"]
REPS = ["rep1", "rep2"]

SYSTEMS = {
    f"{lig}_{cond}": [f"{lig}_{cond}_{rep}" for rep in REPS]
    for lig in LIGANDS
    for cond in CONDITIONS
}

PALETTE = {"WT": "#4878CF", "R206H": "#E87F3A"}
LIG_MARKERS = {"L1": "o", "L2": "s", "L3": "^"}

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_rmsd(run):
    path = os.path.join(ANALYSIS_BASE, run, "rmsd.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    t   = np.array([float(r["time_ns"])           for r in rows])
    bb  = np.array([float(r["backbone_rmsd_A"])   for r in rows])
    lig = np.array([float(r["ligand_rmsd_A"])      for r in rows])
    return t, bb, lig


def load_rmsf(run):
    path = os.path.join(ANALYSIS_BASE, run, "rmsf.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    res  = np.array([int(r["residue"])   for r in rows])
    rmsf = np.array([float(r["rmsf_A"]) for r in rows])
    return res, rmsf


def load_mmgbsa(run):
    path = os.path.join(ANALYSIS_BASE, run, "mmgbsa.csv")
    with open(path) as f:
        row = list(csv.DictReader(f))[0]
    return float(row["mean"]), float(row["std"])


def load_structural(run):
    path = os.path.join(ANALYSIS_BASE, run, "acvr1_structural.json")
    with open(path) as f:
        return json.load(f)


def load_contacts(run):
    path = os.path.join(ANALYSIS_BASE, run, "pocket_contacts.json")
    with open(path) as f:
        return json.load(f)


def rep_avg(values):
    return np.mean(values)


# ── Figure 1: RMSD traces (backbone + ligand), 2×3 grid ──────────────────────
def fig_rmsd_traces():
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey="row")
    fig.suptitle("RMSD over 100 ns MD trajectories", fontsize=14, fontweight="bold")

    for col, lig in enumerate(LIGANDS):
        for row, cond in enumerate(CONDITIONS):
            ax = axes[row][col]
            sysname = f"{lig}_{cond}"
            color = PALETTE[cond]

            for rep in REPS:
                t, bb, lrmsd = load_rmsd(f"{sysname}_{rep}")
                alpha = 0.9 if rep == "rep1" else 0.5
                lw = 1.2 if rep == "rep1" else 0.8
                ax.plot(t, bb,    color=color,   alpha=alpha, lw=lw,
                        label=f"Backbone ({rep})" if col == 0 else None)
                ax.plot(t, lrmsd, color=color,   alpha=alpha, lw=lw,
                        ls="--",
                        label=f"Ligand ({rep})" if col == 0 else None)

            ax.set_title(f"{lig} — {cond}", fontsize=11)
            ax.set_xlabel("Time (ns)")
            if col == 0:
                ax.set_ylabel("RMSD (Å)")

    solid = mpatches.Patch(color="grey", label="Backbone")
    dashed_patch = plt.Line2D([0], [0], color="grey", ls="--", label="Ligand")
    wt_patch  = mpatches.Patch(color=PALETTE["WT"],    label="WT")
    r206h_patch = mpatches.Patch(color=PALETTE["R206H"], label="R206H")
    fig.legend(handles=[solid, dashed_patch, wt_patch, r206h_patch],
               loc="lower center", ncol=4, frameon=True, fontsize=10,
               bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(OUT_DIR, "fig1_rmsd_traces.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


# ── Figure 2: MM-GBSA bar chart with error bars ───────────────────────────────
def fig_mmgbsa():
    fig, ax = plt.subplots(figsize=(9, 5))

    sysnames = list(SYSTEMS.keys())
    means, stds, colors = [], [], []
    for sysname, reps in SYSTEMS.items():
        rep_means = [load_mmgbsa(r)[0] for r in reps]
        rep_stds  = [load_mmgbsa(r)[1] for r in reps]
        means.append(np.mean(rep_means))
        stds.append(np.mean(rep_stds))
        cond = sysname.split("_")[1]
        colors.append(PALETTE[cond])

    x = np.arange(len(sysnames))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5,
                  error_kw={"elinewidth": 1.5, "alpha": 0.8},
                  edgecolor="white", linewidth=0.8, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(sysnames, rotation=30, ha="right")
    ax.set_ylabel("MM-GBSA ΔG (kcal/mol)")
    ax.set_title("MM-GBSA Binding Free Energy — base form", fontweight="bold")
    ax.axhline(0, color="black", lw=0.8, ls="--")

    # annotate values
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m - 1.5,
                f"{m:.1f}", ha="center", va="top", fontsize=8.5, color="white",
                fontweight="bold")

    wt_patch    = mpatches.Patch(color=PALETTE["WT"],    label="WT")
    r206h_patch = mpatches.Patch(color=PALETTE["R206H"], label="R206H")
    ax.legend(handles=[wt_patch, r206h_patch], frameon=True)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_mmgbsa.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


# ── Figure 3: Pocket occupancy (bound %) ─────────────────────────────────────
def fig_pocket_occupancy():
    fig, ax = plt.subplots(figsize=(9, 5))

    sysnames = list(SYSTEMS.keys())
    bound_avgs, bound_r1, bound_r2, colors = [], [], [], []
    for sysname, reps in SYSTEMS.items():
        vals = [load_contacts(r)["bound_pct"] for r in reps]
        bound_r1.append(vals[0])
        bound_r2.append(vals[1])
        bound_avgs.append(np.mean(vals))
        cond = sysname.split("_")[1]
        colors.append(PALETTE[cond])

    x = np.arange(len(sysnames))
    bars = ax.bar(x, bound_avgs, color=colors, edgecolor="white",
                  linewidth=0.8, alpha=0.88)

    # individual rep dots
    for i, (r1, r2) in enumerate(zip(bound_r1, bound_r2)):
        ax.plot([i], [r1], "o", color="black", ms=5, zorder=5)
        ax.plot([i], [r2], "D", color="black", ms=4, zorder=5, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(sysnames, rotation=30, ha="right")
    ax.set_ylabel("Pocket-bound frames (%)")
    ax.set_title("Ligand Pocket Occupancy over 100 ns", fontweight="bold")
    ax.set_ylim(0, 60)

    wt_patch    = mpatches.Patch(color=PALETTE["WT"],    label="WT")
    r206h_patch = mpatches.Patch(color=PALETTE["R206H"], label="R206H")
    rep1_dot    = plt.Line2D([0], [0], marker="o", color="black", ls="none", label="rep1")
    rep2_dot    = plt.Line2D([0], [0], marker="D", color="black", ls="none", label="rep2", alpha=0.6)
    ax.legend(handles=[wt_patch, r206h_patch, rep1_dot, rep2_dot], frameon=True)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig3_pocket_occupancy.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


# ── Figure 4: ACVR1 structural metrics heatmap ───────────────────────────────
def fig_structural_heatmap():
    metrics = ["alpha_c_in_pct", "alpha_c_out_pct",
               "hinge_his286_occupancy", "aloop_mean_rmsf_A"]
    labels  = ["αC-in (%)", "αC-out (%)", "H286 occ.", "A-loop RMSF (Å)"]

    sysnames = list(SYSTEMS.keys())
    data = np.zeros((len(sysnames), len(metrics)))
    for i, (sysname, reps) in enumerate(SYSTEMS.items()):
        for j, metric in enumerate(metrics):
            vals = [load_structural(r)[metric] for r in reps]
            data[i, j] = np.mean(vals)

    # normalise each column 0→1 for colour, keep raw for annotation
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data_norm.T, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(sysnames)))
    ax.set_xticklabels(sysnames, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("ACVR1 Structural Metrics (rep-averaged)", fontweight="bold")

    for i in range(len(sysnames)):
        for j in range(len(metrics)):
            val = data[i, j]
            txt = f"{val:.2f}" if metric != "aloop_mean_rmsf_A" else f"{val:.3f}"
            txt = f"{val:.3f}"
            ax.text(i, j, txt, ha="center", va="center", fontsize=8.5,
                    color="black" if 0.2 < data_norm[i, j] < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="Normalized value (within metric)")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig4_structural_heatmap.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


# ── Figure 5: Summary scatter — MM-GBSA vs bound% coloured by ligand ─────────
def fig_summary_scatter():
    fig, ax = plt.subplots(figsize=(7, 5))

    for sysname, reps in SYSTEMS.items():
        lig, cond = sysname.split("_")
        mmgbsa = np.mean([load_mmgbsa(r)[0] for r in reps])
        bound  = np.mean([load_contacts(r)["bound_pct"] for r in reps])

        ax.scatter(bound, mmgbsa,
                   color=PALETTE[cond],
                   marker=LIG_MARKERS[lig],
                   s=120, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(sysname, (bound, mmgbsa),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)

    ax.set_xlabel("Pocket occupancy (%)")
    ax.set_ylabel("MM-GBSA ΔG (kcal/mol)")
    ax.set_title("Binding Quality: MM-GBSA vs Pocket Occupancy", fontweight="bold")

    wt_patch    = mpatches.Patch(color=PALETTE["WT"],    label="WT")
    r206h_patch = mpatches.Patch(color=PALETTE["R206H"], label="R206H")
    l1 = plt.Line2D([0],[0], marker="o", color="grey", ls="none", label="L1")
    l2 = plt.Line2D([0],[0], marker="s", color="grey", ls="none", label="L2")
    l3 = plt.Line2D([0],[0], marker="^", color="grey", ls="none", label="L3")
    ax.legend(handles=[wt_patch, r206h_patch, l1, l2, l3], frameon=True,
              loc="lower right")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig5_summary_scatter.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


# ── Figure 6: RMSF per-residue for R206H runs (highlight A-loop) ─────────────
def fig_rmsf_r206h():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.suptitle("Per-residue RMSF — R206H condition (rep average)", fontweight="bold")

    for col, lig in enumerate(LIGANDS):
        ax = axes[col]
        sysname = f"{lig}_R206H"
        color = PALETTE["R206H"]

        res_all, rmsf_all = [], []
        for rep in REPS:
            res, rmsf = load_rmsf(f"{sysname}_{rep}")
            res_all.append(res)
            rmsf_all.append(rmsf)

        # align on common residues
        common_res = res_all[0]
        avg_rmsf = np.mean(rmsf_all, axis=0)

        ax.plot(common_res, avg_rmsf, color=color, lw=1.2)
        ax.fill_between(common_res, 0, avg_rmsf, alpha=0.2, color=color)
        ax.set_title(f"{lig}_R206H", fontsize=11)
        ax.set_xlabel("Residue index")
        if col == 0:
            ax.set_ylabel("RMSF (Å)")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig6_rmsf_r206h.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating MD analysis figures...")
    fig_rmsd_traces()
    fig_mmgbsa()
    fig_pocket_occupancy()
    fig_structural_heatmap()
    fig_summary_scatter()
    fig_rmsf_r206h()
    print(f"\nAll figures saved to {OUT_DIR}")
