#!/usr/bin/env python3
"""
Publication-ready figures for GCDM guided molecule generation paper.

Generates 4 figures:
  1. pKd vs QED scatter with Pareto front
  2. Property distribution KDEs (MW, LogP, HBD, HBA, TPSA)
  3. Chemical diversity heatmap of top 100 redocked molecules
  4. Top 10 molecule 2D structure grid
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem, Descriptors, Draw
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "figures"

COLORS = {
    "No-Guidance": "#999999",
    "HNN-Denovo": "#E69F00",
    "Multi-Objective": "#56B4E9",
    "Vina-Direct": "#009E73",
}
CAT_ORDER = ["No-Guidance", "HNN-Denovo", "Multi-Objective", "Vina-Direct"]
MARKERS = {"No-Guidance": "D", "HNN-Denovo": "^", "Multi-Objective": "s", "Vina-Direct": "o"}

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_descriptors(smiles: str):
    """Return (SA, HBD, HBA, TPSA, MW) from SMILES, or NaNs on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (np.nan,) * 5
    try:
        sa = sascorer.calculateScore(mol)
    except Exception:
        sa = np.nan
    return (
        sa,
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolWt(mol),
    )


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_all_molecules() -> pd.DataFrame:
    """Load all 13,675 molecules with unified property columns."""
    rows = []

    # HNN-Denovo (3 runs x 50)
    for subdir in ["generated5k", "generated_2k_run1", "generated_2k_gpu1"]:
        csv_path = RESULTS_DIR / "hnn_guided" / subdir / "gcdm_guided_summary.csv"
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            rows.append({
                "category": "HNN-Denovo", "smiles": r["SMILES"],
                "pKd": r["pKd"], "QED": r["QED"], "LogP": r["LogP"],
                "MW": r["MW"], "SA": None, "HBD": None, "HBA": None, "TPSA": None,
            })

    # Multi-Objective (3,524)
    csv_path = RESULTS_DIR / "multi_objective" / "multi_objective_5k_A100" / "summary.csv"
    df = pd.read_csv(csv_path)
    for _, r in df.iterrows():
        rows.append({
            "category": "Multi-Objective", "smiles": r["SMILES"],
            "pKd": r["pKd"], "QED": r["QED"], "LogP": r["LogP"],
            "MW": r["MW"], "SA": r["SA"], "HBD": None, "HBA": None, "TPSA": None,
        })

    # Vina-Direct (9,997)
    json_path = RESULTS_DIR / "vina_guided" / "generated_local_10k" / "results.json"
    with open(json_path) as f:
        data = json.load(f)
    for mol in data["molecules"]:
        rows.append({
            "category": "Vina-Direct", "smiles": mol["smiles"],
            "pKd": mol.get("pkd_approx", np.nan),
            "QED": mol.get("qed", np.nan),
            "LogP": mol.get("logp", np.nan),
            "MW": mol.get("mol_weight", np.nan),
            "SA": mol.get("sa_score", np.nan),
            "HBD": mol.get("hbd", np.nan),
            "HBA": mol.get("hba", np.nan),
            "TPSA": mol.get("tpsa", np.nan),
        })

    # No-Guidance (4)
    csv_path = RESULTS_DIR / "no_guidance" / "generated10" / "gcdm_guided_summary.csv"
    df = pd.read_csv(csv_path)
    for _, r in df.iterrows():
        rows.append({
            "category": "No-Guidance", "smiles": r["SMILES"],
            "pKd": r["pKd"], "QED": r["QED"], "LogP": r["LogP"],
            "MW": r["MW"], "SA": None, "HBD": None, "HBA": None, "TPSA": None,
        })

    all_df = pd.DataFrame(rows)

    # Fill missing descriptors from SMILES
    missing = all_df[["SA", "HBD", "HBA", "TPSA"]].isna().any(axis=1)
    if missing.any():
        n = missing.sum()
        print(f"  Computing descriptors for {n} molecules...")
        desc = all_df.loc[missing, "smiles"].apply(_compute_descriptors)
        desc_df = pd.DataFrame(desc.tolist(), index=desc.index,
                               columns=["SA", "HBD", "HBA", "TPSA", "MW_calc"])
        for col in ["SA", "HBD", "HBA", "TPSA"]:
            all_df.loc[missing, col] = desc_df[col]
        # Also fill MW if it was missing
        mw_na = all_df["MW"].isna() & missing
        if mw_na.any():
            all_df.loc[mw_na, "MW"] = desc_df.loc[mw_na, "MW_calc"]

    return all_df


def load_top100() -> pd.DataFrame:
    """Load top 100 redocked molecules."""
    return pd.read_csv(RESULTS_DIR / "redocking" / "top_100.csv")


# ---------------------------------------------------------------------------
# Figure 1: pKd vs QED scatter with Pareto front
# ---------------------------------------------------------------------------
def compute_pareto_2d(pkd: np.ndarray, qed: np.ndarray) -> np.ndarray:
    """Fast 2D Pareto front (maximize both). Returns boolean mask."""
    idx = np.argsort(-pkd)  # sort by pKd descending
    mask = np.zeros(len(pkd), dtype=bool)
    max_qed = -np.inf
    for i in idx:
        if qed[i] >= max_qed:
            mask[i] = True
            max_qed = qed[i]
    return mask


def plot_pkd_vs_qed(all_df: pd.DataFrame, out_dir: Path):
    """Figure 1: pKd vs QED scatter with Pareto front."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot order: largest first (background), smallest last (foreground)
    plot_order = ["Vina-Direct", "Multi-Objective", "HNN-Denovo", "No-Guidance"]
    sizes = {"Vina-Direct": 6, "Multi-Objective": 14, "HNN-Denovo": 35, "No-Guidance": 80}
    alphas = {"Vina-Direct": 0.25, "Multi-Objective": 0.45, "HNN-Denovo": 0.8, "No-Guidance": 1.0}

    counts = all_df.groupby("category").size()
    for cat in plot_order:
        sub = all_df[all_df["category"] == cat].dropna(subset=["pKd", "QED"])
        ax.scatter(
            sub["pKd"], sub["QED"],
            c=COLORS[cat], marker=MARKERS[cat], s=sizes[cat],
            alpha=alphas[cat], edgecolors="none",
            label=f"{cat} (n={counts.get(cat, 0):,})",
            zorder=2 if cat in ("Vina-Direct", "Multi-Objective") else 3,
        )

    # Pareto front across all molecules
    valid = all_df.dropna(subset=["pKd", "QED"])
    pareto_mask = compute_pareto_2d(valid["pKd"].values, valid["QED"].values)
    pareto_pts = valid[pareto_mask].sort_values("pKd")

    ax.plot(pareto_pts["pKd"], pareto_pts["QED"], "k--", linewidth=1.2, alpha=0.7,
            label="Pareto front", zorder=4)
    ax.scatter(pareto_pts["pKd"], pareto_pts["QED"], c="none", edgecolors="black",
               s=30, linewidths=0.8, zorder=5)

    # Ideal region shading
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.axhspan(0.6, ylim[1], xmin=0, xmax=1, alpha=0.04, color="green", zorder=0)
    ax.axvspan(7.0, xlim[1], ymin=0, ymax=1, alpha=0.04, color="green", zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("Predicted pKd (binding affinity)")
    ax.set_ylabel("QED (drug-likeness)")
    ax.set_title("pKd vs QED: Guidance Strategy Comparison")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.2)

    out = out_dir / "fig1_pkd_vs_qed_pareto.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Figure 2: Property distribution KDEs
# ---------------------------------------------------------------------------
def plot_property_distributions(all_df: pd.DataFrame, out_dir: Path):
    """Figure 2: KDE distributions of MW, LogP, HBD, HBA, TPSA."""
    props = [
        ("MW", "Molecular Weight (Da)", 500, "MW < 500"),
        ("LogP", "LogP", 5, "LogP < 5"),
        ("HBD", "H-Bond Donors", 5, "HBD \u2264 5"),
        ("HBA", "H-Bond Acceptors", 10, "HBA \u2264 10"),
        ("TPSA", "TPSA (\u00c5\u00b2)", 140, "TPSA < 140"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flat

    for idx, (col, label, ref_val, ref_label) in enumerate(props):
        ax = axes_flat[idx]

        for cat in CAT_ORDER:
            sub = all_df.loc[all_df["category"] == cat, col].dropna()
            if len(sub) < 2:
                # Too few points for KDE — use rug only
                ax.plot(sub.values, [0.01] * len(sub), "|", color=COLORS[cat],
                        markersize=15, markeredgewidth=2, label=cat)
                continue
            try:
                sns.kdeplot(sub, ax=ax, color=COLORS[cat], linewidth=1.8,
                            label=cat, fill=True, alpha=0.15)
            except Exception:
                ax.hist(sub, bins=30, density=True, alpha=0.3, color=COLORS[cat], label=cat)

            # Rug for small categories
            if len(sub) <= 150:
                ax.plot(sub.values, [0] * len(sub), "|", color=COLORS[cat],
                        markersize=8, markeredgewidth=1, alpha=0.6)

        ax.axvline(ref_val, color="red", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(ref_val, ax.get_ylim()[1] * 0.92, f" {ref_label}", color="red",
                fontsize=8, va="top")

        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)

    # Use the 6th panel for a shared legend
    ax_legend = axes_flat[5]
    ax_legend.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.95, 0.08),
               fontsize=11, framealpha=0.9, title="Guidance Type")

    # Remove per-subplot legends
    for idx in range(5):
        leg = axes_flat[idx].get_legend()
        if leg:
            leg.remove()

    fig.suptitle("Property Distributions by Guidance Type", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = out_dir / "fig2_property_distributions.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Figure 3: Chemical diversity heatmap
# ---------------------------------------------------------------------------
def plot_diversity_heatmap(top100: pd.DataFrame, out_dir: Path):
    """Figure 3: Tanimoto similarity heatmap of top 100 redocked molecules."""
    # Parse molecules and compute fingerprints
    smiles_list = top100["smiles"].tolist()
    mols = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_idx.append(i)

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    n = len(fps)

    # Pairwise Tanimoto similarity matrix
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sim_matrix[i] = sims

    # Hierarchical clustering
    dist_condensed = squareform(1 - sim_matrix, checks=False)
    dist_condensed = np.clip(dist_condensed, 0, None)  # fix floating point negatives
    Z = linkage(dist_condensed, method="average")
    order = leaves_list(Z)
    sim_ordered = sim_matrix[np.ix_(order, order)]

    # Off-diagonal similarities
    triu_idx = np.triu_indices(n, k=1)
    off_diag = sim_matrix[triu_idx]
    mean_sim = np.mean(off_diag)
    diversity = 1 - mean_sim

    # Plot
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.3)

    # Heatmap
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(sim_ordered, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax1.set_xlabel("Molecule Rank (clustered)")
    ax1.set_ylabel("Molecule Rank (clustered)")
    ax1.set_title("Pairwise Tanimoto Similarity\n(Top 100 Vina-Direct Candidates)")
    cbar = fig.colorbar(im, ax=ax1, shrink=0.8, label="Tanimoto Similarity")

    # Histogram
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(off_diag, bins=40, color=COLORS["Vina-Direct"], alpha=0.7, edgecolor="white")
    ax2.axvline(mean_sim, color="red", linestyle="--", linewidth=1.2)
    ax2.text(mean_sim + 0.02, ax2.get_ylim()[1] * 0.9,
             f"Mean: {mean_sim:.3f}\nDiversity: {diversity:.3f}",
             fontsize=10, color="red", va="top")
    ax2.set_xlabel("Tanimoto Similarity")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Pairwise Similarity\nDistribution")
    ax2.set_xlim(0, 1)

    out = out_dir / "fig3_chemical_diversity.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Figure 4: Top 10 molecule 2D structures
# ---------------------------------------------------------------------------
def plot_top_structures(top100: pd.DataFrame, out_dir: Path):
    """Figure 4: 2D structure grid of top 10 molecules."""
    top10 = top100.sort_values("vina_score").head(10).reset_index(drop=True)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Top 10 Generated Candidates for ACVR1/ALK2\n(Redocked with AutoDock Vina)",
                 fontsize=14, fontweight="bold", y=1.0)

    for idx, (_, row) in enumerate(top10.iterrows()):
        ax = axes.flat[idx]
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is not None:
            AllChem.Compute2DCoords(mol)
            img = Draw.MolToImage(mol, size=(400, 400))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Parse\nError", ha="center", va="center", fontsize=12)

        ax.axis("off")
        ax.set_title(
            f"#{idx + 1}  Vina: {row['vina_score']:.1f} kcal/mol\n"
            f"pKd: {row['pkd']:.2f}   QED: {row['qed']:.2f}\n"
            f"SA: {row['sa_score']:.1f}   MW: {row['mw']:.0f}",
            fontsize=8.5, pad=4,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = out_dir / "fig4_top_molecule_structures.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all molecules...")
    all_df = load_all_molecules()
    for cat in CAT_ORDER:
        n = (all_df["category"] == cat).sum()
        print(f"  {cat}: {n:,}")
    print(f"  Total: {len(all_df):,}\n")

    top100 = load_top100()
    print(f"Top 100 redocked loaded ({len(top100)} rows)\n")

    print("Generating figures...")
    plot_pkd_vs_qed(all_df, OUTPUT_DIR)
    plot_property_distributions(all_df, OUTPUT_DIR)
    plot_diversity_heatmap(top100, OUTPUT_DIR)
    plot_top_structures(top100, OUTPUT_DIR)

    print("\nAll figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
