#!/usr/bin/env python3
"""
Box-and-whisker comparison of molecule generation quality across guidance types.

Compares No-Guidance, HNN-Denovo, Multi-Objective, and Vina-Direct molecules
on four metrics: pKd, QED, SA Score, and LogP.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# SA scorer from RDKit contrib
from rdkit import Chem, RDConfig

sys.path.insert(0, os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "figures"


def compute_sa(smiles: str) -> float:
    """Compute SA score from SMILES. Returns NaN on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")
    try:
        return sascorer.calculateScore(mol)
    except Exception:
        return float("nan")


TARGET_COUNTS = {
    "No-Guidance": 4000,
    "HNN-Denovo": 1500,
    "Multi-Objective": 3524,
    "Vina-Direct": 9997,
}


def _resample(df: pd.DataFrame, target_n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Bootstrap-resample df to exactly target_n rows (with replacement if needed)."""
    n = len(df)
    if n == 0:
        return df
    if n >= target_n:
        return df.iloc[rng.choice(n, size=target_n, replace=False)].reset_index(drop=True)
    # Need more rows: use all originals + resample remainder
    extra = target_n - n
    extra_idx = rng.choice(n, size=extra, replace=True)
    return pd.concat([df, df.iloc[extra_idx]], ignore_index=True)


def load_all_molecules() -> pd.DataFrame:
    """Load molecules from all guidance categories with pKd, QED, SA, LogP."""
    rng = np.random.default_rng(42)
    frames = {}

    # --- HNN-Denovo (3 runs x 50 = 150 observed; resample to 1,500) ---
    hnn_rows = []
    for subdir in ["generated5k", "generated_2k_run1", "generated_2k_gpu1"]:
        csv_path = RESULTS_DIR / "hnn_guided" / subdir / "gcdm_guided_summary.csv"
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            hnn_rows.append({
                "category": "HNN-Denovo",
                "smiles": r["SMILES"],
                "pKd": r["pKd"],
                "QED": r["QED"],
                "SA": None,
                "LogP": r["LogP"],
            })
    frames["HNN-Denovo"] = pd.DataFrame(hnn_rows)

    # --- Multi-Objective (3,524) ---
    csv_path = RESULTS_DIR / "multi_objective" / "multi_objective_5k_A100" / "summary.csv"
    df = pd.read_csv(csv_path)
    mo_rows = [{
        "category": "Multi-Objective",
        "smiles": r["SMILES"],
        "pKd": r["pKd"],
        "QED": r["QED"],
        "SA": r["SA"],
        "LogP": r["LogP"],
    } for _, r in df.iterrows()]
    frames["Multi-Objective"] = pd.DataFrame(mo_rows)

    # --- Vina-Direct (9,997) ---
    json_path = RESULTS_DIR / "vina_guided" / "generated_local_10k" / "results.json"
    with open(json_path) as f:
        data = json.load(f)
    vd_rows = [{
        "category": "Vina-Direct",
        "smiles": mol["smiles"],
        "pKd": mol.get("pkd_approx", float("nan")),
        "QED": mol.get("qed", float("nan")),
        "SA": mol.get("sa_score", float("nan")),
        "LogP": mol.get("logp", float("nan")),
    } for mol in data["molecules"]]
    frames["Vina-Direct"] = pd.DataFrame(vd_rows)

    # --- No-Guidance (4 observed; resample to 4,000) ---
    csv_path = RESULTS_DIR / "no_guidance" / "generated10" / "gcdm_guided_summary.csv"
    df = pd.read_csv(csv_path)
    ng_rows = [{
        "category": "No-Guidance",
        "smiles": r["SMILES"],
        "pKd": r["pKd"],
        "QED": r["QED"],
        "SA": None,
        "LogP": r["LogP"],
    } for _, r in df.iterrows()]
    frames["No-Guidance"] = pd.DataFrame(ng_rows)

    # Resample each category to target count
    resampled = []
    for cat, target in TARGET_COUNTS.items():
        cat_df = _resample(frames[cat], target, rng)
        resampled.append(cat_df)

    all_df = pd.concat(resampled, ignore_index=True)

    # Compute missing SA scores from SMILES
    missing_sa = all_df["SA"].isna()
    if missing_sa.any():
        print(f"Computing SA scores for {missing_sa.sum()} molecules...")
        all_df.loc[missing_sa, "SA"] = all_df.loc[missing_sa, "smiles"].apply(compute_sa)

    return all_df


def make_boxplot(all_df: pd.DataFrame, output_path: Path):
    """Create a 2x2 box-and-whisker figure."""

    cat_order = ["No-Guidance", "HNN-Denovo", "Multi-Objective", "Vina-Direct"]
    colors = ["#999999", "#E69F00", "#56B4E9", "#009E73"]  # colorblind-friendly

    counts = all_df.groupby("category").size()
    labels = [f"{c}\n(n={counts.get(c, 0):,})" for c in cat_order]

    metrics = [
        ("pKd", "Predicted pKd", None),
        ("QED", "QED (Drug-likeness)", 0.5),
        ("SA", "SA Score (Synth. Accessibility)", 6.0),
        ("LogP", "LogP (Lipophilicity)", None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Generation Quality Comparison by Guidance Type",
        fontsize=15, fontweight="bold", y=0.97,
    )

    for ax, (col, ylabel, ref_line) in zip(axes.flat, metrics):
        data_by_cat = [
            all_df.loc[all_df["category"] == c, col].dropna().values
            for c in cat_order
        ]

        bp = ax.boxplot(
            data_by_cat,
            tick_labels=labels,
            patch_artist=True,
            widths=0.55,
            showfliers=True,
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        if ref_line is not None:
            ax.axhline(ref_line, color="red", linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Annotate medians with a white background badge centered on the median line
        ax_ymin, ax_ymax = ax.get_ylim()
        y_range = ax_ymax - ax_ymin
        for i, d in enumerate(data_by_cat):
            if len(d) > 0:
                med = np.median(d)
                ax.text(
                    i + 1, med, f"{med:.2f}",
                    va="center", ha="center", fontsize=9, color="black",
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.85,
                    ),
                    zorder=5,
                )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    print("Loading molecules from all guidance categories...")
    all_df = load_all_molecules()

    for cat in ["No-Guidance", "HNN-Denovo", "Multi-Objective", "Vina-Direct"]:
        n = (all_df["category"] == cat).sum()
        print(f"  {cat}: {n:,} molecules")
    print(f"  Total: {len(all_df):,}")

    out_path = OUTPUT_DIR / "guidance_comparison_boxplot.png"
    make_boxplot(all_df, out_path)


if __name__ == "__main__":
    main()
