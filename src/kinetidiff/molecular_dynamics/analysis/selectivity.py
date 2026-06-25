"""WT-vs-R206H selectivity: compute ΔΔG per lead and flag selective compounds.

ΔΔG_bind = ΔG_R206H − ΔG_WT
    < −0.5 kcal/mol → preferentially binds R206H (FOP-selective)
    > +0.5 kcal/mol → preferentially binds WT

Limitations (must be stated in paper):
    - GBn2 noise floor is ≈ 0.5 kcal/mol (R²≈0.63 for relative ranking).
    - ΔΔG values within ± 0.5 kcal/mol are within the noise and must be
      reported with explicit confidence intervals, not as conclusive.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from kinetidiff._logging import get_logger

logger = get_logger(__name__)

# ΔΔG threshold for FOP-selectivity claim (kcal/mol)
SELECTIVITY_THRESHOLD_KCAL: float = -0.5


def compute_selectivity(
    mmgbsa_df: pd.DataFrame,
    threshold_kcal: float = SELECTIVITY_THRESHOLD_KCAL,
) -> pd.DataFrame:
    """Compute per-ligand ΔΔG_bind = ΔG_R206H − ΔG_WT from MM-GBSA means.

    Averages MM-GBSA scores across replicas per (ligand, receptor) condition,
    computes ΔΔG, propagates the SEM, and flags selective compounds.

    Args:
        mmgbsa_df: Output of ``mmgbsa_table``.  Must have columns
            ``ligand_id``, ``receptor``, ``mmgbsa_mean_kcal``.
        threshold_kcal: ΔΔG below this value is flagged as selective for R206H.

    Returns:
        DataFrame with one row per ligand, columns: ``ligand_id``,
        ``dG_WT``, ``dG_WT_sem``, ``dG_R206H``, ``dG_R206H_sem``,
        ``ddG_bind``, ``ddG_sem``, ``selective``, ``within_noise``.
    """
    if mmgbsa_df.empty:
        logger.warning("Empty MM-GBSA DataFrame — no selectivity to compute.")
        return pd.DataFrame()

    required = {"ligand_id", "receptor", "mmgbsa_mean_kcal"}
    if not required.issubset(mmgbsa_df.columns):
        missing = required - set(mmgbsa_df.columns)
        raise ValueError(f"mmgbsa_df missing columns: {missing}")

    rows = []
    all_ligands = mmgbsa_df["ligand_id"].unique()

    for lig_id in sorted(all_ligands):
        sub = mmgbsa_df[mmgbsa_df["ligand_id"] == lig_id]
        wt_vals    = sub[sub["receptor"] == "WT"]["mmgbsa_mean_kcal"].dropna()
        r206h_vals = sub[sub["receptor"] == "R206H"]["mmgbsa_mean_kcal"].dropna()

        if wt_vals.empty or r206h_vals.empty:
            logger.warning("[%s] Missing WT or R206H data — skipping selectivity.", lig_id)
            continue

        dg_wt      = float(wt_vals.mean())
        dg_wt_sem  = float(wt_vals.std() / np.sqrt(len(wt_vals))) if len(wt_vals) > 1 else float("nan")
        dg_r206h   = float(r206h_vals.mean())
        dg_r206h_sem = float(r206h_vals.std() / np.sqrt(len(r206h_vals))) if len(r206h_vals) > 1 else float("nan")

        ddg = dg_r206h - dg_wt
        ddg_sem = float(np.sqrt(dg_wt_sem**2 + dg_r206h_sem**2)) if not (np.isnan(dg_wt_sem) or np.isnan(dg_r206h_sem)) else float("nan")

        selective    = bool(ddg < threshold_kcal)
        within_noise = bool(abs(ddg) < abs(threshold_kcal))

        rows.append({
            "ligand_id":    lig_id,
            "dG_WT":        round(dg_wt, 2),
            "dG_WT_sem":    round(dg_wt_sem, 2) if not np.isnan(dg_wt_sem) else float("nan"),
            "dG_R206H":     round(dg_r206h, 2),
            "dG_R206H_sem": round(dg_r206h_sem, 2) if not np.isnan(dg_r206h_sem) else float("nan"),
            "ddG_bind":     round(ddg, 2),
            "ddG_sem":      round(ddg_sem, 2) if not np.isnan(ddg_sem) else float("nan"),
            "selective":    selective,
            "within_noise": within_noise,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        selective_leads = df[df["selective"]]["ligand_id"].tolist()
        noise_leads     = df[df["within_noise"]]["ligand_id"].tolist()
        logger.info(
            "Selectivity summary: %d selective (ΔΔG < %.1f): %s | %d within GBn2 noise: %s",
            len(selective_leads), threshold_kcal, selective_leads,
            len(noise_leads), noise_leads,
        )
    return df
