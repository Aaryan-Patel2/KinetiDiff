#!/usr/bin/env python3
"""Compile all results into paper-ready tables.

Outputs:
  results/paper_tables/table1_top_molecules.csv   — top leads with ADMET
  results/paper_tables/table2_clinical_benchmark.csv
  results/paper_tables/admet_summary.txt          — quick narrative for methods
"""
from __future__ import annotations
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ── Load SwissADME ────────────────────────────────────────────────────────────
def load_swissadme(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))

# ── Load top-20 docking (Vina scores + QED/SA) ───────────────────────────────
def load_top20(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # filter out disconnected fragments (dot in SMILES)
    clean = [r for r in rows if "." not in r["smiles"]]
    clean.sort(key=lambda r: float(r["docking_score"] if "docking_score" in r else r.get("vina_affinity", 0)))
    return clean

# ── Load clinical benchmark ───────────────────────────────────────────────────
def load_clinical(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))

# ── Molecule labels ───────────────────────────────────────────────────────────
# pkCSM Absorption data — extracted from screenshots (L1, L2, L3 only)
# Screenshots in results/ADMET_results/pkCSM/ cover Absorption panel only.
# Missing: Distribution, Metabolism, Excretion, Toxicity (hERG, CYP, LD50).
PKCSM = {
    "L1": {"water_sol_logmol": -2.725, "caco2_logpapp": 0.929,
           "HIA_pct": 93.111, "pgp_substrate": "Yes",
           "pgp1_inhibitor": "No", "pgp2_inhibitor": "No"},
    "L2": {"water_sol_logmol": -3.775, "caco2_logpapp": 0.981,
           "HIA_pct": 92.37, "pgp_substrate": "Yes",
           "pgp1_inhibitor": "Yes", "pgp2_inhibitor": "Yes"},
    "L3": {"water_sol_logmol": -3.966, "caco2_logpapp": 0.998,
           "HIA_pct": 95.493, "pgp_substrate": "Yes",
           "pgp1_inhibitor": "Yes", "pgp2_inhibitor": "Yes"},
}

LEAD_SMILES = {
    "L1": "O=C(NC1CCNCC1c1ccc[nH]1)c1c(C2CCCOC2)ccc2ccccc12",
    "L2": "O=S(=O)(Cc1cccnc1)Nc1ccc(F)cc1-c1ccc2ccccc2c1C1CCNCC1",
    "L3": "NC(=O)NC(=O)CC1CNCCC1c1cccc2ccccc12",
}

# Brenk alert details — extracted from SwissADME paste (2026-06-18).
# CSV only stores count; names are for paper narrative.
BRENK_DETAILS = {
    "T4":  "het-C-het_not_in_ring, triple_bond",
    "T6":  "het-C-het_not_in_ring, isolated_alkene",
    "T7":  "het-C-het_not_in_ring, isolated_alkene",
    "T8":  "Three-membered_heterocycle",
    "T11": "imine_1, sulphur_nitrogen_single_bond",
    "T13": "isolated_alkene",
}

def canonical(smi: str) -> str:
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m else smi
    except Exception:
        return smi

def main() -> None:
    out_dir = REPO / "results/paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # SwissADME rows map to molecule order (1-indexed = L1, L2, L3, then clean top-scorers)
    sw_path = REPO / "results/ADMET_results/SwissADME/swissadme.csv"
    sw_rows = load_swissadme(sw_path)

    # Assign labels: rows 0-2 = L1-L3, rows 3+ = T1, T2, ...
    labels = ["L1", "L2", "L3"] + [f"T{i}" for i in range(1, len(sw_rows) - 2)]
    lead_vina = {"L1": -10.63, "L2": -10.62, "L3": -10.59}

    # Top-20 Vina for non-lead molecules (clean set, sorted by score)
    top20_path = REPO / "results/redocking/top_20_redocked.csv"
    top20 = load_top20(top20_path)
    top20_vina = [float(r.get("docking_score", r.get("vina_affinity", 0))) for r in top20]

    # ── Table 1: Top molecules with ADMET ─────────────────────────────────────
    t1_cols = ["ID", "MW", "LogP", "TPSA", "HBD", "HBA", "Rot_bonds",
               "GI_absorption", "BBB", "Pgp_substrate", "Lipinski_viol",
               "PAINS", "Brenk", "SA_score", "Bioavailability", "Vina_WT",
               "pkCSM_HIA_pct", "pkCSM_Caco2_logPapp", "pkCSM_WaterSol_logmol",
               "pkCSM_Pgp1_inhibitor", "pkCSM_Pgp2_inhibitor"]
    t1_rows = []
    for i, (label, sw) in enumerate(zip(labels, sw_rows)):
        vina = lead_vina.get(label, top20_vina[i - 3] if i >= 3 and (i - 3) < len(top20_vina) else "")
        pk = PKCSM.get(label, {})
        t1_rows.append({
            "ID": label,
            "MW": sw["MW"],
            "LogP": sw["Consensus Log P"],
            "TPSA": sw["TPSA"],
            "HBD": sw["#H-bond donors"],
            "HBA": sw["#H-bond acceptors"],
            "Rot_bonds": sw["#Rotatable bonds"],
            "GI_absorption": sw["GI absorption"],
            "BBB": sw["BBB permeant"],
            "Pgp_substrate": sw["Pgp substrate"],
            "Lipinski_viol": sw["Lipinski #violations"],
            "PAINS": sw["PAINS #alerts"],
            "Brenk": sw["Brenk #alerts"],
            "SA_score": sw["Synthetic Accessibility"],
            "Bioavailability": sw["Bioavailability Score"],
            "Vina_WT": vina,
            "pkCSM_HIA_pct": pk.get("HIA_pct", ""),
            "pkCSM_Caco2_logPapp": pk.get("caco2_logpapp", ""),
            "pkCSM_WaterSol_logmol": pk.get("water_sol_logmol", ""),
            "pkCSM_Pgp1_inhibitor": pk.get("pgp1_inhibitor", ""),
            "pkCSM_Pgp2_inhibitor": pk.get("pgp2_inhibitor", ""),
        })

    t1_path = out_dir / "table1_top_molecules.csv"
    with open(t1_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=t1_cols)
        w.writeheader(); w.writerows(t1_rows)
    print(f"Table 1 written: {t1_path} ({len(t1_rows)} molecules)")

    # ── Table 2: Clinical benchmark ───────────────────────────────────────────
    clin_path = REPO / "results/clinical_benchmark/clinical_docking_scores.csv"
    clin_rows = load_clinical(clin_path)

    t2_path = out_dir / "table2_clinical_benchmark.csv"
    with open(t2_path, "w", newline="") as f:
        fieldnames = ["compound", "WT_vina", "R206H_vina", "delta_delta_G", "notes", "error"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(clin_rows)
    print(f"Table 2 written: {t2_path}")

    # ── ADMET narrative summary ───────────────────────────────────────────────
    n_pass_lipinski = sum(1 for r in t1_rows if r["Lipinski_viol"] == "0")
    n_pains_clean   = sum(1 for r in t1_rows if r["PAINS"] == "0")
    n_high_gi       = sum(1 for r in t1_rows if r["GI_absorption"] == "High")
    n_bbb           = sum(1 for r in t1_rows if r["BBB"] == "Yes")
    leads           = t1_rows[:3]

    n_brenk_clean = sum(1 for r in t1_rows if r["Brenk"] == "0")
    brenk_flagged = [(r["ID"], BRENK_DETAILS[r["ID"]]) for r in t1_rows if r["ID"] in BRENK_DETAILS]

    summary = f"""ADMET Summary (SwissADME, n={len(t1_rows)} molecules)
{"="*60}

LEADS (L1-L3):
  L1: MW={leads[0]['MW']} LogP={leads[0]['LogP']} TPSA={leads[0]['TPSA']} GI={leads[0]['GI_absorption']} BBB={leads[0]['BBB']} Pgp={leads[0]['Pgp_substrate']} SA={leads[0]['SA_score']}
  L2: MW={leads[1]['MW']} LogP={leads[1]['LogP']} TPSA={leads[1]['TPSA']} GI={leads[1]['GI_absorption']} BBB={leads[1]['BBB']} Pgp={leads[1]['Pgp_substrate']} SA={leads[1]['SA_score']}
  L3: MW={leads[2]['MW']} LogP={leads[2]['LogP']} TPSA={leads[2]['TPSA']} GI={leads[2]['GI_absorption']} BBB={leads[2]['BBB']} Pgp={leads[2]['Pgp_substrate']} SA={leads[2]['SA_score']}

POPULATION ({len(t1_rows)} molecules):
  Lipinski compliant (0 violations): {n_pass_lipinski}/{len(t1_rows)} ({100*n_pass_lipinski//len(t1_rows)}%)
  PAINS-clean:  {n_pains_clean}/{len(t1_rows)} ({100*n_pains_clean//len(t1_rows)}%)
  Brenk-clean:  {n_brenk_clean}/{len(t1_rows)} ({100*n_brenk_clean//len(t1_rows)}%)
  High GI absorption: {n_high_gi}/{len(t1_rows)} ({100*n_high_gi//len(t1_rows)}%)
  BBB permeable: {n_bbb}/{len(t1_rows)} ({100*n_bbb//len(t1_rows)}%)
  Bioavailability Score: all = 0.55 (Lipinski-space oral bioavailability)

BRENK-FLAGGED MOLECULES (deprioritize for synthesis):
""" + "".join(f"  {mid}: {alerts}\n" for mid, alerts in brenk_flagged) + f"""
CLINICAL BENCHMARK (Vina kcal/mol, WT receptor):
  LDN-193189:    WT={clin_rows[0]['WT_vina']}  R206H={clin_rows[0]['R206H_vina']}  ΔΔG={float(clin_rows[0]['delta_delta_G']):.3f}
  Saracatinib:   WT={clin_rows[1]['WT_vina']}   R206H={clin_rows[1]['R206H_vina']}   ΔΔG={float(clin_rows[1]['delta_delta_G']):.3f}
  Zilurgisertib: WT={clin_rows[2]['WT_vina']}  R206H={clin_rows[2]['R206H_vina']}  ΔΔG={float(clin_rows[2]['delta_delta_G']):.3f}
  LDN-212854:    WT={clin_rows[3]['WT_vina']}   R206H={clin_rows[3]['R206H_vina']}   ΔΔG={float(clin_rows[3]['delta_delta_G']):.3f}

  KinetiDiff leads vs clinical benchmark (WT Vina):
    L1 (-10.63) vs LDN-193189 (-10.79): Δ = -0.16 kcal/mol  [within noise; LDN-193189 wins]
    L1 (-10.63) vs Saracatinib (-10.06): Δ = +0.57 kcal/mol
    L1 (-10.63) vs Zilurgisertib (-9.43): Δ = +1.20 kcal/mol
    L1 (-10.63) vs LDN-212854 (-10.03): Δ = +0.60 kcal/mol

pkCSM ABSORPTION (L1/L2/L3 only — screenshots extracted):
  L1: HIA={PKCSM['L1']['HIA_pct']}%  Caco2={PKCSM['L1']['caco2_logpapp']}  WaterSol={PKCSM['L1']['water_sol_logmol']}  Pgp-I-inh={PKCSM['L1']['pgp1_inhibitor']}  Pgp-II-inh={PKCSM['L1']['pgp2_inhibitor']}
  L2: HIA={PKCSM['L2']['HIA_pct']}%  Caco2={PKCSM['L2']['caco2_logpapp']}  WaterSol={PKCSM['L2']['water_sol_logmol']}  Pgp-I-inh={PKCSM['L2']['pgp1_inhibitor']}  Pgp-II-inh={PKCSM['L2']['pgp2_inhibitor']}
  L3: HIA={PKCSM['L3']['HIA_pct']}%  Caco2={PKCSM['L3']['caco2_logpapp']}  WaterSol={PKCSM['L3']['water_sol_logmol']}  Pgp-I-inh={PKCSM['L3']['pgp1_inhibitor']}  Pgp-II-inh={PKCSM['L3']['pgp2_inhibitor']}

NOTE: SwissADME CYP inhibitor fields unavailable (server-side n/d on run date 2026-06-18).
      pkCSM Distribution/Metabolism/Excretion/Toxicity panels still pending for all 16 molecules.
      pkCSM Absorption run for L1/L2/L3 only (screenshots in results/ADMET_results/pkCSM/).
"""
    summary_path = out_dir / "admet_summary.txt"
    summary_path.write_text(summary)
    print(f"\n{summary}")

if __name__ == "__main__":
    main()
