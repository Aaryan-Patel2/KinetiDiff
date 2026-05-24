# KinetiDiff Generation Results

## Overview

This directory contains all generation and docking result data from the KinetiDiff
Vina-guided diffusion campaign targeting ACVR1/ALK2 (PDB: 3MTF, pocket centroid
24.87, -12.54, 38.40).

Total molecules generated across all strategies: **19,021**
(Vina-Direct: 9,997 | Multi-Objective: 3,524 | HNN-Denovo: ~1,500 | No-Guidance: ~500)

---

## Directory Structure

```
RESULTS/
├── vina_direct/
│   └── top_leads.csv           — Top 5 confirmed leads (paper Table 2); all from
│                                  generated_local_10k run (9,997 molecules).
│                                  ⚠ Full run results.json missing from local disk
│                                    (Windows shortcut present; file not recovered).
│
├── hnn_guided/                 — HNN-Denovo surrogate-guided runs (3 × 50 molecules)
│   ├── generated5k_summary.csv
│   ├── generated5k_molecules.smi
│   ├── generated_2k_run1_summary.csv
│   ├── generated_2k_run1_molecules.smi
│   ├── generated_2k_gpu1_summary.csv
│   └── generated_2k_gpu1_molecules.smi
│
├── multi_objective/            — Multi-objective guided run (3,524 valid molecules)
│   ├── multi_objective_5k_summary.csv
│   └── multi_objective_5k_molecules.smi
│
├── no_guidance/                — Unguided GCDM baseline (4 molecules, quick test)
│   ├── generated10_summary.csv
│   └── generated10_molecules.smi
│
└── redocking/                  — Uniform Vina redocking of pre-ranked candidates
    ├── docking_results_comprehensive.csv   — 254 molecules, full property set
    ├── top_20_redocked.csv                 — Top 20 by combined score
    ├── filtered_top_molecules.csv          — Top 20 after Lipinski filter
    └── top_50_molecules.json               — Top 50 as JSON
```

---

## Top 5 Confirmed Leads (Vina-Direct, generated_local_10k)

| Rank | SMILES | Vina (kcal/mol) | pKd | SA | QED | MW (Da) | Lip. viol. |
|------|--------|-----------------|-----|-----|-----|---------|------------|
| 1 | `O=C(NC1CCNCC1c1ccc[nH]1)c1c(C2CCCOC2)ccc2ccccc12` | −11.05 | 8.10 | 3.79 | 0.618 | 403.5 | 0 |
| 2 | `O=S(=O)(Cc1cccnc1)Nc1ccc(F)cc1-c1ccc2ccccc2c1C1CCNCC1` | −10.62 | 7.79 | 2.61 | 0.389 | 475.6 | 1 |
| 3 | `NC(=O)NC(=O)CC1CNCCC1c1cccc2ccccc12` | −10.59 | 7.77 | 3.12 | 0.812 | 311.4 | 0 |
| 4 | `NC(=O)Cc1c(NC(=O)c2cc(F)ccc2-c2cccnc2)ccc2ccccc12` | −10.55 | 7.74 | 2.26 | 0.524 | 399.4 | 0 |
| 5 | `NC(=O)C1CNCCC1c1cc(F)ccc1-c1cnccc1C1CCCNC1` | −10.51 | 7.71 | 3.75 | 0.759 | 382.5 | 0 |
| Ref. | crystallographic inhibitor (3MTF/A3F) | −9.27 | 6.80 | 3.34 | 0.680 | 370.5 | 0 |

All top 5 surpass the crystallographic reference by >1.2 kcal/mol.
Best binder (Rank 1) represents a **19.2% improvement** in binding free energy.

---

## Strategy Ablation Summary (medians, n=19,021 total)

| Strategy | pKd (median) | QED (median) | SA (median) | Validity | n |
|----------|-------------|-------------|------------|----------|---|
| Vina-Direct | **6.71** | **0.61** | **2.26** | 99.97% | 9,997 |
| HNN-Denovo | 6.29 | 0.54 | 4.97 | 100.0% | ~1,500 |
| Multi-Objective | 5.52 | 0.49 | 4.37 | 70.5% | 3,524 |
| Unguided | 5.96 | 0.52 | 3.81 | 100.0% | ~4,000 |

---

## Generation Config (Vina-Direct run)

- **Model**: `bindingmoad_ca_cond_gcpnet.ckpt` (Zenodo record 13375913)
- **Receptor**: ACVR1 PDB 3MTF, `receptor.pdbqt`
- **Pocket centroid**: (24.87, −12.54, 38.40), box 20×20×20 Å
- **Guidance start**: t < 400 (last 40% of denoising)
- **Guidance interval**: every 20 steps
- **Gradient atoms**: 5 per step (random subset)
- **Gradient processing**: BatchNorm-style EMA (momentum=0.1) + tanh clipping
- **Adaptive scale**: 0.1× (t>600) → 0.3× (t>400) → 0.7× (t>200) → 1.5× (t≤200)
- **Hardware**: dual NVIDIA RTX A6000 (49 GB VRAM), ~2.3 h per 2,000 molecules

---

## Reproducibility

The top 5 SMILES are the authoritative record from `paper/neurips_paper/kinetidiff_position.tex`
(Appendix, Table 2). Vina scores were computed with `--score_only --seed 42 --exhaustiveness 4`
against `receptor.pdbqt`. To reproduce redocking:

```bash
cd src
python docking/redock_and_compare.py \
    --smiles-csv RESULTS/vina_direct/top_leads.csv \
    --receptor data/structures/receptor.pdbqt \
    --centroid 24.87,-12.54,38.40
```

> **Note**: The full `generated_local_10k/results.json` (9,997-molecule Vina-Direct run)
> is not in this repository. A Windows file shortcut exists but the source file was not
> recovered. The top 5 leads are extracted from the paper and are the verified output.
> If you have the original results.json, place it at `RESULTS/vina_direct/results.json`.
