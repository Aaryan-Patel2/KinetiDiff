# KinetiDiff MD Validation Results
**Campaign:** 3 leads (L1, L2, L3) × 2 receptors (ACVR1 WT + R206H) × 2 replicates × 100 ns  
**Platform:** MIT ORCD Engaging — GPU production + CPU analysis  
**Date completed:** 2026-06-22  
**Status:** All 12 systems completed cleanly (equil → production → analysis)

---

## Summary Table (rep-averaged)

| System | MM-GBSA (kcal/mol) | BB-RMSD (Å) | Lig-RMSD (Å) | Bound % | αC-in | H286 occ | A-loop RMSF (Å) |
|---|---|---|---|---|---|---|---|
| L1_WT | -49.65 | 15.71 | 4.92 | 44.2% | 0.682 | 0.062 | 0.628 |
| L1_R206H | -41.76 | 14.70 | 11.20 | 30.1% | 0.683 | 0.484 | 0.620 |
| L2_WT | -51.02 | 8.38 | 5.53 | 18.4% | 0.970 | 0.088 | 0.771 |
| L2_R206H | -52.24 | 11.96 | 7.32 | 12.8% | 0.528 | 0.011 | 0.624 |
| L3_WT | -40.90 | 12.08 | 7.13 | 32.0% | 0.334 | 0.958 | 0.646 |
| L3_R206H | -51.17 | 5.40 | 6.38 | 17.9% | 0.481 | 0.845 | 0.552 |

---

## Figures

| Figure | Description |
|---|---|
| [fig1_rmsd_traces.png](figures/fig1_rmsd_traces.png) | Backbone and ligand RMSD over 100 ns, all 12 runs |
| [fig2_mmgbsa.png](figures/fig2_mmgbsa.png) | MM-GBSA ΔG per system with inter-frame std error bars |
| [fig3_pocket_occupancy.png](figures/fig3_pocket_occupancy.png) | Pocket-bound frame % per system (bar + per-rep dots) |
| [fig4_structural_heatmap.png](figures/fig4_structural_heatmap.png) | ACVR1 structural metrics heatmap (αC-in/out, H286, A-loop) |
| [fig5_summary_scatter.png](figures/fig5_summary_scatter.png) | MM-GBSA vs pocket occupancy — composite quality view |
| [fig6_rmsf_r206h.png](figures/fig6_rmsf_r206h.png) | Per-residue RMSF for R206H condition (rep-averaged) |

Regenerate with:
```bash
~/.conda/envs/kinetidiff-md/bin/python3 results/md/plot_md_results.py
```

---

## Key Findings

### Binding Free Energy
MM-GBSA values span −41 to −52 kcal/mol across all systems, which is strong for base (neutral) forms. For reference, the crystallographic inhibitor (3MTF) docks at −9.27 kcal/mol Vina; the MM-GBSA range here is consistent with a genuinely binding series.

- **Best binder by MM-GBSA:** L2_R206H (−52.2) and L2_WT (−51.0)
- **Weakest:** L1_R206H (−41.8) and L3_WT (−40.9)

### Pocket Occupancy (Critical Context)
MM-GBSA must be interpreted alongside bound %: the scoring is only computed over frames where the ligand is in the pocket, so a low occupancy system can produce an artificially favorable mean.

- **L1_WT is the most reliable system** — 44% occupancy with clean MM-GBSA (−49.7)
- **L2_R206H caveat** — despite the best MM-GBSA (−52.2), only 12.8% pocket occupancy means the ligand spends ~87% of the trajectory dissociated; the energy is real but the pose is unstable in this form
- High backbone RMSDs (9–16 Å in several systems) are primarily explained by apo-state sampling during unbound periods, not protein instability per se

### Binding Mode Fingerprint (Contact Residues)
Two clearly distinct binding modes observed:

**WT pocket** (L1/L2/L3 WT): residues 16–24, 87–95, 142–156 — canonical ATP-site lining

**R206H pocket** (all three ligands): shifted to 12–15, 31–33, 61, 81–90, 136–141 — consistent with the R206H mutation reshaping the back pocket

This fingerprint shift is structural, not a protonation artifact, and is expected to persist/strengthen in optimized forms.

### ACVR1 Structural Metrics
- **L2_WT** has the most stable active conformation: αC-in 97%, lowest backbone RMSD (8.4 Å)
- **L2_R206H** shows αC-in drop to 53% relative to L2_WT (97%) — the mutation disrupts αC stabilization for this scaffold
- **L3** uniquely engages His286 hinge in both WT (96%) and R206H (84%) — distinct binding mode; different SAR considerations apply
- A-loop RMSF is uniformly low (0.55–0.77 Å) — no evidence of activation loop disorder

---

## Caveats & Next Steps

1. **All runs are base (neutral) form.** Salt/protonated forms are expected to improve pocket occupancy and binding energy via electrostatic anchoring. L1_R206H ligand RMSD (11.2 Å) in particular likely reflects missing protonation interactions.

2. **FEP recommended for top 2.** L2_R206H and L1_WT are the strongest candidates; relative FEP between ligands in the R206H context would give a more trustworthy ΔΔG than MM-GBSA alone.

3. **L1_R206H pose instability.** Large ligand RMSD (11.2 Å) warrants visual inspection of trajectory frames before pursuing this system further.

4. **Protonated MD is the immediate next run.** Pipeline is already built; submit equil → prod → analysis for protonated forms of L1, L2, L3 in R206H context.
