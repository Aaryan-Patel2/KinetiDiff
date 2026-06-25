"""KinetiDiff molecular dynamics validation sub-package.

Provides protein/ligand/system preparation, OpenMM-based equilibration and
production MD, MDAnalysis-based trajectory analysis, GBn2 MM-GBSA rescoring,
WT-vs-R206H selectivity computation, and publication-quality figure generation.

Intended execution target: MIT ORCD Engaging cluster (SLURM job array of 30).
Thin orchestration notebook lives at ``notebooks/md_validation.ipynb``.
"""
