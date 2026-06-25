"""Download 3MTF, strip A3F to produce the apo pocket, apply WT/R206H prep.

Key design contract:
- ``prep_protein`` always calls ``removeHeterogens(keepWater=False)`` first,
  which removes the crystallographic inhibitor A3F along with all other HETATM
  records.  The result is the *apo* ACVR1 pocket, suitable for ligand docking
  poses in MD.
- R206H mutation is applied via PDBFixer's ``applyMutations`` API.
- ``validate_pocket_centroid`` sanity-checks that the prepared protein's
  pocket centroid (CA atoms of the 10 closest residues to the canonical
  binding-site center) lies within the configured tolerance.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from kinetidiff._logging import get_logger

logger = get_logger(__name__)

# Canonical binding-site centroid for ACVR1/3MTF (Å).
# Defined in analysis/diagnostic_tests.py:32 and docking/run_vina_test.py:44.
# Do not redefine elsewhere — import from here.
CANONICAL_POCKET_CENTER: tuple[float, float, float] = (-17.66, -13.65, 38.41)


_MAX_INTERNAL_GAP = 20  # residues; gaps longer than this are terminal truncations


def _clear_terminal_missing_residues(fixer: Any, max_gap: int = _MAX_INTERNAL_GAP) -> None:  # noqa: ANN401
    """Remove only long (terminal) missing-residue gaps from a PDBFixer object.

    Why:
        PDBFixer's ``missingResidues = {}`` drops ALL gaps including internal
        GS-domain loops that contain R206.  This helper removes only segments
        with more than *max_gap* residues, which are invariably N/C-terminal
        truncations (ACVR1/3MTF's N-terminal tail is ~180 residues; GS-domain
        loop gaps are 1–8 residues at most).  Internal gaps are kept so that
        ``addMissingAtoms`` can model them.
    """
    keys_to_remove = []
    for key, residue_names in fixer.missingResidues.items():
        if len(residue_names) > max_gap:
            logger.info(
                "Clearing long missing-residue segment at key %s (%d residues) — "
                "treating as terminal truncation.",
                key, len(residue_names),
            )
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del fixer.missingResidues[key]

    n_remaining = len(fixer.missingResidues)
    if n_remaining:
        logger.info(
            "%d internal missing-residue segment(s) kept for PDBFixer to model "
            "(all ≤ %d residues).",
            n_remaining, max_gap,
        )


def download_pdb(pdb_id: str, out_dir: Path) -> Path:
    """Download a PDB entry from RCSB and return the local path.

    Uses Biopython's PDBList; skips download if the file already exists.

    Args:
        pdb_id: Four-character PDB accession code (e.g. ``"3MTF"``).
        out_dir: Directory in which to save the downloaded file.

    Returns:
        Path to the downloaded PDB file.
    """
    from Bio.PDB import PDBList  # type: ignore[import]

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{pdb_id}_raw.pdb"
    if dst.exists():
        logger.info("PDB %s already present, skipping download.", pdb_id)
        return dst

    pdbl = PDBList()
    pdbl.retrieve_pdb_file(pdb_id, pdir=str(out_dir), file_format="pdb")

    # Biopython writes pdb<id>.ent or <id>.pdb depending on version.
    candidates = (
        list(out_dir.glob(f"pdb{pdb_id.lower()}.ent"))
        + list(out_dir.glob(f"{pdb_id.lower()}.pdb"))
        + list(out_dir.glob(f"{pdb_id}.pdb"))
    )
    if not candidates:
        raise FileNotFoundError(f"Downloaded PDB file not found for {pdb_id} in {out_dir}")

    shutil.copy(candidates[0], dst)
    logger.info("Downloaded %s → %s", pdb_id, dst)
    return dst


def prep_protein(
    raw_pdb_path: Path,
    receptor_type: str,
    out_dir: Path,
    ph: float = 7.4,
) -> Path:
    """Prepare the ACVR1 apo protein for MD (strips heterogens, adds missing atoms).

    The inhibitor A3F (and any other HETATM) is removed by
    ``removeHeterogens(keepWater=False)`` before any other processing.
    For ``receptor_type == "R206H"`` the ARG206→HIS mutation is applied.
    Hydrogens are *not* added here — OpenMM's ``ForceField`` adds them during
    system building to avoid template-mismatch errors.

    Args:
        raw_pdb_path: Path to the raw downloaded PDB file.
        receptor_type: Either ``"WT"`` or ``"R206H"``.
        out_dir: Directory to write the prepped PDB.
        ph: Protonation pH (used by PDBFixer's hydrogen-addition, deferred here).

    Returns:
        Path to the prepped PDB file.
    """
    try:
        from pdbfixer import PDBFixer  # type: ignore[import]
        from openmm.app import PDBFile  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pdbfixer and openmm must be installed in the kinetidiff-md conda env."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ACVR1_{receptor_type}_prepped.pdb"

    if out_path.exists():
        logger.info("[%s] Prepped PDB already exists at %s, skipping.", receptor_type, out_path)
        return out_path

    logger.info("[%s] Preparing protein from %s", receptor_type, raw_pdb_path)
    fixer = PDBFixer(filename=str(raw_pdb_path))

    # ── Step 1: remove all heterogens including A3F ───────────────────────────
    fixer.removeHeterogens(keepWater=False)
    logger.info("[%s] Removed heterogens (A3F stripped). Apo pocket ready.", receptor_type)

    # ── Step 2: apply R206H mutation if required ─────────────────────────────
    if receptor_type == "R206H":
        logger.info("[%s] Applying ARG206→HIS mutation...", receptor_type)
        mutation_applied = False
        for mut_spec, chain_kwarg in [
            (["ARG-206-HIS"], {"chain_id": "A"}),
            (["ARG-206-HIS"], {"chainid": "A"}),
            (["R206H"], {"chain_id": "A"}),
        ]:
            try:
                fixer.applyMutations(mut_spec, **chain_kwarg)
                mutation_applied = True
                logger.info("[%s] Mutation applied successfully.", receptor_type)
                break
            except TypeError:
                continue
            except Exception as exc:
                logger.warning("[%s] Mutation attempt failed: %s", receptor_type, exc)

        if not mutation_applied:
            logger.error(
                "[%s] Could not apply ARG206→HIS mutation via PDBFixer API. "
                "The structure will be WT. Manual inspection required.",
                receptor_type,
            )

    # ── Step 2b: assert R206 is present before attempting mutation ───────────
    # 3MTF chain A begins at ~residue 200; if it starts after 207 the GS domain
    # is entirely absent and the R206H mutation would silently fail.
    res_ids = {int(r.id) for r in fixer.topology.residues() if r.chain.id == "A"}
    if not res_ids:
        res_ids = {int(r.id) for r in fixer.topology.residues()}
    if 206 not in res_ids:
        raise ValueError(
            f"Residue 206 not found in chain A of {raw_pdb_path}. "
            "The downloaded structure does not cover the GS domain. "
            "Check the PDB file and residue numbering."
        )
    logger.info("[%s] GS domain check passed: residue 206 present in structure.", receptor_type)

    # ── Step 3: fill missing residues and atoms ───────────────────────────────
    # Clear ONLY terminal truncation overhangs — preserve internal missing
    # residues (including any GS-domain loops) so PDBFixer can model them.
    fixer.findMissingResidues()
    _clear_terminal_missing_residues(fixer)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    logger.info("[%s] Missing atoms added.", receptor_type)

    # ── Step 4: write output ─────────────────────────────────────────────────
    with open(out_path, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)

    logger.info("[%s] Prepped PDB written: %s", receptor_type, out_path)
    return out_path


def validate_pocket_centroid(
    prepped_pdb_path: Path,
    tolerance_A: float = 3.0,
    pocket_center: tuple[float, float, float] = CANONICAL_POCKET_CENTER,
    n_pocket_residues: int = 10,
) -> bool:
    """Assert that the pocket CA centroid is within *tolerance_A* of the canonical center.

    Selects the *n_pocket_residues* protein CA atoms closest to *pocket_center*,
    computes their geometric centroid, and returns True if the distance is within
    *tolerance_A* angstroms.  Raises ``ValueError`` on failure in strict mode.

    Args:
        prepped_pdb_path: Path to the prepped protein PDB (apo, from ``prep_protein``).
        tolerance_A: Maximum allowed centroid deviation in Å.
        pocket_center: Canonical binding-site centroid; defaults to the 3MTF value.
        n_pocket_residues: Number of nearest CA residues to define the pocket.

    Returns:
        True if the centroid is within tolerance; raises ``ValueError`` otherwise.
    """
    try:
        import MDAnalysis as mda  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("mdanalysis must be installed in kinetidiff-md.") from exc

    center = np.array(pocket_center)
    u = mda.Universe(str(prepped_pdb_path))
    ca_atoms = u.select_atoms("protein and name CA")
    if len(ca_atoms) == 0:
        raise ValueError(f"No CA atoms found in {prepped_pdb_path}")

    dists = np.linalg.norm(ca_atoms.positions - center, axis=1)
    pocket_ca = ca_atoms[np.argsort(dists)[:n_pocket_residues]]
    computed_centroid = pocket_ca.positions.mean(axis=0)
    deviation = float(np.linalg.norm(computed_centroid - center))

    logger.info(
        "Pocket centroid check: computed=(%.2f, %.2f, %.2f), "
        "canonical=(%.2f, %.2f, %.2f), deviation=%.2f Å (tolerance=%.1f Å)",
        *computed_centroid,
        *center,
        deviation,
        tolerance_A,
    )

    if deviation > tolerance_A:
        raise ValueError(
            f"Pocket centroid deviation {deviation:.2f} Å exceeds tolerance {tolerance_A:.1f} Å. "
            f"Check that the correct PDB (3MTF) was used and that heterogens were stripped."
        )

    logger.info("Pocket centroid validation passed (deviation=%.2f Å).", deviation)
    return True
