"""Strip solvent + stride trajectories to produce the Drive-upload bundle.

Why:
    Full solvated DCDs (~21 k atoms) accumulate quickly: 30 runs × 100 ns ×
    10 ps/frame × ~120 B/atom ≈ 30–50 GB.  After stripping water/ions and
    striding to every 10th frame we get ≈ 3–5 GB of analysis-ready trajectories
    suitable for manual Drive upload (< 100 GB total bundle).

    Full solvated trajectories stay on ORCD Pool for archiving.
"""

from __future__ import annotations

from pathlib import Path

from kinetidiff._logging import get_logger

logger = get_logger(__name__)


def strip_and_stride(
    topology_pdb: str | Path,
    traj_path: str | Path,
    out_dir: Path,
    stride: int = 10,
    lig_resname: str = "LIG",
) -> Path:
    """Write a water-stripped, strided trajectory for the Drive bundle.

    Keeps protein + ligand heavy atoms only (no water, no ions).
    Writes a companion stripped PDB for the new atom ordering.

    Args:
        topology_pdb: Original solvated topology PDB.
        traj_path: Full solvated DCD trajectory.
        out_dir: Directory to write stripped outputs.
        stride: Keep every *stride*-th frame (default: 10, ~100 ps resolution).
        lig_resname: Residue name of the ligand.

    Returns:
        Path to the stripped DCD.
    """
    try:
        import mdtraj as md  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("mdtraj must be installed in kinetidiff-md.") from exc

    tp = Path(topology_pdb)
    tr = Path(traj_path)

    if not tr.exists() or tr.stat().st_size == 0:
        logger.warning("Skipping strip_and_stride — trajectory missing or empty: %s", tr)
        return tr

    out_dir.mkdir(parents=True, exist_ok=True)
    stem        = tr.stem
    out_dcd     = out_dir / f"{stem}_stripped.dcd"
    out_top_pdb = out_dir / f"{stem}_stripped.pdb"

    if out_dcd.exists():
        logger.info("[%s] Stripped trajectory already exists; skipping.", stem)
        return out_dcd

    logger.info("[%s] Loading full trajectory...", stem)
    try:
        traj = md.load_dcd(str(tr), top=str(tp))
    except Exception as exc:
        logger.error("[%s] Could not load trajectory: %s", stem, exc)
        return tr

    # Select protein + ligand atoms
    sel = traj.topology.select(
        f"protein or resname {lig_resname} or resn {lig_resname}"
    )
    if len(sel) == 0:
        sel = traj.topology.select("protein")
        logger.warning("[%s] Ligand resname '%s' not found; keeping protein only.", stem, lig_resname)

    stripped = traj.atom_slice(sel)[::stride]
    logger.info(
        "[%s] Stripped: %d atoms, %d frames (stride=%d).",
        stem, len(sel), stripped.n_frames, stride,
    )

    stripped.save_dcd(str(out_dcd))
    stripped[0].save_pdb(str(out_top_pdb))

    logger.info("[%s] Stripped trajectory written: %s", stem, out_dcd)
    return out_dcd


def make_drive_bundle_manifest(
    bundle_dir: Path,
    analysis_dir: Path,
    figures_dir: Path,
    master_csv: Path,
) -> Path:
    """Write a JSON manifest of all files in the Drive bundle.

    Args:
        bundle_dir: Root of the bundle to be tar-ed and uploaded to Drive.
        analysis_dir: Directory of stripped DCDs.
        figures_dir: Directory of PNG/SVG figures.
        master_csv: Master results CSV.

    Returns:
        Path to the written manifest JSON.
    """
    import json
    from datetime import datetime

    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "stripped_trajectories": sorted(
            [str(p) for p in analysis_dir.glob("*_stripped.dcd")]
        ),
        "figures": sorted(
            [str(p) for p in figures_dir.glob("*.png")]
            + [str(p) for p in figures_dir.glob("*.svg")]
        ),
        "csvs": sorted([str(p) for p in bundle_dir.rglob("*.csv")]),
        "master_results": str(master_csv) if master_csv.exists() else None,
    }

    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Bundle manifest written: %s", manifest_path)
    return manifest_path
