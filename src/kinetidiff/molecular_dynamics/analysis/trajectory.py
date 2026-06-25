"""Trajectory analysis: RMSD, RMSF, hydrogen bonds, pocket contacts.

All functions accept file paths (not Universe objects) and return DataFrames,
keeping I/O and analysis logic separated per MD-PIPELINE.md rules.

Usage (standalone CLI):
    python analysis/trajectory.py \\
        --topology path/to/topology.pdb \\
        --trajectory path/to/run.dcd \\
        --out-dir path/to/analysis/ \\
        --lig-resname L01
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from kinetidiff._logging import get_logger

logger = get_logger(__name__)


def _find_acvr1_offset(topology) -> int:
    """Auto-detect canonical ACVR1 residue offset from the DLG motif in chain 0.

    ACVR1/ALK2 has DLG (Asp354–Leu355–Gly356) instead of the classic DFG at
    the activation loop. Returns `canonical_resSeq - sim_resSeq` for chain A.
    """
    chain_a_prot = [
        r for r in topology.residues
        if r.chain.index == 0 and r.is_protein
    ]
    for i in range(len(chain_a_prot) - 2):
        if (chain_a_prot[i].name == "ASP" and
                chain_a_prot[i + 1].name == "LEU" and
                chain_a_prot[i + 2].name == "GLY"):
            offset = 354 - chain_a_prot[i].resSeq
            logger.info(
                "ACVR1 DLG offset auto-detected: %d (DLG Asp at sim resSeq %d)",
                offset, chain_a_prot[i].resSeq,
            )
            return offset
    logger.warning("DLG motif not found in chain A — defaulting to WT offset 198.")
    return 198


def _atoms_by_res_name(topology, chain_idx: int, sim_resSeq: int, names: tuple[str, ...]) -> list[int]:
    """Return atom indices in chain_idx/sim_resSeq matching any of *names*."""
    return [
        a.index for r in topology.residues
        if r.chain.index == chain_idx and r.resSeq == sim_resSeq
        for a in r.atoms if a.name in names
    ]


def _load_traj(topology_pdb: str | Path, traj_path: str | Path, stride: int = 1):
    """Load a DCD trajectory with MDTraj. Returns (traj, None) or (None, reason_str)."""
    import mdtraj as md  # type: ignore[import]

    tr = Path(traj_path)
    tp = Path(topology_pdb)
    if not tr.exists() or tr.stat().st_size == 0:
        return None, "traj_missing"
    if not tp.exists():
        return None, "topology_missing"
    try:
        return md.load_dcd(str(tr), top=str(tp), stride=stride), None
    except Exception as exc:
        return None, f"load_error:{exc}"


def run_rmsd_analysis(
    topology_pdb: str | Path,
    traj_path: str | Path,
    lig_resname: str = "LIG",
    _traj: Optional[object] = None,
    frame_time_ps: Optional[float] = None,
) -> pd.DataFrame:
    """Compute backbone Cα and ligand RMSD over the trajectory.

    Uses MDTraj for speed and correct PBC handling:
    1. Superpose every frame onto frame 0 using protein backbone atoms.
    2. Backbone RMSD: standard RMSD of Cα atoms after superposition.
    3. Ligand RMSD: minimum-image convention applied per heavy atom after
       superposition — prevents 90 Å jumps when the ligand crosses a periodic
       boundary during the run.

    Args:
        topology_pdb: Path to the solvated-system PDB (atom ordering reference).
        traj_path: Path to the DCD trajectory.
        lig_resname: Residue name of the ligand (3-char PDB code, e.g. ``"L01"``).
        _traj: Pre-loaded MDTraj Trajectory (avoids duplicate disk reads when
               multiple analysis functions share the same trajectory).

    Returns:
        DataFrame with columns: ``time_ns``, ``backbone_rmsd_A``, ``ligand_rmsd_A``.
    """
    tr = Path(traj_path)
    tp = Path(topology_pdb)
    if not tr.exists() or not tp.exists():
        logger.warning("RMSD skipped: traj or topology missing.")
        return pd.DataFrame()

    try:
        if _traj is None:
            traj, err = _load_traj(topology_pdb, traj_path)
            if traj is None:
                logger.warning("RMSD skipped: %s", err)
                return pd.DataFrame()
        else:
            traj = _traj

        bb_idx  = traj.topology.select("backbone")
        ca_idx  = traj.topology.select("name CA")
        lig_idx = traj.topology.select(f"resname {lig_resname} and not name H")
        if len(lig_idx) == 0:
            lig_idx = traj.topology.select(
                "not protein and not water and not (resname NA CL) and not name H"
            )

        if len(bb_idx) == 0:
            logger.warning("No backbone atoms — skipping RMSD.")
            return pd.DataFrame()

        # Atom-slice to only backbone + ligand before superposition. This keeps
        # the copy cost at ~300 MB (5k atoms) vs 8.3 GB (full solvated system),
        # and avoids mutating the shared trajectory when _traj is provided.
        keep = np.unique(np.concatenate([
            bb_idx, ca_idx,
            lig_idx if len(lig_idx) > 0 else np.array([], dtype=int),
        ])).astype(int)
        work = traj.atom_slice(keep)
        old_to_new = {int(old): new for new, old in enumerate(keep)}
        bb_w  = np.array([old_to_new[i] for i in bb_idx])
        ca_w  = np.array([old_to_new[i] for i in ca_idx])
        lig_w = np.array([old_to_new[i] for i in lig_idx]) if len(lig_idx) > 0 else np.array([], dtype=int)

        work.superpose(work[0], atom_indices=bb_w)

        # Backbone (Cα) RMSD — straightforward after superposition
        ref_ca = work.xyz[0, ca_w]
        bb_rmsd_nm = np.sqrt(np.mean((work.xyz[:, ca_w] - ref_ca) ** 2, axis=(1, 2)))

        # Ligand RMSD with per-atom minimum-image convention (PBC correction)
        lig_rmsd_A: list[float] = []
        if len(lig_w) > 0:
            ref_lig = work.xyz[0, lig_w]  # (n_lig, 3) nm
            boxes   = traj.unitcell_lengths  # (n_frames, 3) nm or None
            for i in range(work.n_frames):
                diff = work.xyz[i, lig_w] - ref_lig
                if boxes is not None:
                    diff -= boxes[i] * np.round(diff / boxes[i])
                lig_rmsd_A.append(float(np.sqrt(np.mean(diff ** 2))) * 10.0)  # nm→Å

        # If the DCD header has IFREQ=0/DELTA=0 (OpenMM bug), MDTraj defaults to
        # 1 ps/frame. Override with the known frame spacing from the config.
        if frame_time_ps is not None:
            times_ns = np.arange(work.n_frames) * frame_time_ps / 1000.0
        else:
            times_ns = work.time / 1000.0  # ps → ns
        df = pd.DataFrame({
            "time_ns":         times_ns,
            "backbone_rmsd_A": bb_rmsd_nm * 10.0,  # nm → Å
        })
        if lig_rmsd_A:
            df["ligand_rmsd_A"] = lig_rmsd_A

        logger.info("RMSD: %d frames analysed.", len(df))
        return df

    except Exception as exc:
        logger.error("RMSD analysis failed: %s", exc)
        return pd.DataFrame()


def run_rmsf_analysis(
    topology_pdb: str | Path,
    traj_path: str | Path,
    _traj: Optional[object] = None,
) -> pd.DataFrame:
    """Compute per-residue RMSF for protein Cα atoms using MDTraj.

    Args:
        topology_pdb: Path to the system topology PDB.
        traj_path: Path to the DCD trajectory.
        _traj: Pre-loaded MDTraj Trajectory (avoids duplicate disk reads).

    Returns:
        DataFrame with columns: ``residue`` (resSeq), ``resname``, ``rmsf_A``.
    """
    import mdtraj as md  # type: ignore[import]

    tr = Path(traj_path)
    tp = Path(topology_pdb)
    if not tr.exists() or not tp.exists():
        logger.warning("RMSF skipped: traj or topology missing.")
        return pd.DataFrame()

    try:
        if _traj is None:
            traj, err = _load_traj(topology_pdb, traj_path)
            if traj is None:
                logger.warning("RMSF skipped: %s", err)
                return pd.DataFrame()
        else:
            traj = _traj
        ca_idx = traj.topology.select("name CA and protein")
        if len(ca_idx) == 0:
            logger.warning("No Cα protein atoms found — RMSF skipped.")
            return pd.DataFrame()

        ca_traj = traj.atom_slice(ca_idx)
        # Align to first frame, then compute per-atom RMSF (nm)
        rmsf_nm = md.rmsf(ca_traj, ca_traj, frame=0)

        residues = [traj.topology.atom(i).residue for i in ca_idx]
        logger.info("RMSF: %d Cα residues analysed.", len(residues))
        return pd.DataFrame({
            "residue": [r.resSeq for r in residues],
            "resname": [r.name for r in residues],
            "rmsf_A":  rmsf_nm * 10.0,  # nm → Å
        })

    except Exception as exc:
        logger.error("RMSF analysis failed: %s", exc)
        return pd.DataFrame()


def run_hbond_analysis(
    topology_pdb: str | Path,
    traj_path: str | Path,
    lig_resname: str = "LIG",
    min_occupancy: float = 0.10,
    _traj: Optional[object] = None,
) -> pd.DataFrame:
    """Compute protein–ligand hydrogen bond occupancies using MDTraj baker_hubbard.

    Uses a purely geometric H-bond criterion (no partial charges required):
    D–H···A with D–A distance < 3.5 Å and D–H–A angle > 120°.

    Args:
        topology_pdb: Path to the system topology PDB.
        traj_path: Path to the DCD trajectory.
        lig_resname: Residue name of the ligand.
        min_occupancy: Minimum fraction of frames for an H-bond to be reported.
        _traj: Pre-loaded MDTraj Trajectory (avoids duplicate disk reads).

    Returns:
        DataFrame with columns: ``donor_resid``, ``donor_resname``,
        ``acceptor_resid``, ``acceptor_resname``, ``occupancy_pct``.
        Sorted by descending occupancy.
    """
    import mdtraj as md  # type: ignore[import]

    tr = Path(traj_path)
    tp = Path(topology_pdb)
    if not tr.exists() or not tp.exists():
        logger.warning("H-bond skipped: traj or topology missing.")
        return pd.DataFrame()

    try:
        if _traj is None:
            traj, err = _load_traj(topology_pdb, traj_path)
            if traj is None:
                logger.warning("H-bond skipped: %s", err)
                return pd.DataFrame()
        else:
            traj = _traj

        # baker_hubbard: geometric H-bond detection — no partial charges needed
        hbonds = md.baker_hubbard(traj, periodic=True)
        # hbonds shape: (n_hbonds, 3) — columns: donor_atom, H_atom, acceptor_atom

        if len(hbonds) == 0:
            logger.info("H-bonds: no bonds detected.")
            return pd.DataFrame()

        lig_idx = set(traj.topology.select(f"resname {lig_resname}"))
        if not lig_idx:
            lig_idx = set(traj.topology.select(
                "not protein and not water and not (resname NA CL)"
            ))
        prot_idx = set(traj.topology.select("protein"))

        # baker_hubbard returns one row per H-bond instance (donor, H, acceptor)
        # across all frames; count per unique (donor, acceptor) pair
        bond_counts: dict[tuple[int, int], int] = {}
        for donor, _h, acceptor in hbonds:
            d, a = int(donor), int(acceptor)
            if (d in prot_idx and a in lig_idx) or (d in lig_idx and a in prot_idx):
                key = (d, a)
                bond_counts[key] = bond_counts.get(key, 0) + 1

        n_frames = traj.n_frames
        rows = []
        for (d_ix, a_ix), count in bond_counts.items():
            occupancy = count / n_frames
            if occupancy >= min_occupancy:
                d_atom = traj.topology.atom(d_ix)
                a_atom = traj.topology.atom(a_ix)
                rows.append({
                    "donor_resid":      d_atom.residue.resSeq,
                    "donor_resname":    d_atom.residue.name,
                    "donor_name":       d_atom.name,
                    "acceptor_resid":   a_atom.residue.resSeq,
                    "acceptor_resname": a_atom.residue.name,
                    "acceptor_name":    a_atom.name,
                    "occupancy_pct":    round(occupancy * 100, 1),
                })

        if not rows:
            logger.info("H-bonds: no bonds with ≥ %.0f%% occupancy.", min_occupancy * 100)
            return pd.DataFrame()

        df = (
            pd.DataFrame(rows)
            .sort_values("occupancy_pct", ascending=False)
            .reset_index(drop=True)
        )
        logger.info(
            "H-bonds: %d persistent bonds (≥ %.0f%% occupancy).",
            len(df), min_occupancy * 100,
        )
        return df

    except Exception as exc:
        logger.error("H-bond analysis failed: %s", exc)
        return pd.DataFrame()


def run_pocket_contacts(
    topology_pdb: str | Path,
    traj_path: str | Path,
    lig_resname: str = "LIG",
    cutoff_A: float = 4.5,
    _traj: Optional[object] = None,
) -> dict:
    """Compute pocket residue contact persistence using MDTraj compute_contacts.

    Args:
        topology_pdb: Path to the system topology PDB.
        traj_path: Path to the DCD trajectory.
        lig_resname: Residue name of the ligand.
        cutoff_A: Distance cutoff for protein–ligand heavy-atom contacts.
        _traj: Pre-loaded MDTraj Trajectory (avoids duplicate disk reads).

    Returns:
        Dict with keys: ``bound_pct`` (float, % of bound frames),
        ``contact_residues`` (list of resSeq with ≥ 20% contact occupancy).
    """
    import mdtraj as md  # type: ignore[import]

    tr = Path(traj_path)
    tp = Path(topology_pdb)
    if not tr.exists() or not tp.exists():
        logger.warning("Pocket contacts skipped: traj or topology missing.")
        return {"bound_pct": float("nan"), "contact_residues": []}

    try:
        if _traj is None:
            traj, err = _load_traj(topology_pdb, traj_path)
            if traj is None:
                logger.warning("Pocket contacts skipped: %s", err)
                return {"bound_pct": float("nan"), "contact_residues": []}
        else:
            traj = _traj
        cutoff_nm = cutoff_A / 10.0

        lig_heavy = traj.topology.select(f"resname {lig_resname} and not name H")
        if len(lig_heavy) == 0:
            lig_heavy = traj.topology.select(
                "not protein and not water and not (resname NA CL) and not name H"
            )
        if len(lig_heavy) == 0:
            return {"bound_pct": float("nan"), "contact_residues": []}

        # Identify unique residues for the ligand and protein (closest-heavy scheme)
        lig_res_set = sorted({traj.topology.atom(i).residue.index for i in lig_heavy})
        prot_res_set = sorted({
            r.index for r in traj.topology.residues
            if r.is_protein
        })

        pairs = np.array([[lr, pr] for lr in lig_res_set for pr in prot_res_set])
        # compute_contacts returns (n_frames, n_pairs) in nm, closest-heavy scheme
        contacts, _ = md.compute_contacts(
            traj, contacts=pairs, scheme="closest-heavy", periodic=True,
        )
        # contact fraction per (lig_res, prot_res) pair
        in_contact = contacts < cutoff_nm  # (n_frames, n_pairs)

        # Bound frames: ligand COM within 10 Å of its frame-0 position
        lig_com = traj.xyz[:, lig_heavy, :].mean(axis=1)  # (n_frames, 3) nm
        ref_com = lig_com[0]
        bound_frames = int(np.sum(np.linalg.norm(lig_com - ref_com, axis=1) < 1.0))
        bound_pct = (bound_frames / traj.n_frames) * 100.0

        # Per-protein-residue occupancy: max over ligand residues
        n_lig_res = len(lig_res_set)
        n_prot_res = len(prot_res_set)
        occ_reshaped = in_contact.mean(axis=0).reshape(n_lig_res, n_prot_res)
        prot_occ = occ_reshaped.max(axis=0)  # (n_prot_res,)

        contact_residues = [
            traj.topology.residue(prot_res_set[i]).resSeq
            for i in range(n_prot_res)
            if prot_occ[i] >= 0.20
        ]

        logger.info(
            "Pocket contacts: bound_pct=%.1f%%, contact_residues=%d",
            bound_pct, len(contact_residues),
        )
        return {"bound_pct": round(bound_pct, 1), "contact_residues": sorted(contact_residues)}

    except Exception as exc:
        logger.error("Pocket contact analysis failed: %s", exc)
        return {"bound_pct": float("nan"), "contact_residues": []}


def run_acvr1_structural_analysis(
    topology_pdb: str | Path,
    traj_path: str | Path,
    lig_resname: str = "LIG",
    canonical_offset: Optional[int] = None,
    _traj: Optional[object] = None,
) -> dict:
    """Compute ACVR1-specific structural monitors across the trajectory.

    Tracks three mechanistic readouts that distinguish FOP-selective compounds
    from pan-ACVR1 inhibitors:

    1. **αC-helix state** — distance between β3-Lys235 NZ and αC-Glu248 OE1/OE2
       (< 4 Å = active "in"; > 6 Å = inactive "out").
       Note: αC-Glu is canonical 248 (sim resSeq 50 for WT), not 282 (which is ILE
       in 3MTF chain A). Residue numbers derived from DLG-offset auto-detection.
    2. **Hinge H-bond occupancy** — backbone contacts between the ligand and
       His286 backbone CO and Glu287 backbone NH.
    3. **A-loop RMSF** — per-residue Cα fluctuation of the activation loop
       (DLG Asp354 through ~Glu390, ACVR1-specific DLG motif not DFG).

    The R206H mutation destabilises the GS domain and increases A-loop flexibility;
    a selective lead should show damped A-loop RMSF relative to the apo/WT run.

    Args:
        topology_pdb: Path to the solvated-system PDB.
        traj_path: Path to the DCD trajectory.
        lig_resname: Residue name of the ligand.
        canonical_offset: Offset such that sim_resSeq + offset = canonical_ACVR1_resSeq.
            If None, auto-detected from the DLG motif in chain 0.

    Returns:
        Dict with keys:
        - ``alpha_c_distances``: np.ndarray of Lys235–Glu248 distances (Å) per frame
        - ``alpha_c_in_pct``: fraction of frames with distance < 4.0 Å (active state)
        - ``alpha_c_out_pct``: fraction of frames with distance > 6.0 Å (inactive state)
        - ``hinge_his286_occupancy``: H-bond occupancy fraction at His286 backbone O
        - ``hinge_glu287_occupancy``: H-bond occupancy fraction at Glu287 backbone N
        - ``aloop_rmsf_df``: DataFrame(residue, resname, rmsf_A) for residues 354–390
        - ``aloop_mean_rmsf_A``: mean A-loop RMSF (Å) — compare WT vs R206H
        - ``reason``: "ok" or an error description
    """
    import mdtraj as md  # type: ignore[import]

    _nan = {
        "alpha_c_distances":        np.array([]),
        "alpha_c_in_pct":           float("nan"),
        "alpha_c_out_pct":          float("nan"),
        "hinge_his286_occupancy":   float("nan"),
        "hinge_glu287_occupancy":   float("nan"),
        "aloop_rmsf_df":            pd.DataFrame(),
        "aloop_mean_rmsf_A":        float("nan"),
        "reason":                   "",
    }

    tr = Path(traj_path)
    tp = Path(topology_pdb)
    if not tr.exists() or not tp.exists():
        return {**_nan, "reason": "traj_or_topology_missing"}

    try:
        if _traj is None:
            traj, err = _load_traj(topology_pdb, traj_path)
            if traj is None:
                return {**_nan, "reason": f"load_error:{err}"}
        else:
            traj = _traj

        if canonical_offset is None:
            canonical_offset = _find_acvr1_offset(traj.topology)

        # sim_resSeq = canonical - canonical_offset
        lys235_sim = 235 - canonical_offset   # β3-Lys (VAIK motif)
        glu248_sim = 248 - canonical_offset   # αC-Glu (salt bridge partner; 248 in 3MTF)
        his286_sim = 286 - canonical_offset   # hinge backbone CO acceptor
        glu287_sim = 287 - canonical_offset   # hinge backbone NH donor
        asp354_sim = 354 - canonical_offset   # DLG Asp (A-loop start)
        glu390_sim = 390 - canonical_offset   # A-loop end

        # ── 1. αC-helix state: β3-Lys235 NZ ↔ αC-Glu248 OE ──────────────────
        lys_nz = _atoms_by_res_name(traj.topology, 0, lys235_sim, ("NZ",))
        glu_oe = _atoms_by_res_name(traj.topology, 0, glu248_sim, ("OE1", "OE2"))

        ac_arr = np.array([float("nan")])
        ac_in_pct = float("nan")
        ac_out_pct = float("nan")

        if lys_nz and glu_oe:
            pairs = np.array([[nz, oe] for nz in lys_nz for oe in glu_oe])
            dists_nm = md.compute_distances(traj, pairs, periodic=True)
            min_dists_nm = dists_nm.min(axis=1)  # (n_frames,)
            ac_arr = min_dists_nm * 10.0  # nm → Å
            ac_in_pct  = float(np.mean(ac_arr < 4.0))
            ac_out_pct = float(np.mean(ac_arr > 6.0))
            logger.info(
                "αC-helix: in=%.1f%%, out=%.1f%%, mean_dist=%.2f Å",
                ac_in_pct * 100, ac_out_pct * 100, float(np.mean(ac_arr)),
            )
        else:
            logger.warning(
                "β3-Lys235 NZ or αC-Glu248 OE not found (sim resSeq %d/%d, chain 0). "
                "Check canonical offset %d and 3MTF chain A sequence.",
                lys235_sim, glu248_sim, canonical_offset,
            )

        # ── 2. Hinge H-bond occupancy (His286 CO, Glu287 NH) ─────────────────
        his286_o = _atoms_by_res_name(traj.topology, 0, his286_sim, ("O",))
        glu287_n = _atoms_by_res_name(traj.topology, 0, glu287_sim, ("N",))
        lig_ha = list(traj.topology.select(f"resname {lig_resname} and not name H"))
        if not lig_ha:
            lig_ha = list(traj.topology.select(
                "not protein and not water and not (resname NA CL) and not name H"
            ))

        his286_occ = float("nan")
        glu287_occ = float("nan")

        if his286_o and glu287_n and lig_ha:
            pairs_his = np.array([[l, h] for l in lig_ha for h in his286_o])
            pairs_glu = np.array([[l, g] for l in lig_ha for g in glu287_n])
            d_his_A = md.compute_distances(traj, pairs_his, periodic=True).min(axis=1) * 10.0
            d_glu_A = md.compute_distances(traj, pairs_glu, periodic=True).min(axis=1) * 10.0
            his286_occ = float(np.mean(d_his_A < 3.5))
            glu287_occ = float(np.mean(d_glu_A < 3.5))
            logger.info(
                "Hinge: His286_occ=%.1f%%, Glu287_occ=%.1f%%",
                his286_occ * 100, glu287_occ * 100,
            )
        else:
            logger.warning(
                "Hinge residues (His286/Glu287 at sim %d/%d) or ligand not found.",
                his286_sim, glu287_sim,
            )

        # ── 3. A-loop RMSF (DLG Asp354 – Glu390) ─────────────────────────────
        aloop_ca_idx = np.array([
            a.index for r in traj.topology.residues
            if r.chain.index == 0 and asp354_sim <= r.resSeq <= glu390_sim
            for a in r.atoms if a.name == "CA"
        ])

        aloop_rmsf_df = pd.DataFrame()
        aloop_mean = float("nan")

        if len(aloop_ca_idx) > 0:
            aloop_traj = traj.atom_slice(aloop_ca_idx)
            rmsf_nm = md.rmsf(aloop_traj, aloop_traj, frame=0)
            aloop_rmsf_df = pd.DataFrame({
                "residue": [traj.topology.atom(i).residue.resSeq for i in aloop_ca_idx],
                "resname": [traj.topology.atom(i).residue.name for i in aloop_ca_idx],
                "rmsf_A":  rmsf_nm * 10.0,
            })
            aloop_mean = float(rmsf_nm.mean() * 10.0)
            logger.info(
                "A-loop RMSF (canonical %d–%d, %d Cα): mean=%.2f Å",
                354, 390, len(aloop_ca_idx), aloop_mean,
            )
        else:
            logger.warning(
                "No Cα atoms found in A-loop region (sim resSeq %d–%d, chain 0).",
                asp354_sim, glu390_sim,
            )

        return {
            "alpha_c_distances":        ac_arr,
            "alpha_c_in_pct":           ac_in_pct,
            "alpha_c_out_pct":          ac_out_pct,
            "hinge_his286_occupancy":   his286_occ,
            "hinge_glu287_occupancy":   glu287_occ,
            "aloop_rmsf_df":            aloop_rmsf_df,
            "aloop_mean_rmsf_A":        aloop_mean,
            "reason":                   "ok",
        }

    except Exception as exc:
        logger.error("ACVR1 structural analysis failed: %s", exc)
        return {**_nan, "reason": f"error:{exc}"}


def is_pose_stable(
    rmsd_df: pd.DataFrame,
    threshold_A: float = 4.0,
    tail_fraction: float = 0.20,
) -> bool:
    """Return True if the ligand RMSD in the final *tail_fraction* is below *threshold_A*.

    Flags a pose as unstable when the ligand has drifted significantly from the
    docked pose in the equilibrated production tail (last 20% of trajectory).

    Args:
        rmsd_df: Output of ``run_rmsd_analysis``.
        threshold_A: RMSD threshold in Å (default: 4.0 per analysis config).
        tail_fraction: Fraction of trajectory frames considered the "tail".

    Returns:
        True if stable (low RMSD), False if the pose has escaped the pocket.
    """
    if "ligand_rmsd_A" not in rmsd_df.columns or rmsd_df.empty:
        logger.warning("No ligand_rmsd_A column — cannot assess stability.")
        return True  # conservative: assume stable if data missing

    n = len(rmsd_df)
    tail = rmsd_df["ligand_rmsd_A"].iloc[int(n * (1 - tail_fraction)):]
    mean_tail = float(tail.mean())
    stable = mean_tail < threshold_A
    logger.info(
        "Stability check: tail RMSD mean=%.2f Å (threshold=%.1f Å) → %s",
        mean_tail, threshold_A, "STABLE" if stable else "UNSTABLE",
    )
    return stable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory analysis CLI")
    parser.add_argument("--topology",   required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--out-dir",    required=True)
    parser.add_argument("--lig-resname", default="LIG")
    parser.add_argument("--min-hbond-occupancy", type=float, default=0.10)
    parser.add_argument(
        "--stride", type=int, default=2,
        help="Frame stride for trajectory loading (default: 2). "
             "Stride=2 halves memory vs stride=1 with negligible effect on "
             "RMSF/H-bond statistics over 100 ns.",
    )
    parser.add_argument(
        "--frame-time-ps", type=float, default=None,
        help="True time between DCD frames in ps (overrides MDTraj time axis). "
             "Use when OpenMM wrote IFREQ=0/DELTA=0 in the DCD header, causing "
             "MDTraj to default to 1 ps/frame. "
             "Formula: report_interval × timestep_ps  (e.g. 5000 × 0.002 = 10.0).",
    )
    parser.add_argument(
        "--receptor", default=None,
        help="Receptor label (e.g. WT or R206H) — for logging only; "
             "canonical offset is auto-detected from the DLG motif.",
    )
    args = parser.parse_args()

    if args.receptor:
        logger.info("Receptor: %s", args.receptor)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load trajectory once — avoids 5 separate disk reads and prevents the
    # Python allocator from accumulating peak RSS equal to 2× the trajectory size.
    import mdtraj as md  # type: ignore[import]
    logger.info("Loading trajectory %s (stride=%d) …", args.trajectory, args.stride)
    shared_traj, load_err = _load_traj(args.topology, args.trajectory, stride=args.stride)
    if shared_traj is None:
        logger.error("Cannot load trajectory: %s — aborting.", load_err)
        raise SystemExit(1)
    logger.info("Trajectory loaded: %d frames, %d atoms.", shared_traj.n_frames, shared_traj.n_atoms)

    rmsd_df  = run_rmsd_analysis(
        args.topology, args.trajectory, args.lig_resname,
        _traj=shared_traj, frame_time_ps=args.frame_time_ps,
    )
    rmsf_df  = run_rmsf_analysis(args.topology, args.trajectory, _traj=shared_traj)
    hbond_df = run_hbond_analysis(
        args.topology, args.trajectory, args.lig_resname, args.min_hbond_occupancy,
        _traj=shared_traj,
    )
    pocket   = run_pocket_contacts(args.topology, args.trajectory, args.lig_resname, _traj=shared_traj)
    acvr1    = run_acvr1_structural_analysis(
        args.topology, args.trajectory, args.lig_resname, _traj=shared_traj
    )

    if not rmsd_df.empty:
        rmsd_df.to_csv(out / "rmsd.csv", index=False)
    if not rmsf_df.empty:
        rmsf_df.to_csv(out / "rmsf.csv", index=False)
    if not hbond_df.empty:
        hbond_df.to_csv(out / "hbonds.csv", index=False)
    if not acvr1["aloop_rmsf_df"].empty:
        acvr1["aloop_rmsf_df"].to_csv(out / "aloop_rmsf.csv", index=False)

    import json
    (out / "pocket_contacts.json").write_text(json.dumps(pocket, indent=2))

    acvr1_summary = {
        "alpha_c_in_pct":          acvr1["alpha_c_in_pct"],
        "alpha_c_out_pct":         acvr1["alpha_c_out_pct"],
        "hinge_his286_occupancy":  acvr1["hinge_his286_occupancy"],
        "hinge_glu287_occupancy":  acvr1["hinge_glu287_occupancy"],
        "aloop_mean_rmsf_A":       acvr1["aloop_mean_rmsf_A"],
        "reason":                  acvr1["reason"],
    }
    (out / "acvr1_structural.json").write_text(json.dumps(acvr1_summary, indent=2))
    print("Analysis complete. Outputs written to", out)
