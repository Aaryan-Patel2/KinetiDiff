"""
Pocket processing utilities for GCDM-Modified.

Provides functions for:
- Loading pocket information from PDB files
- Converting between pocket representations
- Identifying pocket residues from various inputs
"""

import os

import numpy as np
import torch


def load_pocket_from_pdb(
    pdb_file: str,
    resi_list: list[str] | None = None,
    ref_ligand_coords: np.ndarray | None = None,
    pocket_radius: float = 10.0
) -> tuple[torch.Tensor, dict]:
    """
    Load pocket information from PDB file.
    
    Args:
        pdb_file: Path to PDB file
        resi_list: Optional list of residue IDs (chain:resnum format)
        ref_ligand_coords: Optional reference ligand coordinates
        pocket_radius: Radius for pocket detection (if using ref_ligand_coords)
        
    Returns:
        pocket_coords: (N_res, 3) tensor of CA coordinates
        pocket_features: Dict with residue information
    """
    pocket_coords = []
    pocket_features = {
        'residue_ids': [],
        'residue_names': [],
        'chain_ids': [],
        'residue_nums': []
    }
    
    # Parse PDB
    residues = {}
    
    try:
        with open(pdb_file) as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_name = line[12:16].strip()
                    chain = line[21].strip() or 'A'
                    resnum = line[22:26].strip()
                    resname = line[17:20].strip()
                    
                    resi_id = f"{chain}:{resnum}"
                    
                    # Store CA coordinates
                    if atom_name == 'CA':
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        
                        residues[resi_id] = {
                            'coords': [x, y, z],
                            'name': resname,
                            'chain': chain,
                            'num': resnum
                        }
    except Exception as e:
        raise ValueError(f"Failed to parse PDB file: {e}")
    
    # Filter residues based on input
    if resi_list is not None:
        # Use provided residue list
        for resi_id in resi_list:
            if resi_id in residues:
                res = residues[resi_id]
                pocket_coords.append(res['coords'])
                pocket_features['residue_ids'].append(resi_id)
                pocket_features['residue_names'].append(res['name'])
                pocket_features['chain_ids'].append(res['chain'])
                pocket_features['residue_nums'].append(res['num'])
    
    elif ref_ligand_coords is not None:
        # Find residues near reference ligand
        ligand_center = np.mean(ref_ligand_coords, axis=0)
        
        for resi_id, res in residues.items():
            dist = np.linalg.norm(np.array(res['coords']) - ligand_center)
            if dist <= pocket_radius:
                pocket_coords.append(res['coords'])
                pocket_features['residue_ids'].append(resi_id)
                pocket_features['residue_names'].append(res['name'])
                pocket_features['chain_ids'].append(res['chain'])
                pocket_features['residue_nums'].append(res['num'])
    
    else:
        # Return all residues (full protein)
        for resi_id, res in residues.items():
            pocket_coords.append(res['coords'])
            pocket_features['residue_ids'].append(resi_id)
            pocket_features['residue_names'].append(res['name'])
            pocket_features['chain_ids'].append(res['chain'])
            pocket_features['residue_nums'].append(res['num'])
    
    if not pocket_coords:
        raise ValueError("No pocket residues found")
    
    pocket_coords = torch.tensor(pocket_coords, dtype=torch.float32)
    
    return pocket_coords, pocket_features


def load_pocket_from_sequence(
    sequence: str,
    coords: np.ndarray,
    residue_offset: int = 1
) -> tuple[torch.Tensor, dict]:
    """
    Load pocket from sequence and coordinates.
    
    Args:
        sequence: Amino acid sequence (one-letter codes)
        coords: (N_res, 3) array of CA coordinates
        residue_offset: Starting residue number
        
    Returns:
        pocket_coords: (N_res, 3) tensor
        pocket_features: Dict with residue information
    """
    if len(sequence) != len(coords):
        raise ValueError(
            f"Sequence length ({len(sequence)}) doesn't match "
            f"coordinates ({len(coords)})"
        )
    
    pocket_features = {
        'residue_ids': [],
        'residue_names': [],
        'chain_ids': [],
        'residue_nums': []
    }
    
    # One-letter to three-letter mapping
    aa_map = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
        'X': 'UNK'
    }
    
    for i, aa in enumerate(sequence):
        resnum = residue_offset + i
        pocket_features['residue_ids'].append(f"A:{resnum}")
        pocket_features['residue_names'].append(aa_map.get(aa.upper(), 'UNK'))
        pocket_features['chain_ids'].append('A')
        pocket_features['residue_nums'].append(str(resnum))
    
    pocket_coords = torch.tensor(coords, dtype=torch.float32)
    
    return pocket_coords, pocket_features


def identify_pocket_residues(
    pdb_file: str,
    centroid: tuple[float, float, float],
    radius: float = 10.0
) -> list[str]:
    """
    Identify pocket residues within radius of centroid.
    
    Args:
        pdb_file: Path to PDB file
        centroid: (x, y, z) pocket center coordinates
        radius: Search radius in Angstroms
        
    Returns:
        pocket_ids: List of residue IDs in "chain:resnum" format
    """
    pocket_residues = set()
    cx, cy, cz = centroid
    
    try:
        with open(pdb_file) as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        
                        dist = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
                        
                        if dist <= radius:
                            chain = line[21].strip() or 'A'
                            resnum = line[22:26].strip()
                            if resnum:
                                pocket_residues.add(f"{chain}:{resnum}")
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"Warning: Error reading PDB file: {e}")
        return []
    
    return sorted(list(pocket_residues))


def get_pocket_center(
    pdb_file: str,
    residue_ids: list[str]
) -> tuple[float, float, float]:
    """
    Compute center of mass for pocket residues.
    
    Args:
        pdb_file: Path to PDB file
        residue_ids: List of residue IDs
        
    Returns:
        centroid: (x, y, z) center of mass
    """
    coords = []
    
    residue_set = set(residue_ids)
    
    with open(pdb_file) as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21].strip() or 'A'
                resnum = line[22:26].strip()
                resi_id = f"{chain}:{resnum}"
                
                if resi_id in residue_set:
                    atom_name = line[12:16].strip()
                    if atom_name == 'CA':  # Use CA atoms for center
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
    
    if not coords:
        raise ValueError("No CA atoms found for specified residues")
    
    coords = np.array(coords)
    centroid = coords.mean(axis=0)
    
    return tuple(centroid)


def get_pocket_box(
    pdb_file: str,
    residue_ids: list[str],
    padding: float = 5.0
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Get bounding box for pocket residues.
    
    Args:
        pdb_file: Path to PDB file
        residue_ids: List of residue IDs
        padding: Padding around residues
        
    Returns:
        center: (x, y, z) box center
        size: (x, y, z) box dimensions
    """
    coords = []
    
    residue_set = set(residue_ids)
    
    with open(pdb_file) as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21].strip() or 'A'
                resnum = line[22:26].strip()
                resi_id = f"{chain}:{resnum}"
                
                if resi_id in residue_set:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    
    if not coords:
        raise ValueError("No atoms found for specified residues")
    
    coords = np.array(coords)
    
    min_coords = coords.min(axis=0) - padding
    max_coords = coords.max(axis=0) + padding
    
    center = (min_coords + max_coords) / 2
    size = max_coords - min_coords
    
    return tuple(center), tuple(size)


# Testing
if __name__ == '__main__':
    print("Testing pocket_utils...")
    
    # Create a simple test PDB
    test_pdb = "/tmp/test_pocket.pdb"
    with open(test_pdb, 'w') as f:
        f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
        f.write("ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n")
        f.write("ATOM      3  N   GLY A   2       3.000   0.000   0.000  1.00  0.00           N\n")
        f.write("ATOM      4  CA  GLY A   2       4.458   0.000   0.000  1.00  0.00           C\n")
        f.write("ATOM      5  N   SER A   3       6.000   0.000   0.000  1.00  0.00           N\n")
        f.write("ATOM      6  CA  SER A   3       7.458   0.000   0.000  1.00  0.00           C\n")
    
    # Test loading pocket
    pocket_coords, pocket_features = load_pocket_from_pdb(
        test_pdb,
        resi_list=['A:1', 'A:2']
    )
    print(f"✓ Loaded pocket: {pocket_coords.shape}")
    print(f"  Residues: {pocket_features['residue_ids']}")
    
    # Test centroid detection
    residues = identify_pocket_residues(test_pdb, (2.0, 0.0, 0.0), radius=3.0)
    print(f"✓ Identified residues near (2,0,0): {residues}")
    
    # Test center calculation
    center = get_pocket_center(test_pdb, ['A:1', 'A:2', 'A:3'])
    print(f"✓ Pocket center: {center}")
    
    # Cleanup
    os.remove(test_pdb)
    
    print("\n✓ Tests completed")
