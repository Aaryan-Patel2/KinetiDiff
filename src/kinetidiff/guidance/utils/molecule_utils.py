"""
Molecule utility functions for GCDM-Modified.

Provides functions for:
- SMILES/molecule conversions
- File I/O (SDF, PDB)
- Molecular property calculations
- Molecule sanitization and validation
"""

import os
from typing import Optional

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import QED, AllChem, Descriptors, Draw, Lipinski, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def smiles_to_mol(
    smiles: str,
    add_hs: bool = False,
    generate_3d: bool = False
) -> Optional['Chem.Mol']:
    """
    Convert SMILES string to RDKit molecule.
    
    Args:
        smiles: SMILES string
        add_hs: Add explicit hydrogens
        generate_3d: Generate 3D coordinates
        
    Returns:
        RDKit molecule or None if invalid
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule operations")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    if add_hs:
        mol = Chem.AddHs(mol)
    
    if generate_3d:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    
    return mol


def mol_to_smiles(
    mol: 'Chem.Mol',
    canonical: bool = True,
    isomeric: bool = True
) -> str | None:
    """
    Convert RDKit molecule to SMILES string.
    
    Args:
        mol: RDKit molecule
        canonical: Return canonical SMILES
        isomeric: Include stereochemistry
        
    Returns:
        SMILES string or None if conversion fails
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule operations")
    
    if mol is None:
        return None
    
    try:
        return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)
    except:
        return None


def save_molecules_sdf(
    molecules: list['Chem.Mol'],
    filepath: str,
    properties: list[dict] | None = None
) -> int:
    """
    Save molecules to SDF file.
    
    Args:
        molecules: List of RDKit molecules
        filepath: Output file path
        properties: Optional list of property dicts to attach to molecules
        
    Returns:
        Number of molecules written
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule operations")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    writer = Chem.SDWriter(filepath)
    count = 0
    
    for i, mol in enumerate(molecules):
        if mol is None:
            continue
        
        # Add properties if provided
        if properties and i < len(properties):
            for key, value in properties[i].items():
                if isinstance(value, (int, float)):
                    mol.SetDoubleProp(key, float(value))
                else:
                    mol.SetProp(key, str(value))
        
        writer.write(mol)
        count += 1
    
    writer.close()
    return count


def save_molecules_pdb(
    molecules: list['Chem.Mol'],
    output_dir: str,
    prefix: str = 'molecule'
) -> int:
    """
    Save molecules to individual PDB files.
    
    Args:
        molecules: List of RDKit molecules
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Number of molecules written
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule operations")
    
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    
    for i, mol in enumerate(molecules):
        if mol is None:
            continue
        
        # Generate 3D if needed
        if mol.GetNumConformers() == 0:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        
        filepath = os.path.join(output_dir, f'{prefix}_{i:04d}.pdb')
        Chem.MolToPDBFile(mol, filepath)
        count += 1
    
    return count


def compute_molecular_properties(mol: 'Chem.Mol') -> dict:
    """
    Compute standard molecular properties.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        Dictionary with molecular properties
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule operations")
    
    if mol is None:
        return {}
    
    try:
        properties = {
            'smiles': Chem.MolToSmiles(mol),
            'mol_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'num_rings': rdMolDescriptors.CalcNumRings(mol),
            'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'qed': QED.qed(mol),
            'fraction_sp3': rdMolDescriptors.CalcFractionCSP3(mol),
        }
        
        # Lipinski's Rule of Five
        properties['lipinski_violations'] = sum([
            properties['mol_weight'] > 500,
            properties['logp'] > 5,
            properties['hbd'] > 5,
            properties['hba'] > 10
        ])
        
        return properties
        
    except Exception as e:
        return {'error': str(e)}


def validate_molecule(mol: 'Chem.Mol') -> dict:
    """
    Validate a molecule and check for common issues.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        Dictionary with validation results
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule operations")
    
    results = {
        'valid': False,
        'errors': [],
        'warnings': []
    }
    
    if mol is None:
        results['errors'].append("Molecule is None")
        return results
    
    try:
        # Check sanitization
        Chem.SanitizeMol(mol)
        
        # Check for problematic atoms
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H', 'B', 'Si']:
                results['warnings'].append(f"Unusual atom: {symbol}")
        
        # Check for valid valences
        for atom in mol.GetAtoms():
            try:
                atom.GetTotalValence()
            except:
                results['errors'].append(f"Invalid valence for atom {atom.GetIdx()}")
        
        # Check molecular weight
        mw = Descriptors.MolWt(mol)
        if mw > 1000:
            results['warnings'].append(f"High molecular weight: {mw:.1f}")
        if mw < 100:
            results['warnings'].append(f"Low molecular weight: {mw:.1f}")
        
        results['valid'] = len(results['errors']) == 0
        
    except Exception as e:
        results['errors'].append(str(e))
    
    return results


def get_largest_fragment(mol: 'Chem.Mol') -> 'Chem.Mol':
    """
    Get the largest fragment from a molecule.
    
    Useful for removing small counterions or solvent molecules.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        Largest fragment
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule operations")
    
    if mol is None:
        return None
    
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return None
    
    return max(frags, key=lambda x: x.GetNumAtoms())


# Testing
if __name__ == '__main__':
    print("Testing molecule_utils...")
    
    if not RDKIT_AVAILABLE:
        print("RDKit not available. Skipping tests.")
        exit(1)
    
    # Test SMILES conversion
    test_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
    mol = smiles_to_mol(test_smiles, add_hs=True, generate_3d=True)
    
    if mol:
        print(f"✓ Converted SMILES to mol: {mol.GetNumAtoms()} atoms")
        
        # Test properties
        props = compute_molecular_properties(mol)
        print("✓ Computed properties:")
        for key, value in props.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.2f}")
            else:
                print(f"    {key}: {value}")
        
        # Test validation
        validation = validate_molecule(mol)
        print(f"✓ Validation: {validation}")
    
    print("\n✓ Tests completed")
