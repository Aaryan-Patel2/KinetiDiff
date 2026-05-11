# FOP Drug Discovery Pipeline - AI Agent Instructions

## Project Overview

**FOP (Fibrodysplasia Ossificans Progressiva) Affinity Predictor** - A Bayesian neural network system for designing kinase inhibitors with **fast dissociation kinetics**. Unlike traditional drug discovery that optimizes for tight binding, this project targets moderate affinity (pKd ~7-8) with rapid dissociation (k_off ~0.1-1 s⁻¹) to prevent BMP pathway shutdown while inhibiting aberrant bone formation.

**Key Goal**: Generate drug candidates via GCDM diffusion model, guided by Bayesian affinity predictions to optimize for the FOP-specific binding profile.

## Architecture

### Core Components

1. **Bayesian Affinity Predictor** (`models/bayesian_affinity_predictor.py`)
   - Hybrid CNN architecture: protein sequence encoder + ligand SMILES encoder
   - Bayesian linear layers with variational inference for uncertainty quantification
   - Returns: pKd, k_off, residence time with uncertainty estimates
   - Training uses ELBO loss (Evidence Lower Bound) for Bayesian posterior

2. **k_off Prediction** (`models/utils/bnn_koff.py`)
   - Empirical method (active): Uses literature correlation `log(k_off) ≈ -0.5 * pKd + 3.0`
   - Returns dissociation kinetics without requiring training data
   - Critical for FOP: evaluates if compound has "fast kinetics" profile

3. **GCDM Integration** (Docker-based)
   - **Location**: `docker/gcdm/` (not `models/gcdm/` - removed due to ARM64 issues)
   - Geometry-Complete Diffusion Model for de novo molecule generation
   - Runs in Docker container with Linux x86_64 + CUDA for dependency compatibility
   - Communication via REST API (`gcdm_api.py`) on port 5000

4. **Data Pipeline** (`models/data_preparation.py`)
   - BindingDB data → sequence encoding + SMILES tokenization + molecular descriptors
   - Converts Ki, IC50, Kd to standardized pKd format
   - Extracts kinetics data (k_on, k_off) when available

### Entry Points

- **Training**: `train_model.py` - Simple CLI for training affinity predictor on ACVR1 data
- **Quick Start**: `quick_start.py` - High-level API (`AffinityPredictor` class) for predictions
- **Generation**: `generate_with_guidance_docker.py` - GCDM + affinity guidance pipeline
- **Testing**: `test_predictions.py`, `test_gcdm_direct.py`

## Critical Patterns

### Vocabulary Size Flexibility

The Bayesian model uses dynamic vocabulary sizes loaded from checkpoints to handle different tokenization schemes:

```python
# In quick_start.py:
ligand_vocab_size = checkpoint['state_dict']['ligand_encoder.embedding.weight'].shape[0]
protein_vocab_size = checkpoint['state_dict']['protein_encoder.embedding.weight'].shape[0]
config['ligand_vocab_size'] = ligand_vocab_size
config['protein_vocab_size'] = protein_vocab_size
model = create_hnn_affinity_model(config)
```

**Why**: Prevents shape mismatches when loading models trained with different data preprocessing.

### PyTorch Lightning Warning

`pytorch-lightning` is **optional** and can cause issues (lzma dependency errors). The codebase works without it:
- Core prediction uses pure PyTorch
- Training can use either Lightning (`BayesianAffinityTrainer`) or standard PyTorch loops
- If Lightning fails, fall back to `models/affinity_predictor.py` for training

### Docker-First for GCDM

**Never** try to install GCDM dependencies directly on ARM64. Always use Docker:

```bash
cd docker/gcdm
docker-compose up -d  # Starts GCDM with GPU support
```

The `DockerGCDMClient` (`models/docker_gcdm_client.py`) handles container communication transparently.

## Workflow Commands

### Setup
```bash
# Install core dependencies (no pytorch-lightning needed)
pip install torch rdkit numpy pandas scikit-learn scipy

# Download BindingDB data (~500MB compressed)
bash download_data.sh

# Start GCDM Docker container
cd docker/gcdm && docker-compose up -d && cd ../..
```

### Training
```bash
# Train affinity predictor (30-60 min on CPU)
python train_model.py --epochs 50 --batch-size 32 --target "ACVR1"

# Output: trained_models/best_model.ckpt
```

### Prediction
```python
from quick_start import AffinityPredictor

predictor = AffinityPredictor(checkpoint_path='trained_models/best_model.ckpt')
result = predictor.predict(
    protein_sequence="MTEYKLVVVGAGG...",
    ligand_smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O"
)
# result: {'affinity': 7.5, 'uncertainty': 0.3, 'koff': 0.18, 'residence_time': 5.6}
```

### Generation with Guidance
```bash
python generate_with_guidance_docker.py \
    --pdb data/structures/acvr1_wt_3mtf.pdb \
    --sequence "PROTEIN_SEQUENCE" \
    --resi-list A:1 A:2 A:3 A:4 A:5 \
    --n-samples 50 \
    --top-k 10 \
    --output-dir generated_molecules
```

## Data Flow

```
BindingDB TSV → AffinityDataPreparator → [protein_seqs, ligand_smiles, descriptors, affinities]
                                                      ↓
                                          BayesianAffinityTrainer
                                                      ↓
                                          trained_models/best_model.ckpt
                                                      ↓
                                              AffinityPredictor
                                                      ↓
                        ┌─────────────────────────────┴──────────────────────────────┐
                        ↓                                                             ↓
              predict(seq, smiles)                              GuidedGCDMDockerGenerator
              returns {pKd, koff, τ}                                    ↓
                                                           Docker GCDM generates molecules
                                                                         ↓
                                                           Rank by FOP score (affinity + kinetics)
                                                                         ↓
                                                           Save top candidates: .smi, .json, .csv
```

## Testing Strategy

- **Unit tests**: Each module has `if __name__ == "__main__"` test blocks
- **Integration tests**: `test_gcdm_direct.py` bypasses wrapper to test core GCDM
- **Prediction tests**: `test_predictions.py` verifies checkpoint loading and inference
- **Docker health**: `docker_gcdm_client.py` includes `test_docker_gcdm()` function

## Common Pitfalls

1. **Don't use `models/gcdm/`** - Directory removed; use Docker setup instead
2. **Checkpoint vocab mismatch** - Always load vocab sizes from checkpoint, not config files
3. **k_off returns None** - Old issue fixed; now returns empirical prediction
4. **PyTorch Lightning errors** - Skip it; use core PyTorch or check `requirements.txt` has it commented out
5. **Docker port conflicts** - If port 5000 busy, edit `docker-compose.yml` to use different port

## File Organization

```
FOP-Code/
├── quick_start.py              # Main prediction API (use this)
├── train_model.py              # Main training script
├── generate_with_guidance_docker.py  # GCDM generation pipeline
├── models/
│   ├── bayesian_affinity_predictor.py  # Core BNN architecture
│   ├── bayesian_training_pipeline.py   # PyTorch Lightning trainer
│   ├── data_preparation.py             # BindingDB → training data
│   ├── docker_gcdm_client.py           # Docker API client
│   └── utils/
│       ├── bnn_koff.py        # k_off prediction (empirical method)
│       └── scoring.py         # FOP suitability scoring
├── docker/gcdm/               # GCDM Docker setup (use this, not models/gcdm/)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── gcdm_api.py           # Flask API for container
├── data/
│   ├── bindingdb_data/       # BindingDB_All.tsv (download via script)
│   └── structures/           # PDB files for generation
└── trained_models/
    └── best_model.ckpt       # Trained affinity predictor
```

## Key Design Decisions

- **Bayesian over standard NN**: Provides uncertainty estimates critical for drug discovery decisions
- **Empirical k_off**: Literature-based correlation avoids need for scarce kinetics training data
- **Docker for GCDM**: Sidesteps ARM64/pytorch-scatter compilation issues
- **Separate affinity predictor**: Trained on host (any arch), guides GCDM in container (x86_64+GPU)
- **FOP-specific scoring**: Custom metric balancing moderate affinity + fast dissociation

## Performance Notes

- Training: 30-60 min (CPU), 10-15 min (GPU) for 50 epochs
- Prediction: ~100ms per compound
- GCDM generation: 2-5 min for 50 molecules (GPU required)
- Docker build: ~15 min first time, ~15GB disk space

## References

- Bayesian architecture based on HNN-Affinity (state-of-art 2020s)
- k_off correlation from Copeland et al. (2006), Tonge (2018)
- GCDM from Morehead & Cheng (2024) "Geometry-complete diffusion for 3D molecule generation"
- BindingDB for training data (updated monthly)

---

*Last updated: 2025-11-15*
