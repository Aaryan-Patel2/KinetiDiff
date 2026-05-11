#!/usr/bin/env python
"""
Training script for HNN-Denovo affinity predictor.

Usage:
    python train_affinity.py --epochs 50 --batch-size 64 --device cuda
    python train_affinity.py --max-entries 1000  # Quick test

Example:
    # Full training on all PDBBind data
    python train_affinity.py --epochs 50 --batch-size 64 --patience 20

    # Quick test with 1000 entries
    python train_affinity.py --max-entries 1000 --epochs 5 --device cpu
"""

import argparse
import os
import sys

import torch

from kinetidiff.affinity_pred.data.featurizers import BINANADescriptor
from kinetidiff.affinity_pred.data.preprocessing import (
    PDBBindConfig,
    PDBBindDataset,
    create_dataloaders,
)
from kinetidiff.affinity_pred.models.hnn_denovo import HNNDenovo, HNNDenovoConfig
from kinetidiff.affinity_pred.models.trainer import HNNDenovoTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HNN-Denovo affinity predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full training
    python train_affinity.py --epochs 50 --batch-size 64 --device cuda

    # Quick test
    python train_affinity.py --max-entries 1000 --epochs 5 --device cpu
        """
    )

    # Data arguments
    parser.add_argument(
        '--data-dir', type=str, default='data/pdbbind_extracted',
        help='PDBBind data directory'
    )
    parser.add_argument(
        '--max-entries', type=int, default=None,
        help='Maximum entries to process (None=all)'
    )
    parser.add_argument(
        '--use-binana', action='store_true',
        help='Use BINANA descriptors (requires PDBQT files)'
    )

    # Model arguments
    parser.add_argument(
        '--descriptor-dim', type=int, default=256,
        help='Descriptor dimension (256 for RDKit fallback, 2048 for BINANA)'
    )

    # Training arguments
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--patience', type=int, default=15,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (cuda/cpu)'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir', type=str, default='affinity_pred/checkpoints',
        help='Output directory for checkpoints'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("HNN-Denovo Binding Affinity Predictor")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check for existing processed dataset
    processed_path = os.path.join(args.output_dir, 'pdbbind_processed.pt')

    if os.path.exists(processed_path) and args.max_entries is None:
        print(f"\nLoading processed dataset from {processed_path}")
        dataset = PDBBindDataset()
        dataset.load(processed_path)
    else:
        # Process data
        print("\n" + "=" * 60)
        print("Processing PDBBind Data")
        print("=" * 60)

        config = PDBBindConfig(
            index_dir=os.path.join(args.data_dir, 'index'),
            extracted_dir=os.path.join(args.data_dir, 'P-L'),
        )

        dataset = PDBBindDataset(config)
        n_processed = dataset.process_dataset(max_entries=args.max_entries)

        if n_processed == 0:
            print("ERROR: No data processed!")
            print("Make sure PDBBind data is extracted to:", args.data_dir)
            sys.exit(1)

        # Split data
        dataset.split_data(stratify=True)

        # Save processed dataset
        dataset.save(processed_path)

    # Setup descriptor computer
    descriptor_computer = None
    if args.use_binana:
        print("\nUsing BINANA descriptors")
        descriptor_computer = BINANADescriptor(use_binana=True)
        args.descriptor_dim = 2048
    else:
        print("\nUsing RDKit fallback descriptors")
        descriptor_computer = BINANADescriptor(use_binana=False)
        args.descriptor_dim = 256

    # Create dataloaders
    print(f"\nCreating dataloaders (batch_size={args.batch_size})...")

    train_loader, val_loader, test_loader, norm_stats = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,  # Use 0 for stability
        descriptor_computer=descriptor_computer,
        descriptor_dim=args.descriptor_dim,
    )

    # Create model
    print("\n" + "=" * 60)
    print("Creating HNN-Denovo Model")
    print("=" * 60)

    model_config = HNNDenovoConfig(
        smiles_vocab_size=len(dataset.smiles_vocab) + 10,  # Add buffer
        protein_vocab_size=22,
        descriptor_dim=args.descriptor_dim,
    )

    model = HNNDenovo(model_config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Create trainer
    trainer = HNNDenovoTrainer(
        model,
        device=args.device,
        learning_rate=args.lr,
        checkpoint_dir=args.output_dir,
    )

    # Train
    print("\n" + "=" * 60)
    print("Training HNN-Denovo")
    print("=" * 60)

    best_metrics = trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        patience=args.patience,
        norm_stats=norm_stats,
    )

    # Test
    print("\n" + "=" * 60)
    print("Testing on held-out set")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(
        os.path.join(args.output_dir, 'best_model.pt'),
        map_location=args.device
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = trainer.test(
        test_loader,
        denormalize=True,
        affinity_mean=norm_stats['affinity_mean'],
        affinity_std=norm_stats['affinity_std'],
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation PCC: {best_metrics['pcc']:.4f}")
    print(f"Test PCC: {test_metrics['pcc']:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
