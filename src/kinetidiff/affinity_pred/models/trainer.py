"""
Training utilities for HNN-Denovo model.

Provides:
- Pure PyTorch training loop (no Lightning dependency)
- Gradient clipping and learning rate scheduling
- Checkpoint saving with best model tracking
- Comprehensive metric logging
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: (N,) array of predictions
        targets: (N,) array of targets
        
    Returns:
        Dict with 'pcc', 'rmse', 'mae'
    """
    # Flatten if needed
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # PCC (Pearson Correlation Coefficient)
    if len(predictions) > 1:
        pcc, _ = pearsonr(predictions, targets)
        if np.isnan(pcc):
            pcc = 0.0
    else:
        pcc = 0.0
    
    # RMSE
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    
    return {
        'pcc': float(pcc),
        'rmse': float(rmse),
        'mae': float(mae),
    }


class HNNDenovoTrainer:
    """
    Trainer for HNN-Denovo model.
    
    Features:
    - Pure PyTorch (no Lightning dependency)
    - Gradient clipping
    - Learning rate scheduling
    - Best model checkpointing
    - Early stopping
    
    Usage:
        trainer = HNNDenovoTrainer(model, device='cuda')
        trainer.fit(train_loader, val_loader, epochs=50)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = 'checkpoints',
    ):
        """
        Initialize trainer.
        
        Args:
            model: HNN-Denovo model
            device: Training device
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            max_grad_norm: Gradient clipping threshold
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=5,
            min_lr=1e-7
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        self.epochs_without_improvement = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            # Move to device
            ligand = batch['ligand_smiles'].to(self.device)
            protein = batch['protein_seq'].to(self.device)
            descriptors = batch['descriptors'].to(self.device)
            target = batch['affinity'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(ligand, protein, descriptors)
            
            # Clip predictions for stability
            pred = torch.clamp(pred, -10.0, 10.0)
            
            # MSE loss
            loss = F.mse_loss(pred, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> tuple[float, dict[str, float]]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            val_loss, metrics dict
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        for batch in val_loader:
            # Move to device
            ligand = batch['ligand_smiles'].to(self.device)
            protein = batch['protein_seq'].to(self.device)
            descriptors = batch['descriptors'].to(self.device)
            target = batch['affinity'].to(self.device)
            
            # Forward pass
            pred = self.model(ligand, protein, descriptors)
            pred = torch.clamp(pred, -10.0, 10.0)
            
            # Loss
            loss = F.mse_loss(pred, target)
            total_loss += loss.item()
            
            # Collect predictions
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
        
        # Compute metrics
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        metrics = compute_metrics(all_preds, all_targets)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics['loss'], metrics
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: dict[str, float],
        norm_stats: dict | None = None,
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': metrics,
            'norm_stats': norm_stats,
        }
        torch.save(checkpoint, path)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 15,
        norm_stats: dict | None = None,
    ) -> dict[str, float]:
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum training epochs
            patience: Early stopping patience
            norm_stats: Normalization statistics to save
            
        Returns:
            Best validation metrics
        """
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            print("-" * 40)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"PCC: {val_metrics['pcc']:.4f}, "
                  f"RMSE: {val_metrics['rmse']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_metrics = val_metrics
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, 'best_model.pt'),
                    epoch,
                    val_metrics,
                    norm_stats,
                )
                print("  ✓ Saved best model")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
        
        return self.best_val_metrics
    
    @torch.no_grad()
    def test(
        self,
        test_loader: DataLoader,
        denormalize: bool = False,
        affinity_mean: float = 0.0,
        affinity_std: float = 1.0,
    ) -> dict[str, float]:
        """
        Test model on held-out test set.
        
        Args:
            test_loader: Test data loader
            denormalize: Whether to denormalize predictions
            affinity_mean: Mean for denormalization
            affinity_std: Std for denormalization
            
        Returns:
            Test metrics
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        for batch in test_loader:
            # Move to device
            ligand = batch['ligand_smiles'].to(self.device)
            protein = batch['protein_seq'].to(self.device)
            descriptors = batch['descriptors'].to(self.device)
            target = batch['affinity'].to(self.device)
            
            # Forward pass
            pred = self.model(ligand, protein, descriptors)
            pred = torch.clamp(pred, -10.0, 10.0)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Denormalize if requested
        if denormalize:
            all_preds = all_preds * affinity_std + affinity_mean
            all_targets = all_targets * affinity_std + affinity_mean
        
        metrics = compute_metrics(all_preds, all_targets)
        
        print("Test Results:")
        print(f"  PCC:  {metrics['pcc']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        
        return metrics
