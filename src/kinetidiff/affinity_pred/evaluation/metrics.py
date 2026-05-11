"""
Evaluation metrics for affinity prediction.

Provides:
- PCC (Pearson Correlation Coefficient)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Comprehensive model evaluation utilities
"""


import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Pearson Correlation Coefficient.
    
    PCC = cov(Ŷ, Y) / (σ_Ŷ * σ_Y)
    
    Measures linear correlation between predictions and ground truth.
    Range: [-1, 1], higher is better.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        PCC value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) < 2:
        return 0.0
    
    pcc, _ = pearsonr(y_pred, y_true)
    
    if np.isnan(pcc):
        return 0.0
    
    return float(pcc)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    RMSE = sqrt(Σ(ŷ_i - y_i)² / n)
    
    Penalizes large errors more heavily.
    Units: Same as target variable (pKd/pKi).
    Lower is better.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    mse = np.mean((y_pred - y_true) ** 2)
    return float(np.sqrt(mse))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    MAE = Σ|ŷ_i - y_i| / n
    
    Robust to outliers compared to RMSE.
    Units: Same as target variable.
    Lower is better.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    return float(np.mean(np.abs(y_pred - y_true)))


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Spearman Rank Correlation.
    
    Measures monotonic relationship (more robust than PCC).
    Range: [-1, 1], higher is better.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Spearman correlation value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) < 2:
        return 0.0
    
    corr, _ = spearmanr(y_pred, y_true)
    
    if np.isnan(corr):
        return 0.0
    
    return float(corr)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R-squared (coefficient of determination).
    
    R² = 1 - SS_res / SS_tot
    
    Range: (-inf, 1], higher is better.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        R² value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return float(1 - ss_res / ss_tot)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dict with all metrics
    """
    return {
        'pcc': compute_pcc(y_true, y_pred),
        'spearman': compute_spearman(y_true, y_pred),
        'rmse': compute_rmse(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'r2': compute_r2(y_true, y_pred),
    }


class ModelEvaluator:
    """
    Comprehensive model evaluation framework.
    
    Features:
    - Multi-metric evaluation
    - Statistical significance testing
    - Prediction scatter plots
    - Error analysis
    
    Usage:
        evaluator = ModelEvaluator()
        results = evaluator.evaluate(model, test_loader)
    """
    
    def __init__(
        self,
        affinity_mean: float = 0.0,
        affinity_std: float = 1.0,
    ):
        """
        Initialize evaluator.
        
        Args:
            affinity_mean: Mean for denormalization
            affinity_std: Std for denormalization
        """
        self.affinity_mean = affinity_mean
        self.affinity_std = affinity_std
    
    def denormalize(self, values: np.ndarray) -> np.ndarray:
        """Denormalize values to original pKd scale."""
        return values * self.affinity_std + self.affinity_mean
    
    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        denormalize: bool = True,
    ) -> dict[str, float]:
        """
        Evaluate predictions.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            denormalize: Whether to denormalize
            
        Returns:
            Dict with all metrics
        """
        predictions = np.asarray(predictions).flatten()
        targets = np.asarray(targets).flatten()
        
        if denormalize:
            predictions = self.denormalize(predictions)
            targets = self.denormalize(targets)
        
        return compute_all_metrics(targets, predictions)
    
    def print_results(self, metrics: dict[str, float], title: str = "Evaluation Results"):
        """Print formatted evaluation results."""
        print("=" * 60)
        print(title)
        print("=" * 60)
        print(f"  PCC:      {metrics['pcc']:.4f}")
        print(f"  Spearman: {metrics['spearman']:.4f}")
        print(f"  RMSE:     {metrics['rmse']:.4f}")
        print(f"  MAE:      {metrics['mae']:.4f}")
        print(f"  R²:       {metrics['r2']:.4f}")
        print("=" * 60)
