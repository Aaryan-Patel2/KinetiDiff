# Affinity Predictor Checkpoint

The trained HNN-Denovo model checkpoint (`best_model.pt`, ~25MB) is not tracked by git.

## Performance
- PCC: 0.72
- RMSE: 0.70
- MAE: 0.53
- Training data: BindingDB (70,248 samples)

## Download
Copy from the original location:
```bash
cp /path/to/models/affinity_pred/checkpoints/best_model.pt src_dupe/affinity_pred/checkpoints/
```

Or retrain:
```bash
python train/train_affinity.py --epochs 50 --batch-size 32
```
