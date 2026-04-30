# Weights & Biases (W&B) Integration Summary

## Overview

Weights & Biases has been successfully integrated into the MMHCL training pipeline for experiment tracking, visualization, and performance monitoring.

## Changes Made

### 1. **Notebook Updates** (`Wanb_Upgraded_Safe_Random_seeds_train_mmhcl_clothing_colab_original.ipynb`)

#### Added Cells:
- **Cell 4 (Markdown)**: W&B setup instructions and benefits
- **Cell 5 (Python)**: W&B installation (`%pip install wandb -q`)
- **Cell 6 (Python)**: W&B login with API key
- **Cell 30 (Updated)**: Added W&B arguments to training command

#### W&B Configuration:
- **API Key**: Pre-configured in login cell
- **Entity**: `baitapck51cc-uet`
- **Project**: `mmhcl-clothing`
- **Run Names**: Auto-generated as `run_{run_idx}_seed_{seed}` for multi-seed training

### 2. **Code Updates**

#### `codes/utility/parser.py`
Added W&B command-line arguments:
```python
--use_wandb: Enable/disable W&B (default: 1)
--wandb_project: Project name (default: 'mmhcl-clothing')
--wandb_entity: Entity/team name (default: 'baitapck51cc-uet')
--wandb_run_name: Custom run name (optional)
```

#### `codes/main.py`
- **Import**: Added `wandb` import with fallback if not installed
- **Trainer.__init__**: 
  - Initializes W&B with hyperparameters as config
  - Watches model for gradient/parameter tracking
  - Logs W&B run URL
- **Trainer.train()**: 
  - Logs training metrics (loss, mf_loss, emb_loss, etc.)
  - Logs validation metrics (recall, precision, NDCG at @10 and @20)
  - Logs test metrics when best model is found
  - Updates summary with best metrics
  - Finishes W&B run at end of training

### 3. **Requirements** (`requirements.txt`)
Added `wandb>=0.15.0` to dependencies

## Metrics Logged to W&B

### Training Metrics (every epoch):
- `train/loss`: Total training loss
- `train/mf_loss`: Matrix factorization loss
- `train/emb_loss`: Embedding loss
- `train/reg_loss`: Regularization loss
- `train/contrastive_loss`: Contrastive learning loss
- `train/epoch_time`: Time per epoch
- `learning_rate`: Current learning rate

### Validation Metrics (every `verbose` epochs):
- `val/recall@10`, `val/recall@20`
- `val/precision@10`, `val/precision@20`
- `val/ndcg@10`, `val/ndcg@20`
- `val/hit_ratio@10`, `val/hit_ratio@20`
- `val/eval_time`: Evaluation time

### Test Metrics (when best model found):
- `test/recall@10`, `test/recall@20`
- `test/precision@10`, `test/precision@20`
- `test/ndcg@10`, `test/ndcg@20`
- `test/hit_ratio@10`, `test/hit_ratio@20`
- `best_val_recall@20`, `best_val_ndcg@20`

### Summary Metrics (final):
- `best_test_recall@20`
- `best_test_precision@20`
- `best_test_ndcg@20`

## Hyperparameters Tracked

All training hyperparameters are automatically logged to W&B config:
- Dataset, seed, epochs, batch size
- Learning rate, regularization
- Embedding size, top-k, core
- User/Item layers, loss ratios
- Temperature, early stopping patience
- Number of users/items

## Usage

### In Google Colab:

1. **Run Setup Cells** (Cells 2-6):
   - Mount Drive
   - Setup checkpoints
   - Install W&B
   - Login to W&B

2. **Run Training Cell** (Cell 30):
   - Training will automatically log to W&B
   - Each seed run creates a separate W&B run
   - View runs at: https://wandb.ai/baitapck51cc-uet/mmhcl-clothing

3. **Monitor Training**:
   - Real-time metrics in W&B dashboard
   - Compare multiple runs
   - Analyze hyperparameter importance
   - View training curves

### Disable W&B (if needed):

Set `--use_wandb 0` in training command, or:
```python
import os
os.environ['WANDB_DISABLED'] = 'true'
```

## Benefits

1. **Real-time Monitoring**: Track training progress without waiting for completion
2. **Experiment Comparison**: Easily compare different hyperparameter settings
3. **Reproducibility**: All hyperparameters automatically logged
4. **Visualization**: Automatic charts and graphs for all metrics
5. **Collaboration**: Share results with team via W&B dashboard
6. **Debugging**: Track gradients and parameters to identify issues

## W&B Dashboard

Access your runs at:
**https://wandb.ai/baitapck51cc-uet/mmhcl-clothing**

Each training run will appear with:
- Run name: `run_{run_idx}_seed_{seed}`
- Tags: `[dataset, seed_{seed}, MMHCL]`
- All hyperparameters in config
- All metrics in charts
- Model gradients and parameters (if enabled)

## Notes

- W&B runs are created automatically when training starts
- Each seed in multi-seed training creates a separate run
- Best metrics are saved to run summary
- W&B run URL is logged to console for easy access
- If W&B is not installed, training continues without W&B (graceful fallback)
