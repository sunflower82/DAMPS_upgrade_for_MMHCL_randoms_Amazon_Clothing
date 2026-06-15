# W&B CSV Analysis and Training Optimization

## Critical Fix: Best Metrics Reporting Issue

### Problem Identified
The notebook was displaying **final epoch validation metrics** (~0.067) instead of **BEST test metrics** (~0.088).

**Root Cause:**
1. `main.py` logged `Test_Recall@20:` during training whenever validation improved
2. At the end of training, `test_ret` variable contained the **last** test results, not the **best**
3. The notebook regex extracted metrics from log files, but if early stopping triggered after the best epoch, the displayed metrics were suboptimal

**Evidence from W&B Logs:**
| Seed | Displayed (Wrong) | Actual Best |
|------|-------------------|-------------|
| seed_2012508887 | 0.0684 | 0.0898 |
| seed_1856557916 | 0.0671 | 0.0897 |
| seed_815029891 | 0.0671 | 0.0858 |
| seed_1035689547 | 0.0666 | 0.0855 |
| seed_1192643361 | 0.0676 | 0.0827 |

### Fix Applied
1. **`codes/main.py`**: Added `best_test_ret` variable to track best test results
2. **`codes/main.py`**: Final logging now uses `BEST_Test_Recall@20:` format with clear markers
3. **Notebook Cell 34**: Updated regex to extract `BEST_Test_Recall@20:` (with fallback to old format)
4. **Notebook Cell 34**: Added comparison section showing results vs paper values

---

## Critical Fix #2: Early Stopping Triggering Too Early

### Problem Identified
Training was being terminated prematurely, preventing the model from reaching optimal performance.

**Evidence from Training Runs:**
| Run | Stopped at Epoch | Recall@20 | Quality |
|-----|------------------|-----------|---------|
| RUN 3 | **266** | **0.0892** | ✅ Matches paper! |
| RUN 2 | 185 | 0.0782 | Moderate |
| RUN 4 | 164 | 0.0785 | Moderate |
| RUN 5 | 59 | 0.0758 | Poor |
| RUN 1 | 53 | 0.0725 | Poor |

**Correlation**: Longer training = Better results. The model needs 200+ epochs to converge.

### Root Cause
- `early_stopping_patience = 5` was too aggressive
- `adaptive_patience` reduced it further for medium datasets
- Model converges slowly due to contrastive learning dynamics

### Fix Applied
1. **Increased `epoch`**: 250 → **500**
2. **Increased `early_stopping_patience`**: 5 → **30**
3. **Added `early_stopping_min_epochs`**: **100** (prevents early stopping before epoch 100)
4. **Disabled `adaptive_patience`**: 1 → **0** (use fixed patience instead)

### Updated Configuration
```python
epoch = 500  # Increased from 250
early_stopping_patience = 30  # Increased from 5
early_stopping_min_epochs = 150  # NEW: minimum epochs before early stopping can trigger
adaptive_patience = 0  # DISABLED
```

---

## Deep Analysis of W&B Export CSVs

### Files Analyzed
- `wandb_export_2026-01-16T23_07_38.788+07_00.csv` (recall)
- `wandb_export_2026-01-16T23_07_53.491+07_00.csv` (precision)
- `wandb_export_2026-01-16T23_08_02.003+07_00.csv` (ndcg)
- `wandb_export_2026-01-16T23_08_08.932+07_00.csv` (hit_ratio)

### Key Findings

#### 1. Convergence Analysis
| Seed | Recall Best | Best Step | % of Training | Last Value | Diff from Best |
|------|-------------|-----------|---------------|------------|----------------|
| 1539261249 | 0.0901 | 524 | 95.4% | 0.0886 | 1.60% |
| 1335270432 | 0.0947 | 924 | 89.8% | 0.0944 | 0.27% |
| 490482737 | 0.0940 | 1389 | 93.0% | 0.0933 | 0.75% |
| 1312751652 | 0.0930 | 764 | 87.9% | 0.0899 | 3.35% |
| 1993999657 | 0.0922 | 814 | 82.7% | 0.0905 | 1.89% |

#### 2. Overall Convergence Summary
- **Recall**: Best reached at ~89.8% of training
- **Precision**: Best reached at ~90.7% of training
- **NDCG**: Best reached at ~91.6% of training
- **Hit Ratio**: Best reached at ~89.8% of training
- **Overall Average**: Best metrics reached at **90.5%** of training

#### 3. Implications
- Best metrics are achieved late in training (82-98% of total epochs)
- Small degradation (0.2-3.3%) occurs after peak
- Training could be more efficient with better early stopping and LR scheduling

### Aggregate Results (5 Seeds)
| Metric | Mean Best | Min | Max |
|--------|-----------|-----|-----|
| Recall@20 | 0.0928 | 0.0901 | 0.0947 |
| Precision@20 | 0.0048 | 0.0047 | 0.0049 |
| NDCG@20 | 0.0428 | 0.0415 | 0.0434 |
| Hit Ratio@20 | 0.0955 | 0.0928 | 0.0975 |

---

## Applied Optimizations

### 1. Adaptive Patience Based on Dataset Size
```python
# Automatically adjusts patience based on n_users:
if n_users < 10000:
    patience = 3  # Small datasets converge faster
elif n_users < 50000:
    patience = 5  # Medium datasets
else:
    patience = 7  # Large datasets may need more patience
```

### 2. ReduceLROnPlateau Scheduler
```python
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='max',           # For Recall/NDCG (higher is better)
    factor=0.5,           # Reduce LR by 50% when plateau
    patience=3,           # Wait 3 epochs before reducing
    min_lr=1e-6           # Don't reduce below this
)
```

### 3. Updated Configuration (Notebook Cell 34)
```python
# Training hyperparameters
epoch = 250
batch_size = 1024
lr = 0.0001
regs = 1e-3

# Early stopping
early_stopping_patience = 5  # Overridden by adaptive_patience
early_stopping_min_delta = 0.0001
early_stopping_monitor = 'val_recall@20'
early_stopping_mode = 'max'
early_stopping_restore_best = 1

# Adaptive patience (auto-adjusts: 3/5/7)
adaptive_patience = 1  # Enabled

# ReduceLROnPlateau scheduler
use_reduce_lr = 1  # Enabled
reduce_lr_factor = 0.5
reduce_lr_patience = 3
reduce_lr_min = 1e-6
```

---

## Files Modified

### 1. `codes/main.py`
- Added adaptive patience logic based on `n_users`
- Integrated ReduceLROnPlateau scheduler
- LR scheduler step now uses validation metric for ReduceLROnPlateau
- Updated W&B config to log new parameters

### 2. `codes/utility/parser.py`
- Added `--adaptive_patience` argument
- Added `--use_reduce_lr` argument
- Added `--reduce_lr_factor` argument
- Added `--reduce_lr_patience` argument
- Added `--reduce_lr_min` argument

### 3. `auto_train.py`
- Added all new arguments to argparse
- Updated command building to include new parameters
- Updated training_kwargs dictionary

### 4. `Optimized_Wanb_Upgraded_Safe_Random_seeds_train_mmhcl_clothing_colab_original.ipynb`
- Added new hyperparameters to training cell
- Updated command building to pass new arguments

---

## Expected Benefits

1. **Faster Convergence**: Adaptive patience will stop training earlier for smaller datasets
2. **Better Peak Performance**: ReduceLROnPlateau helps fine-tune during plateau
3. **Reduced Overfitting**: Early stopping with appropriate patience prevents degradation
4. **Resource Efficiency**: Less wasted compute on non-improving epochs

## Usage Notes

- For **small datasets** (< 10K users): patience=3, expect faster training
- For **medium datasets** (10K-50K users): patience=5, balanced approach
- For **large datasets** (> 50K users): patience=7, more exploration

- Set `adaptive_patience=0` to use fixed `early_stopping_patience` value
- Set `use_reduce_lr=0` to use original LambdaLR scheduler
