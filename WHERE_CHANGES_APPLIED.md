# Where Code Changes Are Applied in the Notebook

## Overview

The notebook uses `subprocess` to call `main.py`, which in turn uses `parse_args()` from `codes/utility/parser.py`. Here's where each change is applied:

## 1. `codes/utility/parser.py` Changes

### Location in Code Flow:
```
Notebook Cell → subprocess.run(['python', 'main.py', ...]) 
              → main.py (line 18: from utility.parser import parse_args)
              → codes/utility/parser.py (line 26: args = parse_args())
```

### Where It's Used:
- **File:** `codes/main.py` (line 18, 26)
- **Function:** `parse_args()` from `utility.parser`
- **Default Value:** `early_stopping_patience = 20` (changed from 5)

### In the Notebook:
The notebook **explicitly passes** `--early_stopping_patience 20` in the training command (see below), so it overrides the default. However, if you don't specify it, the new default (20) from `parser.py` will be used.

**Notebook Cell 26** (around line 1647):
```python
early_stopping_patience = 20  # Increased from default 5 to allow more training
```

And in the command building (around line 1677):
```python
'--early_stopping_patience', str(early_stopping_patience)  # Added for better convergence
```

## 2. `auto_train.py` Changes

### Important Note:
**`auto_train.py` is NOT directly used by the notebook!** 

It's a separate Python script for command-line training. However, the same changes were made to maintain consistency:

- Default `epoch = 2000` (instead of 1000)
- Default `early_stopping_patience = 20` (instead of 5)

### If You Use auto_train.py:
You would run it from command line:
```bash
python auto_train.py --dataset Clothing
```

The notebook uses its own training cell instead.

## 3. Where Changes Are Applied in the Notebook

### Cell 26: Multi-Seed Training Cell

**Location:** Around line 1629-1677 in the notebook

**Hyperparameters Section (lines ~1629-1647):**
```python
# Training hyperparameters (IMPROVED for better convergence)
# Based on analysis: increased early stopping patience and epochs to match paper results
dataset = 'Clothing'
gpu_id = 0
epoch = 2000  # Increased from 1000 to allow more training
verbose = 5
batch_size = 1024
lr = 0.0001  # Can try 0.0005 if needed
regs = 1e-3  # Can try 1e-4 if needed
embed_size = 64
topk = 5
core = 5
User_layers = 3
Item_layers = 2
user_loss_ratio = 0.03
item_loss_ratio = 0.07
temperature = 0.6
early_stopping_patience = 20  # Increased from default 5 to allow more training
```

**Command Building Section (lines ~1658-1678):**
```python
# Build training command
cmd = [
    'python', 'main.py',
    '--dataset', dataset,
    '--gpu_id', str(gpu_id),
    '--seed', str(seed),
    '--epoch', str(epoch),  # Uses epoch = 2000 from above
    '--verbose', str(verbose),
    '--batch_size', str(batch_size),
    '--lr', str(lr),
    '--regs', str(regs),
    '--embed_size', str(embed_size),
    '--topk', str(topk),
    '--core', str(core),
    '--User_layers', str(User_layers),
    '--Item_layers', str(Item_layers),
    '--user_loss_ratio', str(user_loss_ratio),
    '--item_loss_ratio', str(item_loss_ratio),
    '--temperature', str(temperature),
    '--early_stopping_patience', str(early_stopping_patience)  # Uses early_stopping_patience = 20 from above
]
```

## How It Works

1. **Notebook sets variables:**
   - `epoch = 2000`
   - `early_stopping_patience = 20`

2. **Notebook builds command:**
   - Includes `--epoch 2000`
   - Includes `--early_stopping_patience 20`

3. **Command is executed:**
   ```python
   subprocess.run(cmd, ...)  # Runs: python main.py --epoch 2000 --early_stopping_patience 20 ...
   ```

4. **main.py receives arguments:**
   - `main.py` calls `parse_args()` which parses command-line arguments
   - If `--early_stopping_patience` is not provided, it uses default from `parser.py` (now 20)
   - Since notebook provides it explicitly, the explicit value (20) is used

## Summary

| Change | Where Applied | How It's Used |
|--------|---------------|---------------|
| `parser.py`: `early_stopping_patience = 20` | `codes/utility/parser.py` line 34 | Used as default if not specified in command |
| `epoch = 2000` | **Notebook Cell 26** line ~1634 | Explicitly passed as `--epoch 2000` |
| `early_stopping_patience = 20` | **Notebook Cell 26** line ~1647 | Explicitly passed as `--early_stopping_patience 20` |
| `auto_train.py` changes | `auto_train.py` | Not used by notebook (separate script) |

## Verification

To verify the changes are applied, check:

1. **In the notebook:** Look at Cell 26, around lines 1634 and 1647
2. **In the command output:** When training runs, you should see:
   ```
   Command: python main.py ... --epoch 2000 ... --early_stopping_patience 20 ...
   ```
3. **In the log file:** The Namespace output should show:
   ```
   early_stopping_patience=20, epoch=2000
   ```

