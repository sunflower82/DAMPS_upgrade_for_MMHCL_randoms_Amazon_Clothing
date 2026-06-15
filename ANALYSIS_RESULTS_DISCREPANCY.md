# Analysis: MMHCL Results Discrepancy

## Problem Summary

**Paper Results (Amazon Clothing):**
- Recall@20: **0.0881**
- Precision@20: **0.0045**
- NDCG@20: **0.0394**

**Your Results (Mean across 5 runs):**
- Recall@20: **0.067007 ± 0.000534** (23.9% lower)
- Precision@20: **0.003730 ± 0.000031** (17.1% lower)
- NDCG@20: **0.032723 ± 0.000189** (16.9% lower)

## Potential Causes

### 1. **Dataset Split Differences** (Most Likely)
The paper may use a different train/test/validation split than your implementation. Common differences:
- **Different split ratios**: Paper might use 80/10/10 vs your current split
- **Different random seeds for splitting**: Even with same ratios, different seeds create different splits
- **Different dataset version**: Paper might use a different version of Amazon Clothing dataset
- **Different core filtering**: Paper might use different preprocessing (e.g., different core value)

**Check:** Look at your dataset statistics:
```
n_users=39387, n_items=23033
n_interactions=238527
n_train=197338, n_test=41189
```

Compare these with the paper's dataset statistics.

### 2. **Early Stopping** (Likely)
Your code uses early stopping with `patience=5`:
```python
early_stopping_patience = 5
```

This means training stops if validation doesn't improve for 5 consecutive evaluations (every 5 epochs = 25 epochs without improvement). The paper might:
- Train for more epochs without early stopping
- Use different early stopping criteria
- Use a different patience value

**Solution:** Try training with:
- `--early_stopping_patience 10` or `20` (more patience)
- Or disable early stopping by setting a very high patience value

### 3. **Learning Rate Schedule**
Your code uses exponential decay:
```python
fac = lambda epoch: 0.96 ** (epoch / 50)
```

This means:
- At epoch 0: LR = 0.0001
- At epoch 50: LR = 0.0001 × 0.96 = 0.000096
- At epoch 100: LR = 0.0001 × 0.96² = 0.000092
- At epoch 200: LR = 0.0001 × 0.96⁴ = 0.000085

The paper might:
- Use a different learning rate schedule
- Use a constant learning rate
- Use step decay instead of exponential

**Solution:** Try training with a constant learning rate or different schedule.

### 4. **Hyperparameter Differences**
The paper might use different hyperparameters. Check the paper's experimental section for:
- Learning rate (you use 0.0001)
- Batch size (you use 1024)
- Embedding size (you use 64)
- Regularization (you use 1e-3)
- Loss ratios (you use user=0.03, item=0.07)
- Temperature (you use 0.6)

### 5. **Evaluation Protocol**
Subtle differences in evaluation:
- **Test set selection**: Paper might test on all users vs only users with interactions
- **Negative sampling**: Paper might use different negative sampling strategy
- **Evaluation metrics calculation**: Minor differences in metric computation

### 6. **Model Initialization**
Different random seeds or initialization methods can affect final performance, though your multi-seed average should mitigate this.

### 7. **Convergence Issues**
Your model might not be fully converged. Check:
- Are training losses still decreasing when early stopping triggers?
- What epoch does training stop at?
- Are validation metrics still improving?

## Recommended Actions

### Step 1: Check Training Logs
Look at your training logs to see:
- At what epoch did training stop?
- What were the best validation metrics?
- Was the model still improving when it stopped?

### Step 2: Try Longer Training
```python
# In your training cell, modify:
--early_stopping_patience 20  # or even 50
--epoch 2000  # allow more epochs
```

### Step 3: Try Different Hyperparameters
Based on common practices, try:
```python
--lr 0.0005  # Higher learning rate
--batch_size 2048  # Larger batch size
--regs 1e-4  # Lower regularization
```

### Step 4: Check Dataset Statistics
Compare your dataset statistics with the paper:
- Number of users/items
- Train/test split ratios
- Sparsity

### Step 5: Contact Authors
If the discrepancy persists, consider:
- Checking the paper's supplementary material
- Looking for official code repository
- Contacting the authors for exact hyperparameters and dataset version

## Quick Fixes to Try

1. **Increase early stopping patience:**
   ```python
   --early_stopping_patience 20
   ```

2. **Train for more epochs:**
   ```python
   --epoch 2000
   ```

3. **Try different learning rate:**
   ```python
   --lr 0.0005
   ```

4. **Check if model is saving correctly:**
   Look for saved model files in the output directory

## Expected Improvements

If the issue is early stopping or hyperparameters:
- You might see improvements of 10-20% in metrics
- This could bring you closer to paper results

If the issue is dataset split:
- You may need to use the exact same dataset version and split as the paper
- This is harder to fix without access to the paper's exact dataset

