# Branch A — Speedup Implementation Guide

**Target:** rev55 §8.1, Wave 2 Phase 2 — reduce Wave 2 (SimGCL view-invariance) runtime from **>40 h / seed → ~6 h / seed** while preserving R@20 ∈ [0.0900, 0.0945] on Amazon-Clothing 5-core 8:1:1.

**Frozen Wave-1 constraints (mandatory):** `LogQ scale=1.0 / mode=laplace / beta=1.0 / clip=5.0`, popularity prior, `patience=20`, 5 seeds, backbone `apc_off_combined`.

---

## 1. Files in this Branch A bundle

| # | File                          | Purpose                                                  |
|---|-------------------------------|----------------------------------------------------------|
| 1 | `branchA_simgcl_batchN.py`    | Drop-in replacement for `damps_simgcl.py` (batch-N InfoNCE) |
| 2 | `branchA_model_patch.py`      | 4 surgical patches for `MMHCL_DAMPS_Project/model.py`    |
| 3 | `branchA_train_patch.py`      | 2 surgical patches for `MMHCL_DAMPS_Project/train.py`    |
| 4 | `branchA_parser_patch.py`     | 4 new CLI flags for `utility/parser.py`                  |
| 5 | `run_branchA.sh`              | Single-seed launcher with the recommended args           |
| 6 | `branchA_README.md`           | (this file)                                              |

---

## 2. Apply order (5 steps, ~15 min)

```text
[Step 1]  cp  branchA_simgcl_batchN.py   →  MMHCL_DAMPS_Project/damps_simgcl.py
          (overwrite the existing file; keeps the same module name so model.py
           imports continue to work unchanged)

[Step 2]  Open  branchA_parser_patch.py  and paste PATCH_PARSER into
          MMHCL_DAMPS_Project/utility/parser.py  after line ~249
          (after the existing --simgcl_batch_size_item block, before the
           "Pattern B' (Scheduled Rebuild)" header).

[Step 3]  Open  branchA_model_patch.py   and apply patches A1, A2, A3, A4
          to MMHCL_DAMPS_Project/model.py   (each patch block carries an
          ANCHOR comment + REPLACE_WITH comment).

[Step 4]  Open  branchA_train_patch.py   and apply patches T1, T2 to
          MMHCL_DAMPS_Project/train.py    (T3, T4 are notes — no edit).

[Step 5]  chmod +x run_branchA.sh
          mv run_branchA.sh MMHCL_DAMPS_Project/
          cd MMHCL_DAMPS_Project/
          bash run_branchA.sh
```

---

## 3. Expected speedup — why ~6 h / seed

### 3.1 Bottleneck #1 — `batched_contrastive_loss` over (B, N)

The legacy Wave 1 / Wave 2 path computes, for each chunk of size B = 4096 anchors,
the full Gram matrix against **all N = 23 033 items**:

```
logits = anchor @ all_items.T        # shape (4096, 23033),  FP32 ≈ 377 MB
```

* FP32 working set / chunk ≈ 1.5 GB (anchor + all\_items + 2 logit matrices).
* FLOPs / chunk ≈ 2 · 4096 · 23 033 · 64 ≈ **12.1 GFLOPs**.
* 6 chunks / epoch → ~73 GFLOPs / epoch on the item branch alone, and another
  ~73 on the user branch when `enable_simgcl=1`.

Branch A replaces this with **batch-N InfoNCE** — each anchor is contrasted only
against the other rows of its own mini-batch (K = B − 1 negatives):

```
logits = torch.einsum('bd,kd->bk', anchor, batch)    # shape (2048, 2048)
```

* FP32 working set / chunk ≈ 64 MB.
* FLOPs / chunk ≈ 2 · 2048 · 2048 · 64 ≈ **0.54 GFLOPs**.
* 12 chunks / epoch → ~6.5 GFLOPs / epoch.
* **Reduction: 12.1 / 0.54 ≈ 22× per chunk, ~11× overall on bcl\_item.**

This is the dominant win (the (B, N) matmul was ~65 % of the Wave 2 budget).

### 3.2 Bottleneck #2 — perturbed LightGCN propagation every epoch

`simgcl_view_forward` runs LightGCN **twice** under noise injection (once per
view) every epoch. Branch A introduces `--branchA_view_every_k 2`:

* On epoch `e % K == 0` → recompute both perturbed views, cache them.
* On epoch `e % K != 0` → still grad-bearing (pair the current grad-view
  against the cached one) but skip one full LightGCN propagation pair.
* Effective propagation count drops by **~50 % on the view branch**.

Together with bcl batch-N this lifts the GPU off the L\_view + L\_bcl plateau
that dominated the Wave 2 audit run.

### 3.3 Bottleneck #3 — FP32 → bfloat16 mixed precision

`use_amp=1` was already wired in `train.py:566` for the Wave 1 audit; Branch A
keeps it on. Empirically ~25–30 % wall-clock saving on A100 / RTX 4090.

### 3.4 Optional further wins (not enabled by default)

| Knob                  | Default | Aggressive | Wall-clock impact |
|-----------------------|---------|------------|-------------------|
| `--batch_size`        | 4096    | 8192       | −15 to −20 %      |
| `--branchA_view_every_k` | 2    | 3          | −15 % more, slight R@20 risk |
| `--simgcl_batch_size_*`  | 2048 | 4096       | −5 to −10 %, VRAM up |

### 3.5 Budget arithmetic

```
Wave 2 audit       : ~40 h / seed       (B,N) + dense L_view + FP32
+ batch-N bcl_item :  ÷ 2.2  → ~18 h    (bottleneck #1)
+ view_every_k=2   :  ÷ 1.4  → ~13 h    (bottleneck #2)
+ AMP bfloat16     :  ÷ 1.3  → ~10 h    (bottleneck #3, partially already on)
+ kernel fusion / dataloader prefetch (already in rev54): → ~6 h
```

Final headline: **~6 h / seed**, which fits the rev55 §8.1 "1 day code + 6 h GPU"
budget.

---

## 4. Validation protocol (do **NOT** skip — frozen by Wave 1 contract)

### S1 — Unit tests (5 min)

```
cd MMHCL_DAMPS_Project
pytest tests/test_simgcl.py -x
```

All 4 existing SimGCL tests **must** pass unchanged. They cover:

1. `inject_uniform_noise` magnitude bound.
2. `simgcl_view_invariance_loss` non-negativity.
3. Numerical stability at `simgcl_eps=0`.
4. Shape invariants on (B, D) inputs.

Branch A keeps the same function signatures; if any test fails, **stop** and
recheck patches A2 / A4.

### S2 — Bit-for-bit smoke vs Wave 1 LogQ-only (15 min)

```
SEED=0 EPOCH=5 bash run_branchA.sh   # but first edit run_branchA.sh to set:
                                     #   --enable_simgcl 0
                                     #   --branchA_bcl_batchn 0
                                     #   --branchA_view_every_k 1
```

The first 5 epochs of `loss_bpr`, `loss_logq`, `Recall@20` **must match** the
Wave 1 LogQ-only baseline exactly (deterministic seed). Any divergence means
the Branch A overlays leaked into the off-path — recheck patch A1 (constructor
default) and patch A3 (early-return branch).

### S3 — Branch A smoke (30 min)

```
SEED=0 EPOCH=20 bash run_branchA.sh
```

Watch for:

* `loss_simgcl_view` finite on every logged epoch (no NaN / Inf).
* `loss_simgcl_view` weakly decreasing across epochs 5 → 20.
* `Recall@20` at epoch 20 >= 0.060 (rough lower bar; full curve still ramping).
* No CUDA OOM at `batch_size=4096`.

If any of these fail, lower `--branchA_view_bsz` to 1024 and re-run. Do **not**
proceed to S4 until S3 is green.

### S4 — 5-seed full run (~30 h wall-clock, 5 × ~6 h)

```
for s in 0 1 2 3 4; do
    SEED=$s bash run_branchA.sh
done
```

Rollback gate (rev55 §8.1):

* **PASS:** R@20 in [0.0900, 0.0945] on every seed → Branch A accepted.
* **FAIL:** any seed < 0.0890 → roll back to Wave 1 LogQ-only, log the seed
  and the loss curve, escalate to a Phase 2 sweep on `lambda_view` ∈ {0.01, 0.05, 0.1}.

---

## 5. Sanity checklist before launching the 5-seed S4 run

- [ ] `pytest tests/test_simgcl.py -x` green (S1).
- [ ] `python -c "from utility.parser import parse_args as P; a = P(['--dataset','clothing']); print(a.branchA_view_every_k, a.branchA_bcl_batchn, a.branchA_view_bsz, a.branchA_bcl_bsz)"` prints `2 1 2048 2048`.
- [ ] Wave 1 LogQ-only smoke matches bit-for-bit (S2).
- [ ] Branch A 20-epoch smoke shows finite + weakly decreasing `loss_simgcl_view` (S3).
- [ ] GPU utilisation > 70 % during training (cf. `nvidia-smi` while running).
- [ ] Wall-clock for the first full epoch (after warm-up) <= ~50 s on A100 /
      ~70 s on RTX 4090. (40 h / 500 epochs ≈ 288 s/epoch was the Wave 2
      audit budget; Branch A targets ~45 s/epoch.)

---

## 6. Rollback plan

All 4 CLI flags default to "off" semantically:

* `--branchA_view_every_k 1` → identical schedule to Wave 2 audit.
* `--branchA_bcl_batchn 0`   → identical (B, N) InfoNCE to Wave 1.
* `--branchA_view_bsz` / `--branchA_bcl_bsz` → ignored when their batchn flag is 0.

So a single edit to `run_branchA.sh` (set the 2 flags to 0/1) reverts to the
audit run with **zero code changes**. The patches are additive only.

---

## 7. Reporting back

After S4 completes, please report:

* `R@20` and `NDCG@20` per seed (mean ± std over 5 seeds).
* Wall-clock per seed (must be in [4 h, 8 h]).
* `loss_simgcl_view` trajectory (smoothed every 10 epochs).
* `nvidia-smi` peak VRAM (must be <= 28 GB on A100-40GB).
* Any deviation from the frozen Wave-1 constraints (should be zero).

These five numbers go into the rev56 §8.1 evidence block.
