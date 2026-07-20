"""main_tercile.py  ──  PACER (DAMPS-MMHCL) tercile-recall wrapper.
================================================================

Drop-in replacement for ``MMHCL_DAMPS_Project/train.py`` that adds four
things on top of the stock PACER training loop:

  1. On every VAL evaluation, computes Recall@20 restricted to items in
     the Head / Mid / Tail popularity tercile (by training frequency).
  2. Whenever val_recall@20 hits a new peak (matching PACER's
     ``BEST_Test_Recall@20`` semantics — see ``train.py`` line 801),
     snapshots both:
         - VAL Head/Mid/Tail at that epoch      (``_best_val_tercile``)
         - TEST Head/Mid/Tail at that epoch     (``_best_test_tercile``)
     independently of WandB.
  3. On every epoch where PACER logs ``val/recall@20`` (line 762) or
     ``test/recall@20`` (line 833) to WandB, we also log:
         val/recall@20_Head|Mid|Tail
         test/recall@20_Head|Mid|Tail
     and at the end of training we write the "best" snapshots into
     ``wandb.summary`` so the run's Overview panel shows them.
  4. Two parser-friendly summary lines are printed at end of run:
         [tercile-final]     BEST_Recall@20_Head=..  Mid=..  Tail=..
         [tercile-test-final] BEST_Test_Recall@20_Head=..  Mid=..  Tail=..

The wrapper does NOT touch the training math -- it only reads the final
user/item embeddings (via a single extra forward pass under ``model.eval``
with ``update_momentum=False``) and computes tercile recall in Python.

Usage (drop-in for train.py; expects the same CLI):
    python main_tercile.py --dataset Clothing --seed <s> \\
        --enable_logq 1 --enable_simgcl 1 [ ... ]
"""
from __future__ import annotations

import math
import os
from typing import Any as _Any

import numpy as np
import torch

# --- 1. Import PACER's train module ---------------------------------------
# Importing ``train`` triggers:
#   * ``utility.parser.parse_args()``  (via utility.batch_test)
#   * ``data_generator`` construction  (via utility.batch_test)
#   * ``Trainer`` class definition
# but NOT training itself (that's guarded by ``if __name__ == "__main__"``).
import train
from utility.batch_test import data_generator, Ks as _BT_KS  # noqa: F401

args = train.args


# --- 2. Item -> tercile assignment (once, at import time) ------------------
_n_items = data_generator.n_items
_item_freq = np.zeros(_n_items, dtype=np.int64)
for _u, _items in data_generator.train_items.items():
    for _i in _items:
        _item_freq[_i] += 1
_order = np.argsort(_item_freq, kind="stable")   # ascending -> tail first
_t1 = _n_items // 3
_t2 = 2 * _n_items // 3
TAIL_IDS: set[int] = set(_order[:_t1].tolist())
MID_IDS:  set[int] = set(_order[_t1:_t2].tolist())
HEAD_IDS: set[int] = set(_order[_t2:].tolist())
print(
    f"[tercile] n_items={_n_items}  "
    f"|Tail|={len(TAIL_IDS)}  |Mid|={len(MID_IDS)}  |Head|={len(HEAD_IDS)}  "
    f"tail-freq<={int(_item_freq[_order[_t1 - 1]])}  "
    f"head-freq>={int(_item_freq[_order[_t2]])}",
    flush=True,
)


# --- 3. GPU-batched tercile recall evaluator (no multiprocessing) ---------
@torch.no_grad()
def compute_tercile_recall(
    ua: torch.Tensor,
    ia: torch.Tensor,
    users_to_test: list[int],
    ground_truth: dict[int, list[int]],
    K: int = 20,
) -> dict[str, float]:
    """Recall@K restricted to each popularity tercile.

    Per-user tercile recall = |hits in top-K that fall in tercile AND
    are ground-truth| / |ground-truth positives that fall in tercile|.
    Users with zero tercile-positives are skipped for that tercile
    (matches Milogradskii et al. 2024, Krichene & Rendle 2020).
    """
    head_scores: list[float] = []
    mid_scores:  list[float] = []
    tail_scores: list[float] = []
    ubs = 2048  # user-batch size for scoring
    for start in range(0, len(users_to_test), ubs):
        batch = users_to_test[start : start + ubs]
        ub = ua[batch]                                # (B, d)
        rate = (ub @ ia.T).cpu().numpy()              # (B, n_items)
        for row, u in enumerate(batch):
            scores = rate[row].copy()
            for ti in data_generator.train_items.get(u, []):
                scores[ti] = -1e9                     # exclude trained items
            ground = ground_truth.get(u, [])
            if not ground:
                continue
            top = np.argpartition(-scores, K)[:K]
            top = top[np.argsort(-scores[top])].tolist()
            gset = set(int(x) for x in ground)
            for tercile, out in (
                (HEAD_IDS, head_scores),
                (MID_IDS,  mid_scores),
                (TAIL_IDS, tail_scores),
            ):
                gt_in_tercile = gset & tercile
                if not gt_in_tercile:
                    continue
                hits = sum(
                    1 for it in top if int(it) in tercile and int(it) in gt_in_tercile
                )
                out.append(hits / len(gt_in_tercile))
    _m = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return {"head": _m(head_scores), "mid": _m(mid_scores), "tail": _m(tail_scores)}


# --- 4. Shared state -------------------------------------------------------
# Most recent val / test tercile (updated every val or test eval).
_last_val_tercile:  dict[str, float] = {"head": float("nan"), "mid": float("nan"), "tail": float("nan")}
_last_test_tercile: dict[str, float] = {"head": float("nan"), "mid": float("nan"), "tail": float("nan")}
# Snapshots at the val_recall@20 peak (matches PACER's BEST_Test_Recall@20
# semantics — see train.py line 801: ``if val["recall"][1] > best_val_recall``).
_best_val_tercile:  dict[str, float] = {"head": float("nan"), "mid": float("nan"), "tail": float("nan")}
_best_test_tercile: dict[str, float] = {"head": float("nan"), "mid": float("nan"), "tail": float("nan")}
_best_val_recall = [0.0]
_val_recall_bumped_this_epoch = [False]  # consumed by the next is_val=False call in the same epoch
_have_val_tercile  = [False]
_have_test_tercile = [False]


# --- 5. Monkey-patch Trainer.test -----------------------------------------
_orig_test = train.Trainer.test


@torch.no_grad()
def _forward_embeddings(self):
    """One extra eval-mode forward pass to read u_ui_emb / i_ui_emb.

    PACER's model.forward returns a dict. ``update_momentum=False`` disables
    Slim Momentum EMA writes; ``epoch=0`` is inert because ``set_epoch`` is
    guarded by ``self.training`` inside model.py.
    """
    was_training = self.model.training
    self.model.eval()
    try:
        out = self.model(
            self.UI_mat,
            self.Item_mat,
            self.User_mat,
            item_indices=None,
            epoch=0,
            update_momentum=False,
        )
    finally:
        if was_training:
            self.model.train()
    return out["u_ui_emb"], out["i_ui_emb"]


def _test_with_terciles(self, users_to_test, is_val):
    result = _orig_test(self, users_to_test, is_val)

    if is_val:
        # ---- VAL branch: compute per-tercile Recall@20 on the val split.
        ua, ia = _forward_embeddings(self)
        ter = compute_tercile_recall(ua, ia, users_to_test, data_generator.val_set)
        _last_val_tercile.update(ter)
        _have_val_tercile[0] = True

        # PACER snapshots BEST_Test_Recall@20 at val_recall PEAK only
        # (train.py:801 `if val["recall"][1] > best_val_recall`). We mirror
        # that with a strict `>` comparison so the next is_val=False call
        # in the SAME epoch knows whether to update _best_test_tercile.
        rec = float(result["recall"][1])
        if rec > _best_val_recall[0]:
            _best_val_recall[0] = rec
            _best_val_tercile.update(ter)
            _val_recall_bumped_this_epoch[0] = True
            print(
                f"[tercile] val@recall-peak: "
                f"head={ter['head']:.6f} mid={ter['mid']:.6f} tail={ter['tail']:.6f} "
                f"(val_recall@20={rec:.6f})",
                flush=True,
            )
        else:
            _val_recall_bumped_this_epoch[0] = False
            print(
                f"[tercile] val: "
                f"head={ter['head']:.6f} mid={ter['mid']:.6f} tail={ter['tail']:.6f} "
                f"(val_recall@20={rec:.6f})",
                flush=True,
            )

        # Per-epoch WandB (adds Head/Mid/Tail alongside PACER's val/recall@20).
        if self.wandb is not None:
            try:
                self.wandb.log({
                    "val/recall@20_Head": ter["head"],
                    "val/recall@20_Mid":  ter["mid"],
                    "val/recall@20_Tail": ter["tail"],
                })
            except Exception as _e:
                print(f"[tercile] wandb.log(val) skipped: {_e}", flush=True)

    else:
        # ---- TEST branch: PACER calls Trainer.test(is_val=False) whenever
        # val_recall@20 OR val_ndcg@20 improved (train.py:789). To match
        # BEST_Test_Recall@20 semantics we snapshot _best_test_tercile ONLY
        # when val_recall bumped its peak in the SAME epoch (train.py:801).
        ua, ia = _forward_embeddings(self)
        ter_t = compute_tercile_recall(ua, ia, users_to_test, data_generator.test_set)
        _last_test_tercile.update(ter_t)
        _have_test_tercile[0] = True

        test_rec = float(result["recall"][1])
        if _val_recall_bumped_this_epoch[0]:
            _best_test_tercile.update(ter_t)
            _val_recall_bumped_this_epoch[0] = False   # consume the flag
            print(
                f"[tercile] test@recall-peak: "
                f"head={ter_t['head']:.6f} mid={ter_t['mid']:.6f} "
                f"tail={ter_t['tail']:.6f} (test_recall@20={test_rec:.6f})",
                flush=True,
            )
        else:
            # val_ndcg-only improvement — record test tercile but don't
            # overwrite the recall-peak snapshot.
            print(
                f"[tercile] test@ndcg-peak: "
                f"head={ter_t['head']:.6f} mid={ter_t['mid']:.6f} "
                f"tail={ter_t['tail']:.6f} (test_recall@20={test_rec:.6f})",
                flush=True,
            )

        # Per-epoch WandB (adds Head/Mid/Tail alongside PACER's test/recall@20).
        if self.wandb is not None:
            try:
                self.wandb.log({
                    "test/recall@20_Head": ter_t["head"],
                    "test/recall@20_Mid":  ter_t["mid"],
                    "test/recall@20_Tail": ter_t["tail"],
                })
            except Exception as _e:
                print(f"[tercile] wandb.log(test) skipped: {_e}", flush=True)

    return result


train.Trainer.test = _test_with_terciles


# --- 6. Monkey-patch Trainer.train to write wandb.summary at the end ------
_orig_train = train.Trainer.train


def _train_with_summary(self):
    try:
        return _orig_train(self)
    finally:
        # Write the best-tercile snapshots into wandb.summary so the run
        # Overview panel surfaces them (PACER writes best_val_* / best_test_*
        # at lines 899-930; we append the Head/Mid/Tail block).
        if self.wandb is not None:
            try:
                if _have_val_tercile[0]:
                    src_v = _best_val_tercile
                    if math.isnan(src_v["head"]):
                        src_v = _last_val_tercile
                    self.wandb.summary["best_recall@20_Head"] = src_v["head"]
                    self.wandb.summary["best_recall@20_Mid"]  = src_v["mid"]
                    self.wandb.summary["best_recall@20_Tail"] = src_v["tail"]
                if _have_test_tercile[0]:
                    src_t = _best_test_tercile
                    if math.isnan(src_t["head"]):
                        src_t = _last_test_tercile
                    self.wandb.summary["best_test_recall@20_Head"] = src_t["head"]
                    self.wandb.summary["best_test_recall@20_Mid"]  = src_t["mid"]
                    self.wandb.summary["best_test_recall@20_Tail"] = src_t["tail"]
            except Exception as _e:
                print(f"[tercile] wandb.summary write skipped: {_e}", flush=True)


train.Trainer.train = _train_with_summary


# --- 7. Formatting + __main__ ---------------------------------------------
def _fmt(x: float) -> str:
    """Format floats for the notebook regex (never emit bare 'nan')."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{float(x):.8f}"


if __name__ == "__main__":
    # Delegate the actual training loop to PACER's train.main() so we stay
    # in lock-step with any future changes to __main__ (seed setup, data
    # config, Trainer instantiation). The Trainer methods we monkey-patched
    # above are already installed, so tercile logic fires automatically.
    train.main()

    # ---- Final snapshots (val_recall peak; matches BEST_Test_Recall@20) --
    final_val = dict(_best_val_tercile)
    if math.isnan(final_val["head"]) and _have_val_tercile[0]:
        final_val = dict(_last_val_tercile)
    final_test = dict(_best_test_tercile)
    if math.isnan(final_test["head"]) and _have_test_tercile[0]:
        final_test = dict(_last_test_tercile)

    # Notebook parser reads these two lines; keep the key format stable.
    print(
        "[tercile-final] "
        f"BEST_Recall@20_Head={_fmt(final_val['head'])} "
        f"BEST_Recall@20_Mid={_fmt(final_val['mid'])} "
        f"BEST_Recall@20_Tail={_fmt(final_val['tail'])}",
        flush=True,
    )
    print(
        "[tercile-test-final] "
        f"BEST_Test_Recall@20_Head={_fmt(final_test['head'])} "
        f"BEST_Test_Recall@20_Mid={_fmt(final_test['mid'])} "
        f"BEST_Test_Recall@20_Tail={_fmt(final_test['tail'])}",
        flush=True,
    )
