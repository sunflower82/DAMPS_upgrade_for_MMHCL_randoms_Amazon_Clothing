"""
utility/batch_test.py — Evaluation Pipeline
=============================================

Evaluates trained DAMPS-MMHCL models on validation/test splits using
GPU-accelerated scoring.

Public API
----------
*   ``data_generator`` — global ``Data`` instance, shared with ``train.py``.
*   ``test_torch(ua_emb, ia_emb, users_to_test, is_val)`` — evaluation.
"""

from __future__ import annotations

import heapq
import multiprocessing
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from utility import metrics as mtr
from utility.load_data import Data
from utility.parser import parse_args


args = parse_args()
Ks: list[int] = eval(args.Ks)

# ---------------------------------------------------------------------------
#  Multiprocessing pool size (capped to avoid host CPU starvation)
# ---------------------------------------------------------------------------
_cpu_total: int = multiprocessing.cpu_count()
cores: int = max(1, min(_cpu_total // 5 if _cpu_total > 5 else 1, 8))


# ---------------------------------------------------------------------------
#  Module-level data generator
# ---------------------------------------------------------------------------
data_generator: Data = Data(
    path=args.data_path + args.dataset, batch_size=args.batch_size
)
USR_NUM: int = data_generator.n_users
ITEM_NUM: int = data_generator.n_items
N_TRAIN: int = data_generator.n_train
N_TEST: int = data_generator.n_test
BATCH_SIZE: int = args.batch_size


MetricsDict = dict[str, Any]


# ---------------------------------------------------------------------------
#  Per-user ranking helpers
# ---------------------------------------------------------------------------
def ranklist_by_heapq(
    user_pos_test: list[int],
    test_items: list[int],
    rating: npt.NDArray[np.floating],
    Ks_local: list[int],
) -> tuple[list[int], float]:
    """Top-K ranking via heapq (faster, no AUC)."""
    item_score = {i: rating[i] for i in test_items}
    K_max = max(Ks_local)
    top_items = heapq.nlargest(K_max, item_score, key=item_score.get)
    r = [1 if i in user_pos_test else 0 for i in top_items]
    return r, 0.0


def ranklist_by_sorted(
    user_pos_test: list[int],
    test_items: list[int],
    rating: npt.NDArray[np.floating],
    Ks_local: list[int],
) -> tuple[list[int], float]:
    """Full sort ranking + AUC."""
    item_score = {i: rating[i] for i in test_items}
    K_max = max(Ks_local)
    top_items = heapq.nlargest(K_max, item_score, key=item_score.get)
    r = [1 if i in user_pos_test else 0 for i in top_items]
    sorted_items = sorted(item_score.items(), key=lambda kv: kv[1], reverse=True)
    posterior = [v for _, v in sorted_items]
    gt = [1 if i in user_pos_test else 0 for i, _ in sorted_items]
    return r, mtr.auc(ground_truth=gt, prediction=posterior)


def get_performance(
    user_pos_test: list[int],
    r: list[int],
    auc_val: float,
    Ks_local: list[int],
) -> MetricsDict:
    """Return per-K metric arrays for one user."""
    precision: list[float] = []
    recall: list[float] = []
    ndcg: list[float] = []
    hit_ratio: list[float] = []
    for K in Ks_local:
        precision.append(mtr.precision_at_k(r, K))
        recall.append(mtr.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(mtr.ndcg_at_k(r, K))
        hit_ratio.append(mtr.hit_at_k(r, K))
    return {
        "recall": np.array(recall),
        "precision": np.array(precision),
        "ndcg": np.array(ndcg),
        "hit_ratio": np.array(hit_ratio),
        "auc": auc_val,
    }


def test_one_user(x: tuple[npt.NDArray[np.floating], int, bool]) -> MetricsDict:
    """Evaluate a single user — designed for multiprocessing.Pool.map."""
    rating: npt.NDArray[np.floating] = x[0]
    u: int = x[1]
    is_val: bool = x[2]

    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []

    user_pos_test = (
        data_generator.val_set[u] if is_val else data_generator.test_set[u]
    )
    candidates = list(set(range(ITEM_NUM)) - set(training_items))

    if args.test_flag == "part":
        r, auc_val = ranklist_by_heapq(user_pos_test, candidates, rating, Ks)
    else:
        r, auc_val = ranklist_by_sorted(user_pos_test, candidates, rating, Ks)
    return get_performance(user_pos_test, r, auc_val, Ks)


# ---------------------------------------------------------------------------
#  Main evaluation entry point
# ---------------------------------------------------------------------------
def test_torch(
    ua_embeddings: torch.Tensor,
    ia_embeddings: torch.Tensor,
    users_to_test: list[int],
    is_val: bool,
    batch_test_flag: bool = False,
) -> MetricsDict:
    """
    Evaluate the model on a list of users.

    Args:
        ua_embeddings   : (n_users, d) all user embeddings (GPU).
        ia_embeddings   : (n_items, d) all item embeddings (GPU).
        users_to_test   : list of user IDs to evaluate.
        is_val          : True for validation, False for test.
        batch_test_flag : if True, score in item chunks (lower VRAM peak).

    Returns:
        dict with keys: 'precision', 'recall', 'ndcg', 'hit_ratio', 'auc'.
    """
    result: MetricsDict = {
        "precision": np.zeros(len(Ks)),
        "recall": np.zeros(len(Ks)),
        "ndcg": np.zeros(len(Ks)),
        "hit_ratio": np.zeros(len(Ks)),
        "auc": 0.0,
    }
    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2
    n_test_users = len(users_to_test)
    n_user_batchs = (n_test_users + u_batch_size - 1) // u_batch_size
    count = 0

    try:
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = min((u_batch_id + 1) * u_batch_size, n_test_users)
            user_batch = users_to_test[start:end]
            if not user_batch:
                continue

            u_emb = ua_embeddings[user_batch]
            rate_batch = (
                torch.matmul(u_emb, ia_embeddings.transpose(0, 1)).detach().cpu().numpy()
            )

            payload = list(zip(rate_batch, user_batch, [is_val] * len(user_batch)))
            batch_result = pool.map(test_one_user, payload)
            count += len(batch_result)

            for re in batch_result:
                result["precision"] += re["precision"] / n_test_users
                result["recall"] += re["recall"] / n_test_users
                result["ndcg"] += re["ndcg"] / n_test_users
                result["hit_ratio"] += re["hit_ratio"] / n_test_users
                result["auc"] += re["auc"] / n_test_users
    finally:
        pool.close()
        pool.join()

    if count != n_test_users:
        # Don't crash on slight mismatches (users without val/test items get skipped)
        pass
    return result
