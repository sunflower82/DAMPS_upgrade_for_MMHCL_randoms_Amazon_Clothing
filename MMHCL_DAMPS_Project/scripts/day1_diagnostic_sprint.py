"""
Day-1 Diagnostic Sprint (Phase1_RootCause_Analysis_and_Remediation_Roadmap.tex)
==============================================================================

Implements **D1**, **D2**, and **D3** from Section 2 of the roadmap: cheap probes
that explain Recall@20 vs NDCG@20 asymmetry before heavy Phase 2 work.

D1 — Metadata / APC integrity (H2)
    Audit ``<data_path>/<dataset>/meta_categories.npy`` vs ``n_items`` inferred
    from JSON splits. Exit code 1 if the roadmap's *H2 suspected* condition holds
    (missing file, shape mismatch, or fewer than 5 unique labels).

D2 — Top-K hypergraph bottleneck (H3)
    Spawn ``train.py`` once per ``K`` in ``--topk-list`` using the Phase-1
    **combined** recipe (``tau=0.3``, static ``tau``, ``avrf=0``). Logs are tagged
    with ``--ablation_target diag_D2_topk{K}`` so they do not collide with
    production runs.

D3 — MMHCL pure baseline on the same protocol (H10)
    Spawn ``train.py`` with all DAMPS modules off + paper-style learnable
    ``tau`` at ``0.1``. If ``BEST_Test_Recall@20`` falls in ``0.078--0.081`` while
    the paper reports ``0.0881``, sampled-eval vs all-ranking (H10) is likely.

Usage (``cwd`` = ``MMHCL_DAMPS_Project/``)::

    python scripts/day1_diagnostic_sprint.py --dataset Clothing
    python scripts/day1_diagnostic_sprint.py --dataset Clothing --print-commands \\
        --run-d2 --run-d3 --run-d1-apc-probe
    python scripts/day1_diagnostic_sprint.py --dataset Clothing --run-d2 --run-d3
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from collections import Counter
from typing import Any, Optional

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_PKG = _HERE.parent
_LOG_ROOT = _PKG.parent

_RE_RECALL = re.compile(r"BEST_Test_Recall@20[:=]\s*([\d.]+)")
_RE_NDCG = re.compile(r"BEST_Test_NDCG@20[:=]\s*([\d.]+)")


def infer_n_items_users(data_path: str, dataset: str, core: int) -> tuple[int, int]:
    """Match ``Data.__init__`` dimension inference (max index + 1)."""
    base = os.path.join(data_path, dataset, f"{core}-core")
    n_items = 0
    n_users = 0
    for fname in ("train.json", "test.json", "val.json"):
        fp = os.path.join(base, fname)
        if not os.path.isfile(fp):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            blob: dict[str, list[int]] = json.load(f)
        for uid_str, items in blob.items():
            if not items:
                continue
            uid = int(uid_str)
            n_users = max(n_users, uid)
            n_items = max(n_items, max(items))
    return n_items + 1, n_users + 1


def audit_d1_metadata(
    data_path: str,
    dataset: str,
    core: int,
    num_categories: int,
) -> dict[str, Any]:
    """Print D1 report; set ``h2_suspected`` when roadmap H2 triggers."""
    n_items, n_users = infer_n_items_users(data_path, dataset, core)
    meta_path = os.path.join(data_path, dataset, "meta_categories.npy")
    exists = os.path.isfile(meta_path)

    report: dict[str, Any] = {
        "meta_path": meta_path,
        "exists": exists,
        "n_items_dataset": n_items,
        "n_users_dataset": n_users,
        "num_categories_arg": num_categories,
        "h2_suspected": False,
        "warnings": [],
    }

    print("=" * 72)
    print("D1 - Metadata file audit (roadmap section 2.1, hypothesis H2)")
    print("=" * 72)
    print(f"  dataset          : {dataset}")
    print(f"  data_path        : {data_path}")
    print(f"  core             : {core}")
    print(f"  inferred n_items : {n_items}")
    print(f"  inferred n_users : {n_users}")
    print(f"  meta file        : {meta_path}")
    print(f"  exists           : {exists}")

    if not exists:
        report["h2_suspected"] = True
        report["warnings"].append(
            "meta_categories.npy MISSING -> load_data.py uses deterministic hash "
            "fallback (phantom APC clusters; roadmap H2)."
        )
        print("\n  [WARN] H2 SUSPECTED: no meta file — APC uses hash fallback.")
        return report

    arr = np.load(meta_path).astype(np.int64).reshape(-1)
    uniq = int(len(set(arr.tolist())))
    clipped = np.clip(arr, 0, max(1, num_categories) - 1)
    hist = Counter(clipped.tolist())
    top_hist = dict(sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))[:15])

    report["shape"] = int(arr.shape[0])
    report["unique_raw"] = uniq
    report["histogram_top15"] = {int(k): int(v) for k, v in top_hist.items()}

    print(f"  shape            : {arr.shape[0]}")
    print(f"  unique (raw)     : {uniq}")
    print(
        "  top-15 hist (clipped to "
        f"[0, {num_categories - 1}]): {top_hist}"
    )

    if arr.shape[0] != n_items:
        report["h2_suspected"] = True
        report["warnings"].append(
            f"shape {arr.shape[0]} != n_items {n_items} — loader ignores file."
        )
        print("\n  [WARN] H2 SUSPECTED: shape mismatch — file ignored at train time.")
    if uniq < 5:
        report["h2_suspected"] = True
        report["warnings"].append(
            f"only {uniq} unique metadata labels (roadmap threshold < 5 => H2)."
        )
        print(f"\n  [WARN] H2 SUSPECTED: unique clusters {uniq} < 5.")
    if not report["h2_suspected"]:
        print("\n  [OK] Metadata file present; length matches; >= 5 unique labels.")
    return report


def _base_train_cmd(args: argparse.Namespace, extra: list[str]) -> list[str]:
    cmd: list[str] = [
        args.python_exe,
        "train.py",
        "--dataset",
        args.dataset,
        "--data_path",
        args.data_path,
        "--core",
        str(args.core),
        "--seed",
        str(args.seed),
        "--epoch",
        str(args.epoch),
        "--verbose",
        str(args.verbose),
        "--batch_size",
        str(args.batch_size),
        "--gpu_id",
        str(args.gpu_id),
        "--use_wandb",
        str(int(args.use_wandb)),
        "--rebuild_R",
        str(args.rebuild_R),
        "--User_layers",
        str(args.User_layers),
        "--Item_layers",
        str(args.Item_layers),
        "--regs",
        str(args.regs),
        "--embed_size",
        str(args.embed_size),
        "--user_loss_ratio",
        str(args.user_loss_ratio),
        "--item_loss_ratio",
        str(args.item_loss_ratio),
        "--damps_num_categories",
        str(args.damps_num_categories),
        "--damps_warmup_epochs",
        str(args.damps_warmup_epochs),
        "--use_torch_compile",
        str(int(args.use_torch_compile)),
    ]
    if args.use_wandb:
        cmd += ["--wandb_project", args.wandb_project]
        if args.wandb_entity:
            cmd += ["--wandb_entity", args.wandb_entity]
    return cmd + extra


def _run_train(args: argparse.Namespace, extra: list[str], label: str) -> int:
    cmd = _base_train_cmd(args, extra)
    print("\n" + "-" * 72)
    print(label)
    print("-" * 72)
    print(" ".join(cmd))
    if args.print_commands:
        return 0
    return subprocess.call(cmd, cwd=str(_PKG))


def _parse_log_metrics(log_path: pathlib.Path) -> Optional[tuple[float, float]]:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m_r = _RE_RECALL.search(text)
    m_n = _RE_NDCG.search(text)
    if not m_r or not m_n:
        return None
    return float(m_r.group(1)), float(m_n.group(1))


def expected_log_path(args: argparse.Namespace, extra_flags: list[str]) -> pathlib.Path:
    """Resolve ``../<dataset>/<path_name>/<path_name>.txt`` from train.py."""
    fmap = dict(zip(extra_flags[0::2], extra_flags[1::2]))
    topk = fmap.get("--topk", str(args.topk_default))
    t = fmap.get("--temperature", "0.3")
    taulearn = fmap.get("--learnable_tau", "0")
    apc = fmap.get("--damps_apc", "1")
    avrf = fmap.get("--damps_avrf", "0")
    imcf = fmap.get("--damps_imcf", "1")
    ablation = fmap.get("--ablation_target", "")
    name = (
        f"damps_uu_ii={args.User_layers}_{args.Item_layers}"
        f"_{args.user_loss_ratio}_{args.item_loss_ratio}"
        f"_topk={topk}_t={t}_taulearn={taulearn}_R={args.rebuild_R}"
        f"_apc={apc}_avrf={avrf}_imcf={imcf}"
        f"_regs={args.regs}_dim={args.embed_size}_seed={args.seed}_"
        f"{ablation}"
    )
    return _LOG_ROOT / args.dataset / name / f"{name}.txt"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Day-1 diagnostic sprint (D1 metadata + optional D2/D3 GPU jobs)."
    )
    p.add_argument("--data_path", type=str, default="../data/")
    p.add_argument("--dataset", type=str, default="Clothing")
    p.add_argument("--core", type=int, default=5)
    p.add_argument("--damps_num_categories", type=int, default=10)
    p.add_argument("--seed", type=int, default=737791071,
                   help="Roadmap default probe seed.")
    p.add_argument("--epoch", type=int, default=250,
                   help="Trainer epochs (try 50--75 for a quick smoke).")
    p.add_argument("--verbose", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--rebuild_R", type=int, default=5)
    p.add_argument("--User_layers", type=int, default=3)
    p.add_argument("--Item_layers", type=int, default=2)
    p.add_argument("--user_loss_ratio", type=float, default=0.03)
    p.add_argument("--item_loss_ratio", type=float, default=0.07)
    p.add_argument("--regs", type=float, default=1e-3)
    p.add_argument("--embed_size", type=int, default=64)
    p.add_argument("--damps_warmup_epochs", type=int, default=10)
    p.add_argument("--use_torch_compile", type=int, default=1)
    p.add_argument("--use_wandb", type=int, default=0)
    p.add_argument("--wandb_project", type=str, default="damps-mmhcl-clothing")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--python_exe", type=str, default=sys.executable)
    p.add_argument(
        "--topk-default",
        type=int,
        default=5,
        dest="topk_default",
        help="Default topk for D3 / APC-off probe (paper default is 5).",
    )

    p.add_argument(
        "--print-commands",
        action="store_true",
        help="Print train commands but do not execute them.",
    )
    p.add_argument("--skip-d1", action="store_true")
    p.add_argument("--run-d2", action="store_true", help="Run top-K sweep (GPU).")
    p.add_argument("--run-d3", action="store_true", help="Run pure-MMHCL baseline (GPU).")
    p.add_argument(
        "--run-d1-apc-probe",
        action="store_true",
        help="Single combined-variant train with --damps_apc 0 (GPU).",
    )
    p.add_argument(
        "--topk-list",
        type=int,
        nargs="+",
        default=[10, 15, 20],
        help="K values for D2 (roadmap: 10 15 20; prepend 5 to compare to default).",
    )
    args = p.parse_args()

    exit_code = 0
    if not args.skip_d1:
        rep = audit_d1_metadata(
            args.data_path, args.dataset, args.core, args.damps_num_categories
        )
        if rep["h2_suspected"]:
            exit_code = 1

    wandb_run: list[str] = []

    if args.run_d1_apc_probe:
        extra = [
            "--temperature", "0.3",
            "--learnable_tau", "0",
            "--damps_avrf", "0",
            "--damps_apc", "0",
            "--damps_imcf", "1",
            "--damps_soft_routing", "1",
            "--damps_momentum", "1",
            "--damps_data_driven_prior", "1",
            "--topk", str(args.topk_default),
            "--ablation_target", "diag_D1_apc_off",
        ]
        if args.use_wandb:
            extra += ["--wandb_run_name", f"diag_D1_apc_off_seed{args.seed}"]
        rc = _run_train(args, extra, "D1 follow-up - APC disabled (roadmap train probe)")
        if rc != 0:
            return rc
        if not args.print_commands:
            lp = expected_log_path(args, extra)
            m = _parse_log_metrics(lp)
            print(f"\n  log file : {lp}")
            if m:
                r20, n20 = m
                print(f"  BEST_Test_Recall@20={r20:.6f}  BEST_Test_NDCG@20={n20:.6f}")
                if r20 >= 0.087:
                    print(
                        "  [NOTE] Recall@20 >= 0.087 with APC off — roadmap: H2 strongly suspected."
                    )

    if args.run_d2:
        baseline_r: Optional[float] = None
        first_k: Optional[int] = None
        for k in args.topk_list:
            extra = [
                "--temperature", "0.3",
                "--learnable_tau", "0",
                "--damps_avrf", "0",
                "--damps_apc", "1",
                "--topk", str(k),
                "--ablation_target", f"diag_D2_topk{k}",
            ]
            if args.use_wandb:
                extra += ["--wandb_run_name", f"diag_D2_topk{k}_seed{args.seed}"]
            rc = _run_train(args, extra, f"D2 - topk={k} (combined Phase-1 recipe)")
            if rc != 0:
                return rc
            if not args.print_commands:
                lp = expected_log_path(args, extra)
                m = _parse_log_metrics(lp)
                print(f"\n  log file : {lp}")
                if m:
                    r20, n20 = m
                    print(f"  BEST_Test_Recall@20={r20:.6f}  BEST_Test_NDCG@20={n20:.6f}")
                    if first_k is None:
                        first_k = k
                        baseline_r = r20
                    elif baseline_r is not None:
                        print(f"  ΔRecall@20 vs K={first_k}: {r20 - baseline_r:+.6f}")

    if args.run_d3:
        extra = [
            "--temperature", "0.1",
            "--learnable_tau", "1",
            "--damps_apc", "0",
            "--damps_avrf", "0",
            "--damps_imcf", "0",
            "--damps_soft_routing", "0",
            "--damps_momentum", "0",
            "--damps_data_driven_prior", "0",
            "--topk", str(args.topk_default),
            "--ablation_target", "diag_D3_mmhcl_pure",
        ]
        if args.use_wandb:
            extra += ["--wandb_run_name", f"diag_D3_mmhcl_pure_seed{args.seed}"]
        rc = _run_train(args, extra, "D3 - pure MMHCL (all DAMPS off; roadmap H10)")
        if rc != 0:
            return rc
        if not args.print_commands:
            lp = expected_log_path(args, extra)
            m = _parse_log_metrics(lp)
            print(f"\n  log file : {lp}")
            if m:
                r20, n20 = m
                print(f"  BEST_Test_Recall@20={r20:.6f}  BEST_Test_NDCG@20={n20:.6f}")
                if 0.078 <= r20 <= 0.081:
                    print(
                        "  [H10] Recall in 0.078--0.081 — paper 0.0881 likely used sampled eval."
                    )
                elif 0.0855 <= r20 <= 0.0875:
                    print(
                        "  [H10 unlikely] Recall matches strict all-ranking scale — investigate model gap."
                    )

    if not any((args.run_d2, args.run_d3, args.run_d1_apc_probe)):
        print(
            "\nNext (GPU, from MMHCL_DAMPS_Project/):\n"
            "  python scripts/day1_diagnostic_sprint.py --dataset Clothing "
            "--run-d2 --run-d3\n"
            "  # optional APC-off probe:\n"
            "  python scripts/day1_diagnostic_sprint.py --dataset Clothing "
            "--run-d1-apc-probe\n"
            "Preview commands only:\n"
            "  python scripts/day1_diagnostic_sprint.py --dataset Clothing "
            "--print-commands --run-d2 --run-d3 --run-d1-apc-probe"
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
