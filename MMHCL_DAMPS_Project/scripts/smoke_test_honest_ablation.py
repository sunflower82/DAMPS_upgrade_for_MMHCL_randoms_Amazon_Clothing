#!/usr/bin/env python3
"""scripts/smoke_test_honest_ablation.py -- fast wiring check for C0/C1.
=============================================================================
Purpose (notebook §9.36):
  Before committing 5-seed wall time (~3 h) to the honest ablation, verify:

  (1) Both C0_pacer_full_new (NRDMC-lite ON) and C1_A1_noNRDMC (NRDMC-lite
      OFF) launch successfully through the driver.
  (2) The two configurations produce NUMERICALLY DIFFERENT BEST_Val_Recall@20
      on the same seed, confirming the --enable_nrdmc_lite flag is correctly
      wired into main_tercile.py (i.e., NRDMC-lite is not silently a no-op
      when disabled, and is not silently active when enabled).

Protocol:
  * Single seed (default: 42 -- not one of the KSE-locked five, so no
    contamination of the paper protocol).
  * Short epoch budget (default: 20 epochs, min_epochs=5, eval_every=5) --
    enough to see divergence between the two configurations but only ~4 min
    wall per run.

Pass criterion:
    abs(R@20_C0 - R@20_C1) > 1e-5  (absolute recall difference, not relative).
    Both runs must exit with code 0.

Non-goal:
    This is NOT a correctness test -- the R@20 magnitudes at 20 epochs are
    not representative of KSE-final numbers.  We only assert the two configs
    diverge, which is a wiring test.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    project_root = here.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--main", type=Path,
                    default=project_root / "main_tercile.py")
    ap.add_argument("--driver", type=Path,
                    default=here / "run_kse_final_5seed.py")
    ap.add_argument("--base_cache", type=Path,
                    default=project_root / "results"
                                        / "interest_tree_clothing.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--output", type=Path,
                    default=project_root / "results"
                                        / "honest_ablation_smoke.json")
    ap.add_argument("--log_dir", type=Path,
                    default=project_root / "results"
                                        / "_honest_ablation_smoke_logs")
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python, str(args.driver),
        "--grid", "honest",
        "--seeds", str(args.seed),
        "--epoch", str(args.epoch),
        "--python", args.python,
        "--main", str(args.main),
        "--base_cache", str(args.base_cache),
        "--output", str(args.output),
        "--log_dir", str(args.log_dir),
        "--dry_run", "0",
    ]
    print(f"[smoke] seed={args.seed}  epoch={args.epoch}  "
          "(min_epochs=30 default so no early stop; ~4 min per run)")
    print(f"[smoke] cmd: {' '.join(cmd)}")
    print("=" * 72)
    proc = subprocess.run(cmd, cwd=project_root, check=False)
    if proc.returncode != 0:
        print(f"[smoke] FAIL driver exit code = {proc.returncode}")
        return proc.returncode

    if not args.output.is_file():
        print(f"[smoke] FAIL no output JSON at {args.output}")
        return 2

    data = json.loads(args.output.read_text())
    rows = {r["tag"]: r for r in data["rows"] if r.get("exit", 0) == 0}
    if "C0_pacer_full_new" not in rows or "C1_A1_noNRDMC" not in rows:
        print(f"[smoke] FAIL missing rows: {sorted(rows.keys())}")
        return 3
    r0 = rows["C0_pacer_full_new"].get("best_val_recall20")
    r1 = rows["C1_A1_noNRDMC"].get("best_val_recall20")
    t0 = rows["C0_pacer_full_new"].get("best_test_recall20")
    t1 = rows["C1_A1_noNRDMC"].get("best_test_recall20")
    print(f"[smoke] C0_pacer_full_new  best_val_R@20 = {r0}   best_test_R@20 = {t0}")
    print(f"[smoke] C1_A1_noNRDMC      best_val_R@20 = {r1}   best_test_R@20 = {t1}")

    if r0 is None or r1 is None:
        print("[smoke] FAIL missing best_val_recall20 in one of the runs")
        return 4

    diff_val  = abs(r0 - r1)
    diff_test = abs((t0 or 0) - (t1 or 0))
    print(f"[smoke] |R@20 val diff|  = {diff_val:.6f}")
    print(f"[smoke] |R@20 test diff| = {diff_test:.6f}")

    if diff_val <= 1e-5:
        print("[smoke] FAIL configs produced IDENTICAL val R@20 -- the "
              "enable_nrdmc_lite flag is NOT correctly wired.")
        return 5
    print("[smoke] PASS -- C0 and C1 produce distinct R@20; ablation "
          "wiring is verified. Safe to launch 5-seed batch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
