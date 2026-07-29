#!/usr/bin/env python3
"""scripts/run_optb_rescue.py -- 5-seed rescue batch for Option B on Clothing.
=============================================================================
Runs the 6 target configs from Option B (13-variant sweep, §9.34b) on the 3
KSE-locked seeds that were not yet covered by the 2-seed sweep, so we obtain
5-seed within-batch paired data when merged with optb_13variants_clothing.json.

Targets (6 configs):
  * B00_A0_alpha010    -- fresh 5-seed A0 baseline (all 3 axes on, alpha=0.10)
  * B01_A1_alpha000    -- fresh 5-seed A1 baseline (base cooc, alpha=0)
  * B02_alpha002       -- P1 rescue candidate (2/2 direction vs A0 in 2-seed)
  * B03_alpha005       -- P2 rescue candidate (2/2 direction vs A0)
  * B06_alpha020       -- P3 rescue candidate (mean-highest, SPLIT direction)
  * B11_noLogQ_cooc    -- P4 champion (no LogQ + base cooc + TAMER + NRDMC)

Seeds (3 remaining KSE-locked): [52093548, 109649638, 372270914]

Total: 6 x 3 = 18 runs. Estimated wall ~5.4 h at 18 min/run mean.

Merge logic + paired-t analysis lives in notebook cell §9.35b.

Usage:
    python scripts/run_optb_rescue.py \\
        --python "C:/.../python.exe" \\
        --main   "C:/.../main_tercile.py" \\
        --output "C:/.../results/optb_rescue_3seed_clothing.json"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TARGETS = [
    "B00_A0_alpha010",
    "B01_A1_alpha000",
    "B02_alpha002",
    "B03_alpha005",
    "B06_alpha020",
    "B11_noLogQ_cooc",
]
NEW_SEEDS = [52093548, 109649638, 372270914]


def main() -> int:
    here = Path(__file__).resolve().parent
    project_root = here.parent  # MMHCL_DAMPS_Project/
    default_driver = here / "run_kse_final_5seed.py"

    ap = argparse.ArgumentParser(
        description="5-seed rescue batch for Option B (Clothing)."
    )
    ap.add_argument("--python", type=str, default=sys.executable,
                    help="Python interpreter to run main_tercile.py")
    ap.add_argument("--main", type=Path,
                    default=project_root / "main_tercile.py",
                    help="Path to main_tercile.py")
    ap.add_argument("--driver", type=Path, default=default_driver,
                    help="Path to run_kse_final_5seed.py")
    ap.add_argument("--output", type=Path,
                    default=project_root / "results"
                                        / "optb_rescue_3seed_clothing.json",
                    help="Aggregated JSON output path")
    ap.add_argument("--log_dir", type=Path,
                    default=project_root / "results"
                                        / "_optb_rescue_3seed_logs")
    ap.add_argument("--rsfp_prefix", type=Path,
                    default=project_root / "results"
                                        / "interest_tree_clothing_rsfp")
    ap.add_argument("--base_cache", type=Path,
                    default=project_root / "results"
                                        / "interest_tree_clothing.npz")
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python, str(args.driver),
        "--grid", "optb13",
        "--only_tags", *TARGETS,
        "--seeds", *[str(s) for s in NEW_SEEDS],
        "--epoch", str(args.epoch),
        "--python", args.python,
        "--main", str(args.main),
        "--output", str(args.output),
        "--log_dir", str(args.log_dir),
        "--rsfp_prefix", str(args.rsfp_prefix),
        "--base_cache", str(args.base_cache),
        "--dry_run", str(args.dry_run),
    ]
    n_runs = len(TARGETS) * len(NEW_SEEDS)
    print(f"[optb_rescue] targets ({len(TARGETS)}): {TARGETS}")
    print(f"[optb_rescue] seeds   ({len(NEW_SEEDS)}): {NEW_SEEDS}")
    print(f"[optb_rescue] total runs: {n_runs}  (est. wall "
          f"~{n_runs * 18 / 60:.1f} h at 18 min/run)")
    print(f"[optb_rescue] output : {args.output}")
    print(f"[optb_rescue] log_dir: {args.log_dir}")
    print("=" * 72)
    proc = subprocess.run(cmd, cwd=project_root, check=False)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
