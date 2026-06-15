"""
scripts/run_wandb_sweep.py -- Initialise + run a wandb Bayesian sweep
======================================================================

Speedup Guide Section 7 -- automates the hyperparameter sweep over (K,
lambda_coh) using ``wandb sweep`` + ``wandb agent``. This is functionally
equivalent to ``run_optuna_hpo.py`` but uses Weights & Biases as the search
backend so the results land on the existing wandb project dashboard.

Usage
-----
::

    # First run (returns a sweep id)
    python run_wandb_sweep.py --action create

    # Then run the agent (can run in parallel on multiple GPUs)
    python run_wandb_sweep.py --action run --sweep <sweep_id> --count 50
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import wandb
except ImportError as exc:                                      # pragma: no cover
    print("ERROR: wandb is required (pip install wandb).", file=sys.stderr)
    raise SystemExit(2) from exc


_SWEEP_YAML = Path(__file__).resolve().parent / "wandb_sweep.yaml"


def _cmd_create(project: str) -> str:
    """Create a new sweep from ``wandb_sweep.yaml`` and return the sweep id."""
    import yaml                                                  # type: ignore
    with _SWEEP_YAML.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    sweep_id = wandb.sweep(config, project=project)
    print(f"Sweep id: {sweep_id}")
    return sweep_id


def _train_with_config() -> None:
    """Wandb agent target: launch ``train.py`` with sweep-suggested config."""
    run = wandb.init()
    cfg = run.config

    # Map sweep keys -> CLI flags consumed by MMHCL_DAMPS_Project/train.py.
    # Anything not in the sweep falls back to the train.py defaults.
    cli = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "train.py"),
        f"--topk={int(cfg.get('topk', 5))}",
        f"--rebuild_R={int(cfg.get('rebuild_R', 5))}",
        f"--damps_warmup_epochs={int(cfg.get('damps_warmup_epochs', 10))}",
        f"--damps_data_driven_prior={int(cfg.get('damps_data_driven_prior', 1))}",
        f"--use_amp={int(cfg.get('use_amp', 1))}",
        f"--use_torch_compile={int(cfg.get('use_torch_compile', 1))}",
        # Wire wandb so the trainer's per-epoch logs land on the same run
        "--use_wandb=1",
        f"--wandb_project={run.project}",
        f"--wandb_run_name={run.name}",
    ]

    env = os.environ.copy()
    env["WANDB_RUN_ID"] = run.id
    print(f"[sweep] launching trainer: {' '.join(cli)}")
    subprocess.run(cli, env=env, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="DAMPS-MMHCL wandb sweep launcher")
    parser.add_argument(
        "--action", choices=("create", "run"), required=True,
        help="'create' = create a sweep id from wandb_sweep.yaml; "
             "'run' = run an agent against an existing sweep id.",
    )
    parser.add_argument("--project", type=str, default="damps-mmhcl-clothing")
    parser.add_argument("--sweep", type=str, default="",
                        help="Sweep id (required for --action=run).")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of trials this agent should run.")
    args = parser.parse_args()

    if args.action == "create":
        _cmd_create(args.project)
        return 0

    if not args.sweep:
        print("--sweep <id> is required when --action=run", file=sys.stderr)
        return 2
    wandb.agent(args.sweep, function=_train_with_config, count=args.count,
                project=args.project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
