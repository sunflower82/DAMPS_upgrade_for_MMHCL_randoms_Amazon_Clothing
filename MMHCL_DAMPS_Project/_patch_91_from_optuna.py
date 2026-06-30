"""Patch §9.1 notebook cell with best_params from branchA_sports_v1_best.json.

Run this after the Optuna sweep finishes:

    python _patch_91_from_optuna.py [--best_json PATH] [--notebook PATH] [--dry_run]

It rewrites the HP variables at the top of cell 38 and adds a
`# Optuna best_params` comment so the change is traceable.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Mapping: Optuna param name  →  variable name in cell 38 (line match regex)
# ---------------------------------------------------------------------------
_PARAM_TO_CELL38: dict[str, str] = {
    "lr":               "LR",
    "batch_size":       "BATCH_SIZE",
    "simgcl_eps":       "SIMGCL_EPS",
    "lambda_view":      "LAMBDA_VIEW_LOCK",
    "logq_scale":       "LOGQ_SCALE_LOCK",
}

# These live inside BACKBONE_FLAGS as string pairs ("--flag", "value")
_PARAM_TO_FLAG: dict[str, str] = {
    "regs":               "--regs",
    "temperature":        "--temperature",
    "embed_size":         "--embed_size",
    "UI_layers":          "--UI_layers",       # may need to add to BACKBONE_FLAGS
    "use_reduce_lr":      "--use_reduce_lr",
    "reduce_lr_factor":   "--reduce_lr_factor",
    "reduce_lr_patience": "--reduce_lr_patience",
}


def _update_variable_line(src: str, var: str, new_val: object) -> str:
    """Replace  VAR: type = old_val  with the new value."""
    # Match assignment lines like:  LR: float = 0.001  or  BATCH_SIZE: int = 4096
    pat = re.compile(
        r"^(" + re.escape(var) + r"\s*(?::\s*\S+)?\s*=\s*)([^\s#\n]+)(.*)",
        re.MULTILINE,
    )
    replacement = r"\g<1>" + str(new_val) + r"\3"
    new_src, n = pat.subn(replacement, src)
    if n == 0:
        # Variable not found; append it before BACKBONE_FLAGS definition
        insert_before = "BACK_TO_PATIENCE_30"
        new_src = src.replace(
            insert_before,
            f"{var} = {new_val}  # Optuna best\n{insert_before}",
        )
    return new_src


def _update_flag_in_backbone(src: str, flag: str, new_val: object) -> str:
    """Replace ("--flag", "old")  pair inside BACKBONE_FLAGS."""
    escaped = re.escape(flag)
    pat = re.compile(
        r'("' + escaped + r'"\s*,\s*)"([^"]*)"',
        re.MULTILINE,
    )
    replacement = r'\g<1>"' + str(new_val) + '"'
    new_src, n = pat.subn(replacement, src)
    if n == 0:
        # Flag not present — inject it just before the W&B block
        insert_anchor = '"--use_wandb"'
        new_src = src.replace(
            insert_anchor,
            f'"{flag}",                       "{new_val}",\n    {insert_anchor}',
        )
    return new_src


def patch_cell38(
    cell_src: str,
    best_params: dict,
    study_name: str = "branchA_sports_v1",
) -> str:
    src = cell_src

    # Add header comment showing which Optuna run produced these params
    header = (
        f"# ---- Optuna best_params from study={study_name!r} "
        f"(auto-patched by _patch_91_from_optuna.py) ----\n"
    )
    # Insert after the '# ---- 1. Hyperparameters' line
    src = re.sub(
        r"(# ---- 1\. Hyperparameters.*\n)",
        r"\1" + header,
        src,
        count=1,
    )

    # Patch scalar variables
    for param, var in _PARAM_TO_CELL38.items():
        if param in best_params:
            val = best_params[param]
            # Format integers without decimal, floats with scientific if small
            if isinstance(val, float) and val == int(val):
                val = int(val)
            src = _update_variable_line(src, var, val)

    # Patch BACKBONE_FLAGS
    for param, flag in _PARAM_TO_FLAG.items():
        if param in best_params:
            val = best_params[param]
            if isinstance(val, float) and val == int(val):
                val = int(val)
            src = _update_flag_in_backbone(src, flag, val)

    return src


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--best_json",
        default="branchA_sports_v1_best.json",
        help="Path to Optuna best_params JSON (default: branchA_sports_v1_best.json)",
    )
    p.add_argument(
        "--notebook",
        default="../Local_Random_seeds_train_mmhcl_clothing_colab_rev54_wave2.ipynb",
        help="Path to notebook to patch",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print patched cell source without writing",
    )
    args = p.parse_args()

    best_json_path = Path(args.best_json)
    if not best_json_path.exists():
        sys.exit(f"[ERROR] best_json not found: {best_json_path}")

    with open(best_json_path, encoding="utf-8") as f:
        best_data = json.load(f)

    best_params = best_data.get("best_params", best_data)
    study_name  = best_data.get("study_name", "branchA_sports_v1")

    print(f"[Patch] best_params from {best_json_path}:")
    for k, v in best_params.items():
        print(f"  {k:25s} = {v}")

    nb_path = Path(args.notebook)
    if not nb_path.exists():
        sys.exit(f"[ERROR] notebook not found: {nb_path}")

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 38 is §9.1
    cell = nb["cells"][38]
    old_src = "".join(cell["source"])
    new_src = patch_cell38(old_src, best_params, study_name=study_name)

    if args.dry_run:
        # Print only the changed lines
        old_lines = old_src.splitlines()
        new_lines = new_src.splitlines()
        for i, (o, n) in enumerate(zip(old_lines, new_lines), 1):
            if o != n:
                print(f"  Line {i:3d} OLD: {o}")
                print(f"  Line {i:3d} NEW: {n}")
        return

    # Write new source back
    cell["source"] = [ln + "\n" for ln in new_src.splitlines()]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")

    # Clear old outputs so re-run starts fresh
    cell["outputs"] = []
    cell["execution_count"] = None

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"\n[Patch] §9.1 patched in {nb_path}")
    print("[Patch] Old outputs cleared — re-run cell 38 to get 5-seed final eval.")


if __name__ == "__main__":
    main()
