# =====================================================================
# Section 8.0 -- Branch A pre-flight  (rev55 §8.1, mandatory before S4)
#
#   S1  pytest tests/test_simgcl.py -x          (~5 min)
#   S2  verify Branch A CLI defaults            (~10 s)
#   S3  optional 20-epoch smoke                 (set RUN_SMOKE=True)
# =====================================================================
from __future__ import annotations

import os
import subprocess
import sys

try:
    _DAMPS_DIR = DAMPS_DIR  # type: ignore[name-defined]  # noqa: F821
except NameError:
    _cwd = os.getcwd()
    _DAMPS_DIR = (
        _cwd
        if os.path.basename(_cwd) == "MMHCL_DAMPS_Project"
        else os.path.join(_cwd, "MMHCL_DAMPS_Project")
    )

if _DAMPS_DIR not in sys.path:
    sys.path.insert(0, _DAMPS_DIR)
if os.path.normpath(os.getcwd()) != os.path.normpath(_DAMPS_DIR):
    os.chdir(_DAMPS_DIR)

RUN_SMOKE: bool = False
SMOKE_EPOCHS: int = 20
SMOKE_SEED: int = 0

print("=" * 72)
print("Branch A pre-flight (rev55 §8.1)")
print(f"  cwd = {os.getcwd()}")
print("=" * 72)

# ---- S1: SimGCL unit tests ---------------------------------------------
print("\n[S1] pytest tests/test_simgcl.py -x ...")
_env = os.environ.copy()
_env["PYTHONIOENCODING"] = "utf-8"
_proc = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_simgcl.py", "-x", "-q"],
    cwd=_DAMPS_DIR,
    capture_output=True,
    text=True,
    env=_env,
)
print(_proc.stdout or "")
if _proc.returncode != 0:
    print(_proc.stderr or "")
    raise RuntimeError(
        "S1 FAILED: SimGCL unit tests must pass before Branch A sweep. "
        "Recheck branchA_simgcl_batchN.py and model.py patches."
    )
print("[S1] PASS")

# ---- S2: parser defaults -----------------------------------------------
print("\n[S2] verify Branch A CLI defaults ...")
_argv_bak = sys.argv[:]
try:
    sys.argv = ["train.py", "--dataset", "clothing"]
    from utility.parser import parse_args

    _args = parse_args()
finally:
    sys.argv = _argv_bak

_expected = (2, 1, 2048, 2048)
_got = (
    _args.branchA_view_every_k,
    _args.branchA_bcl_batchn,
    _args.branchA_view_bsz,
    _args.branchA_bcl_bsz,
)
print(f"  got      = {_got}")
print(f"  expected = {_expected}")
if _got != _expected:
    raise RuntimeError(
        f"S2 FAILED: Branch A parser defaults mismatch {_got} != {_expected}"
    )
print("[S2] PASS")

# ---- S3: optional short smoke ------------------------------------------
if RUN_SMOKE:
    print(
        f"\n[S3] 20-epoch Branch A smoke (seed={SMOKE_SEED}) ..."
    )
    _smoke = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--dataset",
            "Clothing",
            "--seed",
            str(SMOKE_SEED),
            "--epoch",
            str(SMOKE_EPOCHS),
            "--batch_size",
            "4096",
            "--enable_logq",
            "1",
            "--logq_scale",
            "1.0",
            "--logq_clip",
            "5.0",
            "--enable_simgcl",
            "1",
            "--lambda_view",
            "0.05",
            "--simgcl_eps",
            "0.1",
            "--branchA_view_every_k",
            "2",
            "--branchA_bcl_batchn",
            "1",
            "--branchA_view_bsz",
            "2048",
            "--branchA_bcl_bsz",
            "2048",
            "--use_amp",
            "1",
            "--damps_apc",
            "0",
            "--damps_avrf",
            "0",
            "--damps_imcf",
            "1",
            "--damps_soft_routing",
            "1",
            "--damps_momentum",
            "1",
            "--temperature",
            "0.3",
            "--learnable_tau",
            "0",
        ],
        cwd=_DAMPS_DIR,
        env=_env,
    )
    if _smoke.returncode != 0:
        raise RuntimeError("S3 FAILED: Branch A smoke run exited non-zero.")
    print("[S3] PASS")
else:
    print("\n[S3] skipped (set RUN_SMOKE=True to enable 20-epoch smoke)")

print("\n" + "=" * 72)
print("[BranchA pre-flight] ALL CHECKS PASSED — proceed to Section 8 sweep.")
print("=" * 72)
