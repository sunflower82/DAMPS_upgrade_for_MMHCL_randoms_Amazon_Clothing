"""killswitch.py -- OOM / NaN / wall-clock guard for long PACER grid runs.

Wraps a child command (typically a grid driver) and monitors:

  * VRAM used on the target GPU (via ``nvidia-smi`` polling every ``--poll``
    seconds). If usage stays >= ``--vram_frac_max`` for
    ``--vram_hits_max`` consecutive polls the child is terminated.

  * Child stdout stream (tee'd to our stdout so notebook capture still
    works). Lines matching ``--nan_regex`` beyond ``--nan_epochs_max``
    times trigger a kill. Lines matching ``--oom_regex`` (default catches
    ``torch.cuda.OutOfMemoryError`` / ``CUDA out of memory``) trigger an
    IMMEDIATE kill on the first hit.

  * Wall-clock. Child running longer than ``--wall_hours_max`` hours is
    killed with SIGTERM, then SIGKILL after ``--grace_seconds`` if still
    alive.

Exit codes:
    0                    child exited cleanly
    <child rc>           child exited non-zero on its own
    130                  killed by us (any guard condition)
    2                    argparse / setup error

Usage (from ``MMHCL_DAMPS_Project/``):

    python scripts/killswitch.py \\
        --gpu_id 0 --vram_frac_max 0.95 --nan_epochs_max 3 \\
        --wall_hours_max 8 --log_json killswitch_p3.json \\
        -- python scripts/run_p3_ptv_grid.py

Everything after the double-dash is the child command. Standalone
(no monitored child) mode is not supported -- we only guard our own
subprocess so we always have a PID and a live stdout to inspect.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Signal-safe kill helper
# ---------------------------------------------------------------------------
def _terminate(proc: subprocess.Popen, grace: float, log) -> None:
    """SIGTERM the child, then SIGKILL after ``grace`` seconds if alive."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            # Windows: send CTRL_BREAK to the process group.
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            proc.terminate()
        log(f"[killswitch] sent SIGTERM to PID {proc.pid}")
    except Exception as exc:                                      # pragma: no cover
        log(f"[killswitch] SIGTERM failed: {exc!r}")
    t0 = time.time()
    while time.time() - t0 < grace and proc.poll() is None:
        time.sleep(0.5)
    if proc.poll() is None:
        try:
            proc.kill()
            log(f"[killswitch] escalated to SIGKILL (grace {grace}s expired)")
        except Exception as exc:                                  # pragma: no cover
            log(f"[killswitch] SIGKILL failed: {exc!r}")


# ---------------------------------------------------------------------------
# nvidia-smi polling
# ---------------------------------------------------------------------------
def _query_gpu(gpu_id: int) -> tuple[float, float] | None:
    """Return (used_MiB, total_MiB) for ``gpu_id`` or None on failure."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip().splitlines()
        if not raw:
            return None
        used_s, total_s = raw[0].split(",")
        return float(used_s.strip()), float(total_s.strip())
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main guard loop
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="OOM / NaN / wall-clock kill-switch for PACER runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--gpu_id", type=int, default=0,
                    help="GPU index to monitor with nvidia-smi.")
    ap.add_argument("--poll", type=float, default=5.0,
                    help="VRAM poll interval (seconds).")
    ap.add_argument("--vram_frac_max", type=float, default=0.95,
                    help="VRAM used/total fraction that counts as 'saturated'.")
    ap.add_argument("--vram_hits_max", type=int, default=3,
                    help="Consecutive saturated polls before kill.")
    ap.add_argument("--nan_regex", type=str,
                    default=r"loss[^\n=]*=\s*(nan|inf|-inf)",
                    help="Regex matched against child stdout lines "
                         "(case-insensitive).")
    ap.add_argument("--nan_epochs_max", type=int, default=3,
                    help="Kill after this many NaN-match lines.")
    ap.add_argument("--oom_regex", type=str,
                    default=r"(CUDA out of memory|OutOfMemoryError|"
                            r"cudaErrorMemoryAllocation)",
                    help="Regex that triggers IMMEDIATE kill on first match.")
    ap.add_argument("--wall_hours_max", type=float, default=12.0,
                    help="Kill child after this many wall-clock hours.")
    ap.add_argument("--grace_seconds", type=float, default=15.0,
                    help="Seconds between SIGTERM and SIGKILL.")
    ap.add_argument("--log_json", type=str, default="killswitch_events.json",
                    help="Append-mode JSON file for guard events. "
                         "Directory is created if missing.")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress kill-switch status prints "
                         "(child stdout still tee'd).")
    # Everything after `--` is the child command.
    args, rest = ap.parse_known_args()
    if not rest:
        print("[killswitch] no child command after '--'.", file=sys.stderr)
        return 2
    # Strip the "--" separator if argparse left it in.
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        print("[killswitch] empty child command.", file=sys.stderr)
        return 2

    events: list[dict] = []

    def _log(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)
        events.append({
            "t": datetime.now(timezone.utc).isoformat(),
            "msg": msg,
        })

    _log(f"[killswitch] guarding: {shlex.join(rest)}")
    _log(f"[killswitch] gpu_id={args.gpu_id} "
         f"vram_frac_max={args.vram_frac_max} "
         f"nan_epochs_max={args.nan_epochs_max} "
         f"wall_hours_max={args.wall_hours_max}")

    # Probe nvidia-smi once so we fail fast if it's missing.
    probe = _query_gpu(args.gpu_id)
    if probe is None:
        _log(f"[killswitch] WARNING: nvidia-smi probe on gpu {args.gpu_id} "
             "returned nothing; VRAM guard disabled.")
    else:
        _log(f"[killswitch] gpu {args.gpu_id} initial "
             f"{probe[0]:.0f}/{probe[1]:.0f} MiB "
             f"({probe[0] / probe[1] * 100:.1f}%)")

    # Ensure log dir exists early so late-run OS errors don't lose events.
    log_path = Path(args.log_json)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Spawn child.
    popen_kwargs: dict = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    t0 = time.time()
    try:
        proc = subprocess.Popen(rest, **popen_kwargs)
    except FileNotFoundError as exc:
        _log(f"[killswitch] failed to spawn child: {exc!r}")
        return 2

    kill_reason = {"reason": None}
    nan_hits = [0]
    vram_hits = [0]

    # --- VRAM guard thread ------------------------------------------------
    def _vram_guard() -> None:
        while proc.poll() is None:
            time.sleep(args.poll)
            q = _query_gpu(args.gpu_id)
            if q is None:
                continue
            used, total = q
            frac = used / total if total > 0 else 0.0
            if frac >= args.vram_frac_max:
                vram_hits[0] += 1
                _log(f"[killswitch] VRAM {frac * 100:.1f}% "
                     f"({used:.0f}/{total:.0f} MiB) "
                     f"hit {vram_hits[0]}/{args.vram_hits_max}")
                if vram_hits[0] >= args.vram_hits_max:
                    kill_reason["reason"] = (
                        f"vram_saturated: {frac * 100:.1f}% for "
                        f"{vram_hits[0]} consecutive polls"
                    )
                    _terminate(proc, args.grace_seconds, _log)
                    return
            else:
                vram_hits[0] = 0

    # --- Wall-clock guard thread ------------------------------------------
    def _walltime_guard() -> None:
        while proc.poll() is None:
            elapsed_h = (time.time() - t0) / 3600.0
            if elapsed_h >= args.wall_hours_max:
                kill_reason["reason"] = (
                    f"wall_clock: {elapsed_h:.2f}h "
                    f">= {args.wall_hours_max}h"
                )
                _log(f"[killswitch] {kill_reason['reason']}")
                _terminate(proc, args.grace_seconds, _log)
                return
            # Sleep until the wall-clock cap or 60s -- whichever is sooner.
            remaining = args.wall_hours_max * 3600.0 - (time.time() - t0)
            time.sleep(min(60.0, max(1.0, remaining)))

    tv = threading.Thread(target=_vram_guard, daemon=True)
    tw = threading.Thread(target=_walltime_guard, daemon=True)
    tv.start()
    tw.start()

    nan_re = re.compile(args.nan_regex, re.IGNORECASE)
    oom_re = re.compile(args.oom_regex, re.IGNORECASE)

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            # Tee to our stdout so wrapping notebooks / files still see it.
            sys.stdout.write(line)
            sys.stdout.flush()
            if oom_re.search(line):
                kill_reason["reason"] = f"oom_stdout: {line.strip()[:200]}"
                _log(f"[killswitch] OOM signature detected -> kill")
                _terminate(proc, args.grace_seconds, _log)
                break
            if nan_re.search(line):
                nan_hits[0] += 1
                _log(f"[killswitch] NaN/Inf detected "
                     f"({nan_hits[0]}/{args.nan_epochs_max}): "
                     f"{line.strip()[:200]}")
                if nan_hits[0] >= args.nan_epochs_max:
                    kill_reason["reason"] = (
                        f"nan_persistent: {nan_hits[0]} matching lines"
                    )
                    _terminate(proc, args.grace_seconds, _log)
                    break
    except KeyboardInterrupt:
        kill_reason["reason"] = "keyboard_interrupt"
        _log("[killswitch] KeyboardInterrupt -> forwarding to child")
        _terminate(proc, args.grace_seconds, _log)

    rc = proc.wait()
    wall_h = (time.time() - t0) / 3600.0
    if kill_reason["reason"]:
        _log(f"[killswitch] child TERMINATED after {wall_h:.2f}h "
             f"reason={kill_reason['reason']!r} child_rc={rc}")
        final_rc = 130
    else:
        _log(f"[killswitch] child exited rc={rc} after {wall_h:.2f}h")
        final_rc = rc

    # Persist event log (append-safe by writing an array with prior events).
    prior: list = []
    if log_path.exists():
        try:
            prior = json.loads(log_path.read_text(encoding="utf-8"))
            if not isinstance(prior, list):
                prior = []
        except Exception:
            prior = []
    prior.append({
        "cmd": rest,
        "start_utc": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
        "wall_hours": round(wall_h, 4),
        "child_rc": rc,
        "kill_reason": kill_reason["reason"],
        "final_rc": final_rc,
        "events": events,
        "settings": {
            "gpu_id": args.gpu_id,
            "vram_frac_max": args.vram_frac_max,
            "vram_hits_max": args.vram_hits_max,
            "nan_regex": args.nan_regex,
            "nan_epochs_max": args.nan_epochs_max,
            "oom_regex": args.oom_regex,
            "wall_hours_max": args.wall_hours_max,
        },
    })
    log_path.write_text(json.dumps(prior, indent=2), encoding="utf-8")
    _log(f"[killswitch] events written to {log_path.as_posix()}")

    return final_rc


if __name__ == "__main__":
    raise SystemExit(main())
