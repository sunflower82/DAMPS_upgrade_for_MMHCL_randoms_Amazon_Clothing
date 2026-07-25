"""scripts/rehydrate_p6_json.py -- offline best_epoch back-fill for P6.x logs.

Historical context
------------------
Early P6 driver scripts (``run_p6_2_pca_convergence.py`` before the fix
in commit 3b8aeb0) had a stale regex looking for the token
``best_epoch: <N>``, but the actual trainer emits::

    [<ts>] BEST_Val_Recall_Peak_Epoch: <int>
    [<ts>] BEST_Val_NDCG_Peak_Epoch:   <int>

This mismatch left ``per_seed[i]['best_epoch']`` at NaN in the emitted
JSON payload and produced ``best_epoch mean=nan`` at the bottom of the
run log. The training itself was unaffected -- only the summary field.

This script reparses an existing ``.txt`` training log with the correct
regex, patches the corresponding ``.json`` in-place (or writes a new
file when ``--out`` is set), and re-aggregates the ``ranked[]``
``best_epoch`` cell so downstream tooling sees a real integer.

The rehydration is idempotent: rerunning it on an already-patched
JSON is a no-op (the NaN check is the only trigger for a write).

Preferred token
---------------
By default we back-fill ``best_epoch`` from
``BEST_Val_Recall_Peak_Epoch`` because early-stopping is driven by
``val_recall@20``. Pass ``--peak_from ndcg`` to use
``BEST_Val_NDCG_Peak_Epoch`` instead, or ``--peak_from either`` to
accept whichever token appears first (useful for older log formats
that only emit one).

Usage
-----
Rehydrate the attached P6.2 result JSON using the matching log::

    python scripts/rehydrate_p6_json.py \
        --log  path/to/p6_2_pca_convergence_clothing.txt \
        --json path/to/p6_2_pca_convergence_clothing.json

Write to a sibling file instead of mutating in place::

    python scripts/rehydrate_p6_json.py \
        --log ... --json ... \
        --out path/to/p6_2_pca_convergence_clothing.rehydrated.json

The script matches per-seed rows to log sections by ``seed=<int>`` in
the run banner. Rows that already carry a finite ``best_epoch`` are
left untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
#  Regex library.
# --------------------------------------------------------------------------- #
# Trainer-emitted peak-epoch tokens (post-training summary block).
_PEAK_RECALL_RX = re.compile(
    r"BEST_Val_Recall_Peak_Epoch\s*[:=]\s*(\d+)"
)
_PEAK_NDCG_RX = re.compile(
    r"BEST_Val_NDCG_Peak_Epoch\s*[:=]\s*(\d+)"
)
# Legacy token used by older training loops (kept for compatibility).
_LEGACY_BEST_EPOCH_RX = re.compile(
    r"best_epoch\s*[:=]\s*(\d+)"
)

# The P6.x driver banner is a single anchored line like::
#     [P6.2] cfg=<tag>  text_mode=...  ... seed=<int>
# We match ONLY that banner (not the ``Namespace(seed=<int>, ...)``
# dump the trainer prints, nor the ``[P6.x] <tag> seed=<int> wall=...``
# per-run summary line at the end -- both of these also contain
# ``seed=<int>`` but would produce spurious segment boundaries that
# leave the real peak-epoch tokens in the wrong bucket).
_SEED_BANNER_RX = re.compile(
    r"\[P6\.[0-9a-zA-Z_]+\]\s+cfg=\S+.*?seed\s*=\s*(?P<seed>\d+)",
    re.IGNORECASE,
)


def _peak_regex(peak_from: str) -> tuple[re.Pattern[str], ...]:
    """Return the ordered list of regex candidates for a peak source."""
    if peak_from == "recall":
        return (_PEAK_RECALL_RX, _LEGACY_BEST_EPOCH_RX)
    if peak_from == "ndcg":
        return (_PEAK_NDCG_RX, _LEGACY_BEST_EPOCH_RX)
    # either
    return (_PEAK_RECALL_RX, _PEAK_NDCG_RX, _LEGACY_BEST_EPOCH_RX)


def _find_peak_epoch(
    window: str, patterns: tuple[re.Pattern[str], ...]
) -> float:
    for pat in patterns:
        m = pat.search(window)
        if m:
            return float(m.group(1))
    return float("nan")


def _segment_by_seed(log_text: str) -> list[tuple[int, str]]:
    """Slice the log into ``(seed, window)`` chunks using seed banners.

    The window for seed ``s_k`` runs from the ``seed=s_k`` banner up
    to (but not including) the next seed banner. If no banner is
    present at all we return one wildcard chunk with seed=-1 covering
    the whole log -- appropriate for single-seed logs.
    """
    banners = list(_SEED_BANNER_RX.finditer(log_text))
    if not banners:
        return [(-1, log_text)]
    chunks: list[tuple[int, str]] = []
    for i, m in enumerate(banners):
        start = m.start()
        end = banners[i + 1].start() if i + 1 < len(banners) else len(log_text)
        chunks.append((int(m.group("seed")), log_text[start:end]))
    return chunks


def _pick_window_for_seed(
    seed: int, chunks: list[tuple[int, str]]
) -> str:
    """Return the concatenated log-window for a given seed.

    A run may contain multiple banners for the same seed (e.g. a
    driver retries). We concatenate all matching windows and let the
    peak-regex match the last occurrence (regex `search` returns the
    first, so we scan windows in reverse and stop at the first hit).
    """
    matching = [w for s, w in chunks if s == seed]
    if not matching:
        # Fall back to wildcard bucket if the driver did not stamp seeds.
        wildcard = [w for s, w in chunks if s == -1]
        return "\n".join(wildcard)
    # Prefer the last window (most recent run for that seed).
    return matching[-1]


def _agg(vals: list[float]) -> dict[str, float]:
    finite = [v for v in vals if not math.isnan(v)]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "n": 0.0}
    return {
        "mean": float(statistics.mean(finite)),
        "std":  float(statistics.stdev(finite)) if len(finite) > 1 else 0.0,
        "n":    float(len(finite)),
    }


def rehydrate(
    *,
    log_path: Path,
    json_path: Path,
    out_path: Path,
    peak_from: str,
    force: bool,
) -> dict[str, Any]:
    if not log_path.is_file():
        raise FileNotFoundError(f"log missing: {log_path}")
    if not json_path.is_file():
        raise FileNotFoundError(f"json missing: {json_path}")

    # Load JSON tolerantly (older payloads may embed NaN literals which
    # are not strict JSON but Python's json module accepts them).
    with json_path.open(encoding="utf-8") as fh:
        payload = json.load(fh, parse_constant=lambda x: float("nan"))

    with log_path.open(encoding="utf-8", errors="replace") as fh:
        log_text = fh.read()

    chunks = _segment_by_seed(log_text)
    patterns = _peak_regex(peak_from)

    per_seed = payload.get("per_seed", [])
    if not isinstance(per_seed, list) or not per_seed:
        raise ValueError(
            f"{json_path}: missing or empty 'per_seed' list."
        )

    n_patched = 0
    n_skipped = 0
    for row in per_seed:
        seed = int(row.get("seed", -1))
        cur = row.get("best_epoch", float("nan"))
        cur_f = float(cur) if isinstance(cur, (int, float)) else float("nan")
        if not force and not math.isnan(cur_f):
            n_skipped += 1
            continue
        window = _pick_window_for_seed(seed, chunks)
        peak = _find_peak_epoch(window, patterns)
        if math.isnan(peak):
            print(
                f"[rehydrate] seed={seed}: no peak token found "
                f"({peak_from}); leaving NaN.",
                file=sys.stderr,
            )
            continue
        row["best_epoch"] = peak
        n_patched += 1
        print(
            f"[rehydrate] seed={seed}: best_epoch NaN -> {int(peak)}  "
            f"(source={peak_from})",
            flush=True,
        )

    # Re-aggregate ranked[].best_epoch by tag.
    ranked = payload.get("ranked", [])
    for cell in ranked:
        tag = cell.get("tag")
        rows = [r for r in per_seed if r.get("tag") == tag]
        cell["best_epoch"] = _agg(
            [float(r.get("best_epoch", float("nan"))) for r in rows]
        )

    # Stamp provenance so downstream tools can audit.
    meta = payload.setdefault("meta", {})
    meta.setdefault("rehydration", {}).update({
        "script": "scripts/rehydrate_p6_json.py",
        "source_log": str(log_path),
        "peak_from": peak_from,
        "n_rows_patched": n_patched,
        "n_rows_skipped": n_skipped,
    })

    # NaN-safe emit: convert any remaining NaN floats to JSON ``null``
    # so the payload is strict-JSON-compliant. Downstream code that
    # reads with ``json.load(parse_constant=lambda _: float('nan'))``
    # or ``pandas.read_json`` will still see the missingness.
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float):
            return None if math.isnan(obj) or math.isinf(obj) else obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(_sanitize(payload), fh, indent=2, allow_nan=False)
    print(
        f"[rehydrate] wrote {out_path}  "
        f"(patched {n_patched}, skipped {n_skipped})",
        flush=True,
    )
    return payload


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Back-fill best_epoch in a P6.x result JSON from "
                     "its matching training log."))
    p.add_argument("--log", required=True, type=Path,
                   help="Path to the .txt training log.")
    p.add_argument("--json", required=True, type=Path,
                   help="Path to the .json payload to patch.")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional output path (default: overwrite --json).")
    p.add_argument("--peak_from", type=str, default="recall",
                   choices=["recall", "ndcg", "either"],
                   help=("Which trainer token to read as best_epoch. "
                         "'recall' (default) uses "
                         "BEST_Val_Recall_Peak_Epoch, 'ndcg' uses "
                         "BEST_Val_NDCG_Peak_Epoch, 'either' picks "
                         "whichever appears first per seed window."))
    p.add_argument("--force", action="store_true",
                   help=("Overwrite even if best_epoch is already finite. "
                         "By default only NaN cells are patched."))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_cli(argv)
    out_path = args.out if args.out is not None else args.json
    rehydrate(
        log_path=args.log.resolve(),
        json_path=args.json.resolve(),
        out_path=out_path.resolve(),
        peak_from=args.peak_from,
        force=bool(args.force),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
