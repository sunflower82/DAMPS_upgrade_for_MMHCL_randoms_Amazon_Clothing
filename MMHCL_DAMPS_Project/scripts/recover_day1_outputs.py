"""
Index completed Day-1 diagnostic runs (D2 / D3 / APC probe) under <log_root>/Clothing/.

Parses BEST_Test_Recall@20 / BEST_Test_NDCG@20 from per-run logs so you can reuse
results without re-launching multi-hour GPU jobs.  Intended log layout matches
``train.py`` / ``day1_diagnostic_sprint.expected_log_path``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
from typing import Any, Optional

_RE_RECALL = re.compile(r"BEST_Test_Recall@20[:=]\s*([\d.]+)")
_RE_NDCG = re.compile(r"BEST_Test_NDCG@20[:=]\s*([\d.]+)")


def _parse_best(path: pathlib.Path) -> tuple[Optional[float], Optional[float]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    m_r = _RE_RECALL.search(text)
    m_n = _RE_NDCG.search(text)
    r = float(m_r.group(1)) if m_r else None
    n = float(m_n.group(1)) if m_n else None
    return r, n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--log-root",
        type=pathlib.Path,
        default=None,
        help="Workspace root containing Clothing/ (default: parent of MMHCL_DAMPS_Project).",
    )
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="JSON manifest path (default: <log-root>/Clothing/day1_recovered_manifest.json).",
    )
    args = p.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    pkg = here.parent
    log_root = args.log_root or pkg.parent
    out_path = args.out or (log_root / "Clothing" / "day1_recovered_manifest.json")

    clothing = log_root / "Clothing"
    if not clothing.is_dir():
        print(f"[recover] no directory: {clothing}")
        return 1

    markers = ("diag_D2_", "diag_D3_", "diag_D1_apc")
    rows: list[dict[str, Any]] = []
    for d in sorted(clothing.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not any(m in name for m in markers):
            continue
        kind = "D2" if "diag_D2_" in name else "D3" if "diag_D3_" in name else "D1_apc"
        topk: Optional[int] = None
        if kind == "D2":
            m = re.search(r"diag_D2_topk(\d+)", name)
            topk = int(m.group(1)) if m else None
        log_file = d / f"{name}.txt"
        r20, n20 = _parse_best(log_file) if log_file.is_file() else (None, None)
        mm_sum = clothing / "MM" / f"sum_{name}.txt"
        rows.append(
            {
                "kind": kind,
                "topk": topk,
                "run_dir": str(d.relative_to(log_root)).replace("\\", "/"),
                "primary_log": str(log_file.relative_to(log_root)).replace("\\", "/"),
                "mm_sum": str(mm_sum.relative_to(log_root)).replace("\\", "/")
                if mm_sum.is_file()
                else None,
                "BEST_Test_Recall@20": r20,
                "BEST_Test_NDCG@20": n20,
                "log_exists": log_file.is_file(),
            }
        )

    rows.sort(key=lambda x: (x["kind"], x["topk"] or 0))
    payload = {
        "log_root": str(log_root.resolve()),
        "n_runs_indexed": len(rows),
        "runs": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[recover] wrote {out_path} ({len(rows)} diagnostic runs)")
    for row in rows:
        print(
            f"  {row['kind']:8} topk={str(row['topk']):>4}  R@20={row['BEST_Test_Recall@20']}  "
            f"NDCG@20={row['BEST_Test_NDCG@20']}  log={row['primary_log']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
