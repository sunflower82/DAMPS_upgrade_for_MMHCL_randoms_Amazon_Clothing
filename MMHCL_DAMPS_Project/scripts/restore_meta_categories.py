"""
Copy ``meta_categories.npy`` into the DAMPS dataset tree (Phase-1 precondition).

Default destination: ``<workspace>/data/<Dataset>/meta_categories.npy`` (matches
``utility/load_data._load_metadata_categories``).

Typical MMHCL paper checkout layout uses ``data/clothing/`` (lowercase) while this
trainer defaults to ``dataset=Clothing`` — the script normalizes case on Windows.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

# Allow importing infer_n_items_users from the sibling sprint script
_PKG = pathlib.Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

try:
    from scripts.day1_diagnostic_sprint import infer_n_items_users  # noqa: E402
except ImportError:  # pragma: no cover
    from day1_diagnostic_sprint import infer_n_items_users  # noqa: E402


def _pick_source(candidates: list[pathlib.Path]) -> pathlib.Path | None:
    for c in candidates:
        if c.is_file():
            return c
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--workspace",
        type=pathlib.Path,
        default=None,
        help="DAMPS workspace root (contains data/ and MMHCL_DAMPS_Project/). "
        "Default: parent of MMHCL_DAMPS_Project.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="Clothing",
        help="Dataset folder name under data/ (default: Clothing).",
    )
    p.add_argument(
        "--core",
        type=int,
        default=5,
        help="K-core suffix for n_items inference (default: 5).",
    )
    p.add_argument(
        "--source",
        type=pathlib.Path,
        action="append",
        default=[],
        help="Explicit path to meta_categories.npy (repeatable). Tried first.",
    )
    args = p.parse_args()

    pkg = pathlib.Path(__file__).resolve().parent.parent
    workspace = args.workspace or pkg.parent
    data_path = workspace / "data"
    dest_dir = data_path / args.dataset
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "meta_categories.npy"

    default_candidates: list[pathlib.Path] = []
    for rel in (
        pathlib.Path(
            r"C:\Users\Anh Khoi\Desktop\MyCode\Saved research codes"
            r"\5090_Professional_Original_paper_MMHCL_randoms\data\Clothing\meta_categories.npy"
        ),
        pathlib.Path(
            r"C:\Users\Anh Khoi\Desktop\MyCode\Saved research codes"
            r"\5090_Professional_Original_paper_MMHCL_randoms\data\clothing\meta_categories.npy"
        ),
        data_path / "Clothing" / "meta_categories.npy",
        data_path / "clothing" / "meta_categories.npy",
    ):
        default_candidates.append(rel)

    src = _pick_source([pathlib.Path(s) for s in args.source] + default_candidates)
    if src is None:
        print(
            "[restore-meta] meta_categories.npy not found in default search paths.\n"
            "  Download the MMHCL authors' Google Drive data bundle, extract "
            "meta_categories.npy for Clothing, then run:\n"
            f"  python scripts/restore_meta_categories.py --source <path/to/meta_categories.npy>\n"
            f"  (cwd should be MMHCL_DAMPS_Project/; workspace inferred as {workspace})"
        )
        return 1

    shutil.copy2(src, dest)
    print(f"[restore-meta] copied\n  from: {src}\n  to  : {dest}")

    try:
        import numpy as np

        arr = np.load(dest).astype(np.int64).reshape(-1)
        n_items, _ = infer_n_items_users(str(data_path) + os.sep, args.dataset, args.core)
        print(f"[restore-meta] len(meta)={len(arr):,}  inferred n_items={n_items:,}")
        if len(arr) != n_items:
            print(
                "[restore-meta] WARNING: length mismatch — APC may still fall back; "
                "verify you used the 5-core Clothing release."
            )
            return 2
    except Exception as exc:  # pragma: no cover
        print(f"[restore-meta] post-check skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
