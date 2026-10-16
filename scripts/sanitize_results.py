#!/usr/bin/env python3
"""Strip machine-specific paths from result files before they are added to ``results/``.

    python3 scripts/sanitize_results.py FILE.json [FILE.json ...] [--out-dir DIR] [--check]

The trainer and the evaluation scripts record where a run lived (``run``, ``run_dir``, ``src``, ``out`` ...) and where
its data came from (``data.meta.source``).  This script rewrites a JSON file so that no absolute path and no path of
another machine remains:

* string values that are absolute paths (``/...``, ``~/...``) are shortened to their part inside the repository when
  they contain one of the repository directories (``datasets/``, ``runs/``, ``results/``), e.g.
  ``/home/user/checkout/datasets/v2/AFM.pt`` -> ``datasets/v2/AFM.pt``; otherwise the key is removed;
* the keys ``run``, ``run_dir``, ``src``, ``_src`` and ``checkpoint`` are always removed when their value names a
  directory outside the repository layout;
* ``host:path`` strings (``cab:...``) are removed.

Relative paths (``runs/...``, ``datasets/...``), hardware names (``gpu``, ``device``) and every number are kept.  With
``--out-dir`` the cleaned files are written there (same base names), otherwise in place; ``--check`` only reports the
offending values and exits with status 1 if any are found.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

PATH_KEYS = {"run", "run_dir", "src", "_src", "checkpoint"}
REPO_DIRS = ("datasets/", "runs/", "results/")
HOST_RE = re.compile(r"^[A-Za-z0-9_.-]+:[^ ]*/")      # host:path/to/...


def _looks_like_path(s: str) -> bool:
    return s.startswith("/") or s.startswith("~/") or s.startswith("../") or bool(HOST_RE.match(s))


def _shorten(s: str) -> str | None:
    """Repository-relative form of an absolute path, or None if it points outside the repository layout."""
    for d in REPO_DIRS:
        i = s.find("/" + d)
        if i >= 0:
            return s[i + 1:]
    return None


def sanitize(obj, path: str = "", report: list | None = None):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            here = f"{path}/{k}"
            if isinstance(v, str) and (_looks_like_path(v) or (k in PATH_KEYS and "/" in v and not v.startswith(tuple(REPO_DIRS)))):
                short = _shorten(v) if not HOST_RE.match(v) else None
                if report is not None:
                    report.append((here, v, short))
                if short is None or k in PATH_KEYS:
                    continue                      # drop the key
                out[k] = short
            else:
                out[k] = sanitize(v, here, report)
        return out
    if isinstance(obj, list):
        return [sanitize(v, f"{path}[]", report) for v in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--check", action="store_true", help="report offending values only (exit 1 if any)")
    a = ap.parse_args()
    found = 0
    for fn in a.files:
        with open(fn) as f:
            data = json.load(f)
        report: list = []
        clean = sanitize(data, "", report)
        found += len(report)
        for here, v, short in report:
            print(f"{fn}: {here} = {v!r} -> {'removed' if short is None else short!r}")
        if a.check:
            continue
        out = fn if a.out_dir is None else os.path.join(a.out_dir, os.path.basename(fn))
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(clean, f, indent=1)
            f.write("\n")
    if a.check and found:
        sys.exit(1)


if __name__ == "__main__":
    main()
