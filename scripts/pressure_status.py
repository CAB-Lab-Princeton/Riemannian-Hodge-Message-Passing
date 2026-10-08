"""One line per run under runs/airfrans (and runs/airfrans/triage): epochs done, best / last validation R2, seconds
per epoch, and the test100 metrics of finished runs.

    python3 scripts/pressure_status.py [--root runs/airfrans] [--tasks AFP AFPi ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/airfrans")
    ap.add_argument("--tasks", nargs="*", default=None)
    a = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(a.root, "*", "*_s*")) + glob.glob(os.path.join(a.root, "triage", "*", "*_s*")))
    for d in dirs:
        task = os.path.basename(os.path.dirname(d))
        if a.tasks and task not in a.tasks:
            continue
        hp, rp = os.path.join(d, "history.json"), os.path.join(d, "result.json")
        if not os.path.exists(hp):
            continue
        try:
            h = json.load(open(hp))
        except ValueError:
            continue
        best = max(h, key=lambda r: r["R2"])
        line = (f"{os.path.relpath(d, a.root):46s} ep {len(h):3d}  val best {best['R2']:.4f}@{best['epoch']:<3d} "
                f"last {h[-1]['R2']:.4f}  loss {h[-1]['train_loss']:.5f}  {sum(r['time'] for r in h) / len(h):6.1f} s/ep")
        if os.path.exists(rp):
            R = json.load(open(rp))
            t = R.get("test100", {})
            if t:
                line += (f"  | test100 R2 {t['R2']:.4f} SSIM {t['SSIM']:.4f} P {t['Pearson']:.4f} NRMSE {t['NRMSE']:.4f}"
                         f" params {R.get('params')}")
            for k in ("fine_test100", "coarse_test100"):
                if k in R and "R2" in R[k]:
                    line += f" {k.split('_')[0]} {R[k]['R2']:.4f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
