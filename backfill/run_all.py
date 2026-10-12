"""
backfill/run_all.py — หาช่องว่างใน log จริงตั้งแต่ 2026-09-28 แล้วทำนายย้อนหลังให้ครบ

    python backfill/run_all.py --dry-plan        # แค่แสดงว่าจะ backfill อะไรบ้าง
    python backfill/run_all.py                   # รันจริง (ข้ามช่องที่ backfill แล้ว -> รันซ้ำได้)
    python backfill/run_all.py --only task_a,task_c

ช่องว่าง = วันทำการ (ตามไฟล์วันหยุดของ task_b) ที่ log จริงมี 0 แถว (missing) หรือน้อยกว่า 6 แถว = 2 หุ้น × 3 โมเดล
(partial_models: log รอบแรกๆ ก่อนเปลี่ยนเป็นทำนาย 3 โมเดล) แต่ละ task รันทีละวันตามลำดับ ต่าง task รันขนานกัน
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import ROOT, TASKS  # noqa: E402

PY = sys.executable
LOG_DIR = (common.RESULTS_DIR.parent / "logs") if common.RESULTS_DIR != common.BACKFILL_DIR / "results" else common.BACKFILL_DIR / "logs"   # ทดลองด้วย BACKFILL_RESULTS_DIR -> log แยก ไม่ทับของจริง


def plan(only=None):
    days = common.trading_days()
    out = {}
    for task in TASKS:
        if only and task not in only:
            continue
        done = common.backfill_counts(task)
        out[task] = [(d, r) for d, r in common.find_gaps(task, days) if d not in done]
    return out


def run_task(task, items):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for d, reason in items:
        log = LOG_DIR / f"{task}_{d}.log"
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run([PY, "-X", "utf8", str(Path(__file__).parent / "run_one.py"),
                                 "--task", task, "--target-date", d, "--reason", reason],
                                stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        print(f"[{task}] {d} ({reason}) -> {'OK' if rc == 0 else 'FAIL exit ' + str(rc)}  (log: {log.name})", flush=True)
        results.append((task, d, rc))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-plan", action="store_true")
    ap.add_argument("--only", help="รายชื่อ task คั่นด้วย comma")
    a = ap.parse_args()
    only = a.only.split(",") if a.only else None

    p = plan(only)
    for task, items in p.items():
        print(f"{task:12s} {len(items):2d} วัน: " + ", ".join(f"{d[5:]}{'*' if r == 'partial_models' else ''}" for d, r in items))
    print("(* = log จริงมีไม่ครบ 3 โมเดล)")
    if a.dry_plan:
        return 0

    targets = [d for items in p.values() for d, _ in items]
    if any(t in p and p[t] for t in ("task_b",)) and targets:
        subprocess.run([PY, "-X", "utf8", str(Path(__file__).parent / "run_one.py"),
                        "--prepare-b-raw", "--through", max(targets)], cwd=ROOT, check=True)

    with ThreadPoolExecutor(max_workers=len(p)) as ex:
        futs = [ex.submit(run_task, t, items) for t, items in p.items() if items]
        res = [r for f in futs for r in f.result()]
    bad = [r for r in res if r[2] != 0]
    print(f"\nเสร็จ: สำเร็จ {len(res) - len(bad)}/{len(res)}" + (f" · ล้มเหลว {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
