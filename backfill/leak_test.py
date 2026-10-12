"""
backfill/leak_test.py — พิสูจน์ว่า backfill ไม่แอบเห็นข้อมูลหลังเวลาตัด (data leak) ด้วยการ "วางยา" ข้อมูล

    python backfill/leak_test.py --dates 2026-09-29 2026-10-07

หลักการ: คัดลอกข้อมูลที่แคชไว้ (backfill/data/) แล้วทำให้ข้อมูล "หลังเวลาตัด" ผิดเพี้ยนมาก (ราคา x1.5, volume x5)
รัน backfill ใหม่ด้วยข้อมูลที่วางยาแล้ว (ออฟไลน์ -- ห้ามดาวน์โหลดใหม่) ถ้าไม่มี leak คำทำนายต้อง "เหมือนเดิมเป๊ะ"

  future  = วางยาเฉพาะข้อมูลที่ไม่ควรเห็น: next_day -> ตั้งแต่วัน target เป็นต้นไป ·
            t1600 -> แท่ง 16:00 ของวัน target และทุกอย่างหลังจากนั้น   => คาดหวัง: ผลเหมือนเดิมเป๊ะ
  control = วางยาข้อมูลที่ "ควรเห็น" (วันทำการก่อน target / แท่ง 15:00 ของ target) => คาดหวัง: ผลต้องเปลี่ยน
            (ยืนยันว่าการวางยาไวพอ ถ้า control ไม่เปลี่ยน แปลว่าการทดสอบนี้ใช้ไม่ได้)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import ROOT, TASKS, BACKFILL_DIR  # noqa: E402

LEAK_DIR = BACKFILL_DIR / "leak_test"
PRICE, VOL = 1.5, 5
PRED_COLS = ["predicted_flip_proba", "predicted_up_proba", "predicted_return", "predicted_close",
             "predicted_parity", "predicted_updown", "predicted_up_to_close", "price_1600", "cutoff_price", "prev_close"]


def _poison_frame(df, kind, target, mode):
    df = df.copy()
    if kind == "1h":
        idx = df.index.tz_convert("Asia/Bangkok") if df.index.tz else df.index.tz_localize("UTC").tz_convert("Asia/Bangkok")
        d, h = idx.normalize().tz_localize(None), idx.hour
        mask = (d > target) | ((d == target) & (h >= 16)) if mode == "future" else ((d == target) & (h == 15))
    else:
        d = pd.to_datetime(df.index).normalize()
        if mode == "future":
            mask = d >= target
        else:
            prev = d[d < target].max()
            mask = d == prev
    mask = np.asarray(mask)
    for c in df.columns:
        name = c[0] if isinstance(c, tuple) else c
        if name in ("Open", "High", "Low", "Close", "Adj Close"):
            df.loc[mask, c] = df.loc[mask, c] * PRICE
        elif name == "Volume":
            df.loc[mask, c] = df.loc[mask, c] * VOL
    return df, int(mask.sum())


def _poison_b_raw(path, target, mode):
    lines = Path(path).read_bytes().decode("ascii").split("\r\n")
    out, n = [lines[0]], 0
    dates = [pd.Timestamp(l.split(",")[0]) for l in lines[1:] if l]
    prev = max(d for d in dates if d < target)
    for l in lines[1:]:
        if not l:
            continue
        f = l.split(",")
        d = pd.Timestamp(f[0])
        hit = d >= target if mode == "future" else d == prev
        if hit:
            for i in (1, 2, 3, 4):
                f[i] = str(float(f[i]) * PRICE)
            f[5] = str(float(f[5]) * VOL)
            n += 1
        out.append(",".join(f))
    Path(path).write_bytes(("\r\n".join(out) + "\r\n").encode("ascii"))
    return n


def build_poisoned(mode, target):
    root = LEAK_DIR / f"{mode}_{target}"
    if root.exists():
        shutil.rmtree(root)
    data = root / "data"
    shutil.copytree(common.DATA_DIR / "yf_cache", data / "yf_cache")
    shutil.copytree(common.DATA_DIR / "task_b_raw_data", data / "task_b_raw_data")
    t = pd.Timestamp(target)
    n = 0
    for p in (data / "yf_cache").glob("*.pkl"):
        kind = "1h" if "_1h_" in p.name else "1d"
        df, k = _poison_frame(pd.read_pickle(p), kind, t, mode)
        pd.to_pickle(df, p)
        n += k
    for p in (data / "task_b_raw_data").glob("*.csv"):
        n += _poison_b_raw(p, t, mode)
    return root, n


def run(task, target, root):
    env = {**os.environ, "BACKFILL_DATA_DIR": str(root / "data"), "BACKFILL_RESULTS_DIR": str(root / "results"),
           "BACKFILL_OFFLINE": "1", "PYTHONUTF8": "1"}
    log = root / f"{task}.log"
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run([sys.executable, "-X", "utf8", str(BACKFILL_DIR / "run_one.py"), "--task", task,
                             "--target-date", target], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env).returncode
    return task, rc


def compare(task, target, root):
    key = ["ticker", "target_date", "model"]
    ref = pd.read_csv(common.out_path(task))
    ref = ref[ref.target_date == target]
    p = root / "results" / common.out_path(task).name
    if not p.exists():
        return None
    new = pd.read_csv(p)
    m = ref.merge(new, on=key, suffixes=("_ref", "_new"))
    cols = [c for c in PRED_COLS if f"{c}_ref" in m]
    diff = max(float((m[f"{c}_ref"] - m[f"{c}_new"]).abs().max()) for c in cols)
    return len(m), diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=["2026-09-29", "2026-10-07"])
    a = ap.parse_args()
    rows = []
    for target in a.dates:
        for mode in ("future", "control"):
            root, n = build_poisoned(mode, target)
            print(f"[{mode} {target}] วางยา {n} แถว/บรรทัดในข้อมูลที่แคชไว้", flush=True)
            with ThreadPoolExecutor(len(TASKS)) as ex:
                res = list(ex.map(lambda t: run(t, target, root), TASKS))
            for task, rc in res:
                c = compare(task, target, root) if rc == 0 else None
                rows.append({"target_date": target, "mode": mode, "task": task, "exit": rc,
                             "rows": c[0] if c else None, "max_abs_diff_vs_backfill": c[1] if c else None})
                print(rows[-1], flush=True)
    rep = pd.DataFrame(rows)
    rep["result"] = rep.apply(lambda r: ("ERROR" if r["exit"] != 0 else
                                         (("PASS (ไม่มี leak)" if r.max_abs_diff_vs_backfill == 0 else "FAIL (พบ leak)")
                                          if r["mode"] == "future" else
                                          ("OK (ผลเปลี่ยนตามที่คาด)" if r.max_abs_diff_vs_backfill > 0 else "ใช้ไม่ได้ (ไม่เปลี่ยน)"))), axis=1)
    rep.to_csv(LEAK_DIR / "leak_test_report.csv", index=False)
    print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
