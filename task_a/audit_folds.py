"""
audit_folds.py  — ตรวจว่า walk-forward ล้ำเข้า test set หรือไม่
==============================================================
รันได้เลย ไม่ต้องเทรนอะไร ไม่แตะ test set

ที่มาของความสงสัย
-----------------
stage1_screening_results.csv รายงาน n = 1815 ทุกแถว

walk-forward 5 folds ในโค้ดเดิมแบ่งแบบนี้
    fs = n_pool // 6
    fold k ทำนายช่วง [fs*k, fs*(k+1))  สำหรับ k = 1..5
รวมแถวที่ถูกทำนาย = fs * 5 ~= n_pool * 5/6

ย้อนกลับ:  n_pool ~= 1815 * 6/5 = 2178 แถว
แต่ dev ที่ตั้งไว้ (85% ของ 2433) = 2068 แถว

2178 > 2068 อยู่ 110 แถว  ->  น่าสงสัยว่า fold สุดท้ายกินเข้าไปใน test
ถ้าจริง ผลทั้งหมดใน stage1 คือการประเมินที่ปนข้อมูล test ไปแล้ว
ต้องรู้ให้ชัดก่อนเขียนรายงาน

สคริปต์นี้ทำอะไร
  1. คำนวณขอบ dev/test จากสัดส่วนจริง (ไม่ hardcode)
  2. จำลองการแบ่ง fold ตามสูตรเดิม แล้วพิมพ์วันที่แรก-สุดท้ายของทุก fold
  3. บอกตรง ๆ ว่า fold ไหนล้ำขอบ และล้ำกี่แถว
  4. ย้อนคำนวณจาก n ที่รายงานไว้ ว่าตรงกับ pool ขนาดไหน

วิธีใช้
    python audit_folds.py
    python audit_folds.py --reported-n 1815 --n-folds 5
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from data_adapter import load_investing_csv
from tick_utils import DEV_RATIO, dev_cutoff, resolve_data_path

DEFAULT_WARMUP = 0


def simulate_folds(n_pool: int, n_folds: int = 5, scheme: str = "splits",
                   min_train: int = 250):
    """
    จำลองการแบ่ง fold คืน list ของ (k, train_end, pred_start, pred_end)

    scheme="splits"  -> สูตรของ splits.walk_forward_splits() ที่ main.py และ
                        experiment_matrix.py ใช้จริง
                            fold_size = (n - min_train) // n_splits
                            fold i: train[0:min_train + i*fold_size]
                                    predict[train_end : train_end + fold_size]
                        รวมแถวที่ทำนาย = n_folds * fold_size

    scheme="perm"    -> สูตรของ run_permutation.py เดิม (คนละสูตร!)
                            fs = n // (n_folds + 1)
                            fold k: predict[fs*k : fs*(k+1)]
                        รวมแถวที่ทำนาย = n_folds * fs
    """
    folds = []
    if scheme == "splits":
        fold_size = (n_pool - min_train) // n_folds
        for i in range(n_folds):
            train_end = min_train + i * fold_size
            pred_end = min(train_end + fold_size, n_pool)
            if pred_end <= train_end:
                break
            folds.append((i + 1, train_end, train_end, pred_end))
    else:
        fs = n_pool // (n_folds + 1)
        for k in range(1, n_folds + 1):
            a, b = fs * k, min(fs * (k + 1), n_pool)
            if b - a < 30:
                continue
            folds.append((k, a, a, b))
    return folds


def audit(df: pd.DataFrame, name: str, n_folds: int = 5,
          warmup: int = DEFAULT_WARMUP, reported_n: int | None = None,
          min_train: int = 250):
    n_raw = len(df)
    n_usable = n_raw - warmup            # หลังสร้าง feature แล้ว dropna
    cut = dev_cutoff(n_usable)           # ขอบ dev/test ที่ถูกต้อง

    usable_index = df.index[warmup:]
    dev_index = usable_index[:cut]
    test_index = usable_index[cut:]

    print("=" * 74)
    print(f"  ตรวจสอบขอบเขต fold : {name}")
    print("=" * 74)
    print(f"  แถวดิบทั้งหมด        : {n_raw}")
    print(f"  หักช่วง warmup       : {warmup} แถว")
    print(f"  แถวที่ใช้ได้จริง      : {n_usable}")
    print(f"  สัดส่วน dev          : {DEV_RATIO:.2f}")
    print(f"  ขอบ dev/test         : แถวที่ {cut}  "
          f"({dev_index[-1].date()} | {test_index[0].date()})")
    print(f"  DEV  : {len(dev_index):5d} แถว  "
          f"{dev_index[0].date()} -> {dev_index[-1].date()}")
    print(f"  TEST : {len(test_index):5d} แถว  "
          f"{test_index[0].date()} -> {test_index[-1].date()}  << ห้ามแตะ")

    # ---------- จำลองทั้งสองสูตร แล้วเทียบกับ n ที่รายงาน ----------
    verdict, matched = "UNKNOWN", None
    for scheme, label in [("splits", "splits.walk_forward_splits (main.py)"),
                          ("perm", "run_permutation.py เดิม")]:
        folds = simulate_folds(cut, n_folds, scheme, min_train)
        total = sum(b - a for _, _, a, b in folds)
        last_row = max(b for _, _, _, b in folds) if folds else 0
        overrun = last_row - cut
        mark = ""
        if reported_n and total == reported_n:
            matched, mark = scheme, "   <<< ตรงกับ n ที่รายงาน"
        print(f"\n  --- สูตร {label} ---")
        for k, train_end, a, b in folds:
            print(f"    fold {k}: train[0:{train_end:4d}] "
                  f"predict[{a:4d}:{b:4d}]  "
                  f"{dev_index[a].date()} -> {dev_index[b-1].date()}")
        print(f"    รวมแถวที่ถูกทำนาย = {total}{mark}")
        print(f"    fold สุดท้ายจบที่ index {last_row} / ขอบ dev {cut}"
              f"  -> {'ล้ำ ' + str(overrun) + ' แถว' if overrun > 0 else 'ไม่ล้ำ'}")

    print(f"\n  --- สรุป ---")
    if reported_n is None:
        print("    ไม่ได้ระบุ --reported-n จึงเทียบไม่ได้")
    elif matched:
        print(f"    n = {reported_n} ตรงกับสูตร '{matched}' พอดี")
        print(f"    fold ทั้งหมดอยู่ภายในขอบ dev ({cut} แถว)")
        print(f"    => ไม่ล้ำ test set")
        verdict = "OK"
    else:
        print(f"    !! n = {reported_n} ไม่ตรงกับสูตรไหนเลย")
        print(f"    !! ต้องไปดูโค้ดว่าแบ่ง fold ยังไง และ pool ใหญ่แค่ไหน")
        verdict = "CHECK"

    print()
    return {"name": name, "n_raw": n_raw, "n_usable": n_usable,
            "dev_cut": cut, "reported_n": reported_n, "verdict": verdict}


def find_inputs(pattern: str):
    files = sorted(glob.glob(pattern))
    for folder in ["raw_data", "../raw_data", "data", "../data"]:
        if not files:
            files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise SystemExit(
            f"ไม่พบไฟล์ตรงกับ {pattern!r}\n"
            f"ระบุด้วย --files KBANK_10Y_Cleaned.csv ADVANC_10Y_Cleaned.csv"
        )
    return files


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--glob", default="*_10Y_Cleaned.csv")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--min-train", type=int, default=250,
                    help="ต้องตรงกับ walk_forward_splits ใน splits.py")
    ap.add_argument("--reported-n", type=int, default=1815,
                    help="ค่า n ที่ปรากฏใน stage1_screening_results.csv")
    ap.add_argument("--date-format", default="%m/%d/%Y")
    args = ap.parse_args()

    paths = args.files or find_inputs(args.glob)

    summaries = []
    for path in paths:
        path = resolve_data_path(path)
        frame = load_investing_csv(path, date_format=args.date_format,
                                   verbose=False)
        label = os.path.basename(path).split("_")[0]
        summaries.append(
            audit(frame, label, args.n_folds, args.warmup,
                  args.reported_n, args.min_train)
        )

    print("=" * 74)
    print("  สรุป")
    print("=" * 74)
    print(pd.DataFrame(summaries).to_string(index=False))

    if any(s["verdict"] == "CHECK" for s in summaries):
        print("\n  ต้องทำต่อ: ไปดูโค้ดแบ่ง fold ให้แน่ใจว่า pool หยุดที่ dev")
    else:
        print("\n  ไม่พบการล้ำขอบ test set")