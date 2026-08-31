"""
diagnostics.py — งาน C (ขึ้น/ลง)
================================
วิเคราะห์ข้อมูลก่อนเทรน  <-- ส่วนที่จะทำให้รายงานดูแข็งแรง

มี 2 การวิเคราะห์:
  1. updown_analysis - base rate ของงานขึ้น/ลง
  2. leak_check       - ตรวจหา feature ที่น่าสงสัยว่า leak
"""

import numpy as np
import pandas as pd

from config import LEAK_CORR_THRESHOLD


def updown_analysis(targets, verbose=True):
    """วิเคราะห์งาน C: ขึ้น/ลง"""
    y = targets["y_updown"].dropna()
    up_pct = float((y == 1).mean() * 100)
    flat_pct = float(targets["is_flat"].mean() * 100)

    stats = {"up_pct": up_pct, "down_pct": 100 - up_pct,
             "flat_pct": flat_pct}

    if verbose:
        print("\n--- การวิเคราะห์งาน C (ขึ้น/ลง) ---")
        print(f"ขึ้น {up_pct:.2f}% / ลงหรือเท่าเดิม {100-up_pct:.2f}%")
        print(f"วันราคานิ่งเป๊ะ: {flat_pct:.2f}%")
        print(f"=> Always-Up baseline: {up_pct:.2f}%")

    return stats


def leak_check(X, y_updown, verbose=True):
    """
    ตรวจหา feature ที่น่าสงสัยว่า leak

    วิธี: ดู correlation ระหว่าง feature กับ target ขึ้น/ลง
    ถ้า correlate สูงเกินไป -> น่าสงสัยว่ามีข้อมูลอนาคตหลุดเข้ามา
    """
    suspicious = []

    common = X.index.intersection(y_updown.dropna().index)
    Xc = X.loc[common]
    yu = y_updown.loc[common]

    if verbose:
        print("\n--- ตรวจสอบ Data Leakage ---")

    for col in Xc.columns:
        s = Xc[col]
        if s.nunique() <= 1:
            continue
        c = abs(s.corr(yu))
        c = c if pd.notna(c) else 0
        if c > LEAK_CORR_THRESHOLD:
            suspicious.append((col, c))

    if verbose:
        if suspicious:
            print(f"!! พบ {len(suspicious)} feature น่าสงสัย "
                  f"(|corr| > {LEAK_CORR_THRESHOLD}):")
            for col, c in sorted(suspicious, key=lambda x: -x[1]):
                print(f"   {col:35s} |corr| = {c:.3f}")
            print("   ตรวจสอบว่า feature พวกนี้ shift แล้วจริงหรือไม่")
        else:
            print(f"ผ่าน: ไม่มี feature ไหน correlate กับขึ้น/ลง "
                  f"เกิน {LEAK_CORR_THRESHOLD}")

    return suspicious


def run_all_diagnostics(df, targets, X, verbose=True):
    """รันการวิเคราะห์ทั้งหมดรวดเดียว"""
    print("\n" + "#" * 78)
    print("#  การวิเคราะห์ข้อมูลก่อนเทรน (เอาผลส่วนนี้ใส่รายงานด้วย)")
    print("#" * 78)

    out = {}
    out["updown"] = updown_analysis(targets, verbose)
    out["leak"] = leak_check(X, targets["y_updown"], verbose)
    return out
