"""
preds_dump.py — บันทึก P รายแถว (ใช้กับ docs/feedback: calibration / ROC)

ทำงานเฉพาะเมื่อตั้ง env PREDS_DUMP_DIR เท่านั้น ไม่แตะโมเดล/feature/การแบ่งข้อมูล
ไฟล์: <PREDS_DUMP_DIR>/<name>.csv  คอลัมน์
  y_parity (0 คู่ 1 คี่) · base_parity (prev_parity ของ A / parity_1600 ของ A2)
  y_flip · pred|<model> · p_odd|<model> · p_flip|<model> · คอลัมน์เสริมอื่น
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def dump_dir():
    return os.environ.get("PREDS_DUMP_DIR") or None


def dump_preds(name, y_true, base, preds=None, p_odd=None, extra=None):
    """p_odd: {model: P(คี่)} · P(พลิก) คำนวณจาก base: base=0 -> P(คี่), base=1 -> 1-P(คี่)"""
    d = dump_dir()
    if not d:
        return
    base = np.asarray(base, dtype=float)
    y = np.asarray(y_true, dtype=float)
    out = pd.DataFrame({"y_parity": y, "base_parity": base, "y_flip": (y + base) % 2},
                       index=pd.Index(y_true.index, name="date"))
    for n, p in (extra or {}).items():
        out[n] = np.asarray(p)
    for n, p in (preds or {}).items():
        out[f"pred|{n}"] = np.asarray(p, dtype=float)
    for n, p in (p_odd or {}).items():
        p = np.asarray(p, dtype=float)
        out[f"p_odd|{n}"] = p
        out[f"p_flip|{n}"] = np.where(base == 0, p, 1 - p)
    os.makedirs(d, exist_ok=True)
    out.to_csv(os.path.join(d, f"{name}.csv"), encoding="utf-8-sig")
    print(f"  [preds] {name}.csv ({len(out)} แถว)")
