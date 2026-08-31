"""
targets.py — งาน A (คู่/คี่)
===========================
สร้าง target ของงาน A เท่านั้น: ราคาปิดปัดเป็นจำนวนเต็มแล้วเป็นคู่(0)/คี่(1)
"""

import numpy as np
import pandas as pd

from rounding import to_int_baht, parity


def build_targets(df, verbose=True):
    """
    คืน DataFrame ที่มี target งาน A + คอลัมน์ช่วยเหลือ
    ทุกแถว t อ้างอิงราคาปิดของวัน t
    """
    close = df["Close"]
    prev_close = close.shift(1)

    t = pd.DataFrame(index=df.index)

    # คอลัมน์ช่วยเหลือ (ไม่ใช่ target แต่ต้องใช้ตอนแปลงผลกลับ)
    t["close"] = close
    t["prev_close"] = prev_close

    # ---------- งาน A : คู่ / คี่ ----------
    t["int_price"] = to_int_baht(close)
    t["y_parity"] = parity(t["int_price"])

    # parity ของเมื่อวาน (ใช้ทำ persistence baseline)
    t["prev_parity"] = t["y_parity"].shift(1)

    if verbose:
        print(f"[targets] สร้าง target งาน A (parity) เรียบร้อย")

    return t
