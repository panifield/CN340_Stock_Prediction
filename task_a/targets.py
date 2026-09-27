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

    # parity ของเมื่อวาน (ใช้ทำ persistence baseline + reconstruct label)
    t["prev_parity"] = t["y_parity"].shift(1)

    # ---------- target ที่ใช้เทรนจริง : "พลิก parity หรือไม่" ----------
    # y_flip = 1 ถ้า parity วันนี้ไม่เท่าเมื่อวาน, 0 ถ้าเท่าเดิม
    # เท่ากับ (int_price_t - int_price_{t-1}) mod 2 แต่คำนวณตรงจาก
    # parity สองค่าเลย เพื่อเลี่ยงปัญหาจำนวนเต็มลบ mod ติดลบ
    # หมายเหตุ: นี่คือ label เต็มบาท ไม่ใช่ tick — ไม่ผูกกับ tick_size เลย
    # เพราะ "Δticks mod 2" จะเท่ากับ parity flip ก็ต่อเมื่อ tick = 1.00
    # เท่านั้น ในช่วง tick 0.25/0.50 มันคนละความหมายกัน ใช้ parity ตรงๆ
    # แม่นกว่าและพิสูจน์ถูกต้องได้ง่ายกว่า
    t["y_flip"] = (t["y_parity"] + t["prev_parity"]) % 2

    if verbose:
        print(f"[targets] สร้าง target งาน A (parity + y_flip) เรียบร้อย")

    return t
