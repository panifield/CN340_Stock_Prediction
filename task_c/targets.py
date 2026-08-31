"""
targets.py — งาน C (ขึ้น/ลง)
============================
สร้าง target ของงาน C เท่านั้น: ขึ้น(1) หรือ ลง/เท่าเดิม(0)
"""

import numpy as np
import pandas as pd


def build_targets(df, verbose=True):
    """
    คืน DataFrame ที่มี target งาน C + คอลัมน์ช่วยเหลือ
    ทุกแถว t อ้างอิงราคาปิดของวัน t
    """
    close = df["Close"]
    prev_close = close.shift(1)

    t = pd.DataFrame(index=df.index)

    # คอลัมน์ช่วยเหลือ (ไม่ใช่ target แต่ต้องใช้ตอนแปลงผลกลับ)
    t["close"] = close
    t["prev_close"] = prev_close

    # ---------- งาน C : ขึ้น / ลง ----------
    # แถวแรกไม่มี prev_close (NaN) ให้เทียบ -> ต้องเป็น NaN ไม่ใช่ 0 (ลง)
    t["y_updown"] = np.where(
        prev_close.isna(), np.nan, (close > prev_close).astype(float)
    )
    # วันที่ราคาเท่าเดิมเป๊ะ (สำคัญ! หุ้นสภาพคล่องต่ำมีเยอะ)
    t["is_flat"] = (close == prev_close).astype(int)

    if verbose:
        n = len(t)
        n_flat = int(t["is_flat"].sum())
        print(f"[targets] สร้าง target งาน C (ขึ้น/ลง) เรียบร้อย")
        print(f"[targets] วันที่ราคาปิดเท่าเดิมเป๊ะ: {n_flat} วัน "
              f"({n_flat/n*100:.2f}%)")
        if n_flat / n > 0.05:
            print("[targets] !! เตือน: วันราคานิ่งเกิน 5% "
                  "ควรพิจารณาตัดทิ้งหรือทำเป็น 3 คลาส !!")

    return t


def drop_flat_days(X, y, targets, verbose=True):
    """
    ตัดวันที่ราคาปิดเท่าเดิมออก (ใช้ถ้าต้องการ)
    เพราะวันพวกนี้ไม่ใช่ทั้ง "ขึ้น" และ "ลง"
    """
    mask = targets.loc[X.index, "is_flat"] == 0
    if verbose:
        print(f"[targets] ตัดวันราคานิ่งออก {(~mask).sum()} แถว")
    return X[mask], y[mask]
