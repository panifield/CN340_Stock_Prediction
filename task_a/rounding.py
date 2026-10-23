"""
rounding.py
===========
ฟังก์ชันปัดเศษตามกฎอาจารย์ + คำนวณคู่/คี่

*** ทำไมต้องเขียนเอง ไม่ใช้ round() ของ Python ? ***

Python ใช้ "banker's rounding" คือปัดไปหาเลขคู่ที่ใกล้ที่สุด:
    round(142.5) = 142   <- ปัดลง
    round(143.5) = 144   <- ปัดขึ้น
ทั้งสองกรณีได้ "เลขคู่" เสมอ!

ถ้าใช้ round() ตรงๆ ราคาที่ลงท้าย .50 ทุกตัวจะกลายเป็นเลขคู่หมด
ซึ่งเป็นการยัด bias เข้าไปใน target โดยตรง
หุ้นไทยที่ tick = 0.50 จะมีวันที่ราคาลงท้าย .50 เยอะมาก
=> label จะเอียงไปทาง "คู่" อย่างผิดธรรมชาติ

กฎอาจารย์คือ "เศษ > 0.5 ปัดขึ้น" ซึ่งไม่ตรงกับ round()
จึงต้องเขียนเอง และคำนวณในหน่วย "สตางค์" (จำนวนเต็ม)
เพื่อเลี่ยงปัญหา floating point (เช่น 142.50 อาจถูกเก็บเป็น 142.49999...)

*** ตั้งแต่ 2026-10-24 label คู่/คี่ใช้ parity_label() (floor) ***
to_int_baht() ด้านล่างเป็นกฎปัดเดิม เก็บไว้เทียบผลเก่าเท่านั้น
"""

import numpy as np
import pandas as pd

from config import ROUND_THRESHOLD, ROUND_MODE


def _to_satang(price):
    """ราคาเป็นสตางค์จำนวนเต็ม (ปัดเศษ float ทิ้งก่อน: 162.99999999 -> 16300)"""
    return np.rint(np.asarray(price, dtype=float) * 100)


def floor_baht(price):
    """ส่วนจำนวนเต็มบาทหลังแปลงเป็นสตางค์ (162.50 -> 162) — Series ถ้าใส่ Series มา"""
    if np.ndim(price) == 0:
        return np.nan if pd.isna(price) else int(_to_satang(price) // 100)
    s = pd.Series(price, dtype=float) if not isinstance(price, pd.Series) else price.astype(float)
    return pd.Series(np.floor(_to_satang(s.values) / 100), index=s.index, name="floor_baht")


def parity_label(price):
    """
    label คู่/คี่ของงาน A/A2: (สตางค์ // 100) % 2   0 = คู่, 1 = คี่
    แหล่งความจริงเดียว — ทุกที่ที่ต้องการคู่/คี่ของราคาต้องเรียกฟังก์ชันนี้

        162.0 -> 0 · 162.5 -> 0 · 163.0 -> 1 · 199.5 -> 1 · 162.99999999 -> 1 (= 163.00)

    scalar -> float, Series/array -> Series (NaN คงเป็น NaN)
    """
    fb = floor_baht(price)
    if np.ndim(price) == 0:
        return np.nan if pd.isna(fb) else float(fb % 2)
    return pd.Series(np.mod(fb.values, 2), index=fb.index, name="parity")


def to_int_baht_scalar(price, threshold=ROUND_THRESHOLD, mode=ROUND_MODE):
    """
    ปัดราคา 1 ค่า เป็นจำนวนเต็มบาท ตามกฎอาจารย์

    ตัวอย่าง (threshold=0.5, mode='gt'):
        142.30 -> 142
        142.50 -> 142   (0.50 ไม่ > 0.50)
        142.70 -> 143
    """
    if pd.isna(price):
        return np.nan

    # แปลงเป็นสตางค์ (จำนวนเต็ม) เพื่อเลี่ยง floating point error
    satang = int(np.rint(price * 100))
    baht, remainder = divmod(satang, 100)
    cut = threshold * 100

    if mode == "gt":
        return baht + 1 if remainder > cut else baht
    elif mode == "gte":
        return baht + 1 if remainder >= cut else baht
    else:
        raise ValueError("mode ต้องเป็น 'gt' หรือ 'gte'")


def to_int_baht(prices, threshold=ROUND_THRESHOLD, mode=ROUND_MODE):
    """
    เวอร์ชัน vectorized สำหรับ Series/array ทั้งชุด (เร็วกว่า apply มาก)
    """
    s = pd.Series(prices).astype(float)

    satang = np.rint(s * 100)
    baht = np.floor(satang / 100)
    remainder = satang - baht * 100
    cut = threshold * 100

    if mode == "gt":
        bump = (remainder > cut).astype(float)
    elif mode == "gte":
        bump = (remainder >= cut).astype(float)
    else:
        raise ValueError("mode ต้องเป็น 'gt' หรือ 'gte'")

    result = baht + bump
    result[s.isna()] = np.nan
    return pd.Series(result.values, index=s.index, name="int_baht")


def tick_size(prices):
    """
    ขนาด tick ของ SET ตามช่วงราคา (ต้องเช็คกับประกาศตลาดหลักทรัพย์อีกรอบ
    เพราะมีการปรับเป็นระยะ) ใช้เป็น feature เท่านั้น ไม่ใช่ตัวสร้าง label
    คู่/คี่ (label ยังคงมาจาก to_int_baht ตรงๆ เหมือนเดิม)

        < 2      -> 0.01
        2 - 5    -> 0.02
        5 - 10   -> 0.05
        10 - 25  -> 0.10
        25 - 100 -> 0.25
        100- 200 -> 0.50
        200- 400 -> 1.00
        >= 400   -> 2.00
    """
    s = pd.Series(prices).astype(float)
    bounds = [2, 5, 10, 25, 100, 200, 400]
    values = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
    conditions = [s < b for b in bounds]
    out = np.select(conditions, values[:-1], default=values[-1])
    out = pd.Series(out, index=s.index, name="tick_size")
    out[s.isna()] = np.nan
    return out


def parity(int_prices):
    """
    0 = เลขคู่ (even)
    1 = เลขคี่ (odd)
    """
    s = pd.Series(int_prices).astype("float")
    out = np.mod(s, 2)
    return pd.Series(out.values, index=s.index, name="parity")


def self_test():
    """
    ทดสอบว่าฟังก์ชันปัดทำงานตรงกฎอาจารย์จริง
    ถ้า assert ไม่ผ่าน แปลว่าโค้ดพัง อย่าเอาผลไปใช้
    """
    cases = [
        (142.00, 142),
        (142.30, 142),
        (142.49, 142),
        (142.50, 142),   # จุดสำคัญ: 0.50 ไม่ปัดขึ้น
        (142.51, 143),
        (142.75, 143),
        (142.99, 143),
        (143.50, 143),   # Python round() จะได้ 144 -> ต่างกัน!
        (15.10, 15),
        (15.60, 16),
    ]
    for price, expected in cases:
        got = to_int_baht_scalar(price, threshold=0.5, mode="gt")
        assert got == expected, f"ผิด: {price} ควรได้ {expected} แต่ได้ {got}"

    # ทดสอบเวอร์ชัน vectorized ให้ตรงกับเวอร์ชัน scalar
    arr = pd.Series([p for p, _ in cases])
    vec = to_int_baht(arr, threshold=0.5, mode="gt")
    for i, (price, expected) in enumerate(cases):
        assert vec.iloc[i] == expected, f"vectorized ผิดที่ {price}"

    # ทดสอบ parity
    assert parity(pd.Series([142, 143, 0, 1])).tolist() == [0, 1, 0, 1]

    print("[rounding] self_test ผ่านทั้งหมด")
    print("           หมายเหตุ: round(143.5) ของ Python =", round(143.5),
          "แต่กฎอาจารย์ได้ 143")


if __name__ == "__main__":
    self_test()
