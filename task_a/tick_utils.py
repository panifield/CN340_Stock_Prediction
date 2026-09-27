"""
tick_utils.py  — แก้ข้อ 5 และ 6
===============================
ฟังก์ชันที่ไฟล์อื่นใช้ร่วมกัน เพื่อไม่ให้มีตรรกะเดียวกันเขียนซ้ำหลายที่
แล้วเพี้ยนไปคนละทาง (ปัญหาเดิม: ATR ใน features.py กับ run_permutation.py
คำนวณคนละสูตร)

*** เรื่องการวางไฟล์ ***
ไฟล์นี้ออกแบบให้วางใน task_a/ ข้าง ๆ rounding.py ได้เลย

ถ้าวางใน task_a: จะ import to_int_baht / tick_size / parity จาก rounding.py
ที่มีอยู่แล้ว ไม่นิยามซ้ำ -> มีแหล่งความจริงเดียว
ถ้ารันเดี่ยว ๆ นอกโฟลเดอร์: จะใช้นิยามสำรองด้านล่างแทน (ตรรกะเดียวกันเป๊ะ)

ของใหม่ที่ไฟล์นี้เพิ่มเข้ามาจริง ๆ มี 3 อย่าง
  dev_cutoff()      ขอบ dev/test คำนวณจากสัดส่วน ไม่ hardcode   (ข้อ 6)
  atr_wilder()      ATR แบบ Wilder ที่ถูกต้อง นับ gap ด้วย        (ข้อ 5)
  month_end_flag()  ธงสิ้นเดือนที่ไม่บั๊ก                          (ข้อ 3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# สัดส่วนการแบ่ง — ต้องตรงกับ config.py ของ pipeline หลัก
# ถ้า import config ได้จะใช้ค่าจากที่นั่นแทน (แหล่งความจริงเดียว)
# ---------------------------------------------------------------
try:
    from config import TRAIN_RATIO, VAL_RATIO
except ImportError:
    TRAIN_RATIO, VAL_RATIO = 0.70, 0.15

DEV_RATIO = TRAIN_RATIO + VAL_RATIO


def dev_cutoff(n: int) -> int:
    """
    คืนจำนวนแถวของส่วน dev (train + val)

    แทนที่ DEV_FRACTION = 2068/2433 ที่ hardcode ไว้เดิม
    ถ้าโหลดข้อมูลใหม่แล้วจำนวนแถวเปลี่ยน ขอบจะขยับตามอัตโนมัติ
    ไม่ใช่ค้างอยู่ที่สัดส่วนของข้อมูลชุดเก่า
    """
    return int(n * DEV_RATIO)


def split_dev(obj):
    """ตัดเอาเฉพาะส่วน dev ของ DataFrame / Series / array"""
    cut = dev_cutoff(len(obj))
    return obj.iloc[:cut] if hasattr(obj, "iloc") else obj[:cut]


# ---------------------------------------------------------------
# การหาไฟล์ข้อมูลดิบ
# ---------------------------------------------------------------
# ไฟล์ csv อยู่ใน raw_data/ ซึ่งอาจอยู่ระดับเดียวกับ task_a/ หรือข้างใน
# ก็ได้ ตัวนี้ไล่หาให้เอง จะได้ไม่ต้องพิมพ์ path ยาว ๆ ทุกครั้ง
# และไม่พังถ้าย้ายโฟลเดอร์
_SEARCH_DIRS = [
    ".",
    "raw_data",
    "../raw_data",
    "../../raw_data",
    "data",
    "../data",
    "data_cache",
]


def resolve_data_path(name: str) -> str:
    """
    หาไฟล์ข้อมูลจากชื่อ คืน path เต็ม

    ถ้าส่ง path ที่มีอยู่จริงมาก็คืนกลับตรง ๆ
    ถ้าส่งแค่ชื่อไฟล์ จะไล่หาใน _SEARCH_DIRS ตามลำดับ
    """
    import os

    if os.path.isfile(name):
        return name

    base = os.path.basename(name)
    here = os.path.dirname(os.path.abspath(__file__))

    for root in (os.getcwd(), here):
        for folder in _SEARCH_DIRS:
            candidate = os.path.join(root, folder, base)
            if os.path.isfile(candidate):
                return os.path.normpath(candidate)

    searched = ", ".join(_SEARCH_DIRS)
    raise FileNotFoundError(
        f"ไม่พบไฟล์ {base!r}\n"
        f"ค้นหาใน: {searched}\n"
        f"(จาก {os.getcwd()} และ {here})\n"
        f"ถ้าไฟล์อยู่ที่อื่น ให้ส่ง path เต็มมาแทนชื่อไฟล์"
    )


def find_data_files(names) -> list[str]:
    """หาหลายไฟล์พร้อมกัน คืน list ของ path ที่เจอ"""
    return [resolve_data_path(n) for n in names]


# ---------------------------------------------------------------
# ยืมจาก rounding.py ถ้าอยู่ใน task_a  (แหล่งความจริงเดียว)
# ---------------------------------------------------------------
try:
    from rounding import to_int_baht, tick_size as _tick_size_series, parity
    _USING_ROUNDING_PY = True
except ImportError:
    _USING_ROUNDING_PY = False


# ---------------------------------------------------------------
# Tick size
# ---------------------------------------------------------------
_TICK_BOUNDS = [2, 5, 10, 25, 100, 200, 400]
_TICK_VALUES = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00]


def _tick_size_impl(price):
    """
    ขนาด tick ของ SET ตามช่วงราคา รับได้ทั้ง scalar และ Series

        < 2       -> 0.01        25 - 100  -> 0.25
        2 - 5     -> 0.02        100 - 200 -> 0.50
        5 - 10    -> 0.05        200 - 400 -> 1.00
        10 - 25   -> 0.10        >= 400    -> 2.00

    หมายเหตุ: ควรเช็คกับประกาศตลาดหลักทรัพย์อีกรอบ เพราะมีการปรับเป็นระยะ
    ใช้เป็น feature เท่านั้น ไม่ใช่ตัวสร้าง label
    """
    if np.isscalar(price):
        if pd.isna(price):
            return np.nan
        for bound, value in zip(_TICK_BOUNDS, _TICK_VALUES):
            if price < bound:
                return value
        return _TICK_VALUES[-1]

    s = pd.Series(price).astype(float)
    conditions = [s < b for b in _TICK_BOUNDS]
    out = pd.Series(
        np.select(conditions, _TICK_VALUES[:-1], default=_TICK_VALUES[-1]),
        index=s.index, name="tick_size",
    )
    out[s.isna()] = np.nan
    return out


def tick_size(price):
    """
    รับได้ทั้ง scalar และ Series

    rounding.py เวอร์ชันเดิมรับเฉพาะ Series (เรียกด้วย scalar จะ error)
    ตัวนี้ครอบให้รับ scalar ได้ด้วย เพื่อให้ data_adapter.validate()
    และโค้ดที่เรียกทีละค่าใช้ได้เหมือนกัน
    """
    if np.isscalar(price):
        if pd.isna(price):
            return np.nan
        if _USING_ROUNDING_PY:
            return float(_tick_size_series(pd.Series([float(price)])).iloc[0])
        return _tick_size_impl(price)

    if _USING_ROUNDING_PY:
        return _tick_size_series(price)
    return _tick_size_impl(price)


# ---------------------------------------------------------------
# การปัดเศษตามกฎอาจารย์
# ---------------------------------------------------------------
def _to_int_baht_impl(prices, threshold: float = 0.5, mode: str = "gt"):
    """
    ปัดเป็นจำนวนเต็มบาท คำนวณในหน่วยสตางค์เพื่อเลี่ยง floating point error

    mode="gt"  : เศษ > 0.50 ปัดขึ้น   -> 142.50 กลายเป็น 142
    mode="gte" : เศษ >= 0.50 ปัดขึ้น  -> 142.50 กลายเป็น 143

    *** ห้ามใช้ round() ของ Python ***
    Python ใช้ banker's rounding: round(142.5)=142 แต่ round(143.5)=144
    ได้เลขคู่เสมอ ซึ่งยัด bias เข้า label โดยตรง
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


def _parity_impl(int_prices):
    """0 = เลขคู่, 1 = เลขคี่"""
    s = pd.Series(int_prices).astype(float)
    return pd.Series(np.mod(s, 2).values, index=s.index, name="parity")


# ถ้าไม่มี rounding.py (รันเดี่ยว ๆ) ให้ใช้นิยามสำรองด้านบน
if not _USING_ROUNDING_PY:
    to_int_baht = _to_int_baht_impl
    parity = _parity_impl


# ---------------------------------------------------------------
# ATR
# ---------------------------------------------------------------
def atr_wilder(high, low, close, period: int = 14):
    """
    Average True Range แบบ Wilder's smoothing (สูตรที่ถูกต้อง)

    True Range = max ของสามค่า
        High - Low
        |High - Close เมื่อวาน|
        |Low  - Close เมื่อวาน|

    สองค่าหลังคือส่วนที่นับ gap ระหว่างวัน ถ้าใช้แค่ (High-Low).rolling().mean()
    จะได้ค่าที่ต่ำกว่าความจริงในวันที่ราคา gap แรง ๆ ซึ่งเป็นวันสำคัญพอดี
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def month_end_flag(index: pd.DatetimeIndex) -> pd.Series:
    """
    ธงวันสุดท้ายของเดือน/ไตรมาส

    *** อย่าเขียนแบบนี้ ***
        (idx.to_period("M") != idx.to_period("M").shift(-1))
    PeriodIndex.shift(-1) เลื่อน *ค่าของ period* ไปหนึ่งเดือน
    ไม่ใช่เลื่อนตำแหน่ง ผลคือทุกแถวไม่เท่ากันเสมอ -> ได้ 1 ทุกแถว
    """
    s = index.to_series()
    is_month_end = s.groupby(index.to_period("M")).transform("max") == s
    is_quarter_end = s.groupby(index.to_period("Q")).transform("max") == s
    return (is_month_end | is_quarter_end).astype(int)


def self_test():
    cases = [(142.00, 142), (142.30, 142), (142.49, 142), (142.50, 142),
             (142.51, 143), (142.75, 143), (143.50, 143), (15.60, 16)]
    got = to_int_baht([p for p, _ in cases], mode="gt")
    for i, (price, expected) in enumerate(cases):
        assert got.iloc[i] == expected, f"ผิดที่ {price}: ได้ {got.iloc[i]}"

    assert to_int_baht([142.50], mode="gte").iloc[0] == 143
    assert parity([142, 143]).tolist() == [0, 1]
    assert tick_size(150.0) == 0.50 and tick_size(250.0) == 1.00

    idx = pd.bdate_range("2024-01-01", periods=60)
    flags = month_end_flag(idx)
    assert flags.sum() < 10, f"month_end ผิด: ได้ {flags.sum()} จาก 60"

    source = "rounding.py (task_a)" if _USING_ROUNDING_PY else "นิยามสำรองในไฟล์นี้"
    print("[tick_utils] self_test ผ่านทั้งหมด")
    print(f"             แหล่งของ to_int_baht/tick_size/parity = {source}")
    print(f"             round(143.5) ของ Python = {round(143.5)} "
          f"แต่กฎอาจารย์ได้ 143")
    print(f"             month_end_flag: {flags.sum()} จาก {len(idx)} วัน "
          f"(เวอร์ชันเดิมที่บั๊กจะได้ {len(idx)})")


if __name__ == "__main__":
    self_test()
