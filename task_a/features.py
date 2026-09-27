"""
features.py — งาน A (คู่/คี่)
============================
สร้าง Feature สำหรับงาน A เท่านั้น — ทำงานในหน่วย "tick" ทั้งหมด
(ตัด RSI/MACD/Bollinger/MA/ราคาดิบ/return % ทิ้ง เพราะไม่เกี่ยวกับ parity)

เหตุผลที่ใช้หน่วย tick แทนหน่วยบาท/เปอร์เซ็นต์:
parity ขึ้นกับ "ราคาขยับกี่ tick" ไม่ใช่ "ขยับกี่ %" ฟีเจอร์ที่วัดเป็น
บาทดิบหรือ % จึงแทบไม่มีข้อมูลเกี่ยวกับ parity เลย มีแต่ noise

*** กฎเหล็กของไฟล์นี้ ***
Feature ของแถววันที่ t ต้องคำนวณจากข้อมูล "ถึงวันที่ t-1 เท่านั้น"
ห้ามมีข้อมูลของวันที่ t หลุดเข้ามาแม้แต่นิดเดียว

วิธีที่ใช้: คำนวณ indicator ทั้งหมดตามปกติ (ใช้ข้อมูลถึงวัน t) แล้ว
shift(1) ทั้งตาราง ทีเดียวตอนท้าย ปลอดภัยกว่าไล่ shift ทีละคอลัมน์
"""

import numpy as np
import pandas as pd

from config import (
    DTICK_LAGS, DTICK_PARITY_LAGS, ZERO_RATE_WINDOW,
    ATR_PERIOD, STD_WINDOW, VOLUME_MA_WINDOW, BOUNDARY_LEVELS,
)
from rounding import to_int_baht, tick_size


def atr(high, low, close, period=ATR_PERIOD):
    """Average True Range แบบ Wilder's smoothing"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def build_raw_features(df):
    """
    สร้าง feature ทั้งหมดโดย "ยังไม่ shift"
    (ฟังก์ชันนี้ยังมีข้อมูลวัน t อยู่ อย่าเอาไปเทรนตรงๆ)
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    tick = tick_size(close)
    n = np.round(close / tick)          # ราคาเป็นหน่วย tick (จำนวนเต็ม)
    dtick = n.diff()                    # Δticks ดิบ (วันต่อวัน)

    f = pd.DataFrame(index=df.index)

    # ---------- A. State ปัจจุบัน ----------
    f["n_mod2"] = n % 2
    satang = np.rint(close * 100) % 100
    # 0=.00  1=.25  2=.50  3=.75  4=ลงท้ายอย่างอื่น
    f["dec_flag"] = np.select(
        [satang == 0, satang == 25, satang == 50, satang == 75],
        [0, 1, 2, 3],
        default=4,
    )
    f["tick_regime"] = tick
    boundaries = np.array(BOUNDARY_LEVELS, dtype=float)
    dist_baht = np.min(
        np.abs(close.values[:, None] - boundaries[None, :]), axis=1
    )
    f["dist_to_boundary"] = pd.Series(dist_baht, index=df.index) / tick

    # ---------- B. ประวัติการขยับ ----------
    for lag in DTICK_LAGS:
        # หลัง shift(1) รวมท้ายไฟล์อีกที -> ได้ Δticks ของ (t - lag)
        f[f"dtick_lag{lag}"] = dtick.shift(lag - 1)

    dtick_parity = dtick % 2
    for lag in DTICK_PARITY_LAGS:
        f[f"dtick_parity_lag{lag}"] = dtick_parity.shift(lag - 1)

    f["zero_rate_20"] = (dtick == 0).rolling(ZERO_RATE_WINDOW).mean()

    # ---------- C. ความผันผวน (หน่วย tick เท่านั้น) ----------
    f["atr14_ticks"] = atr(high, low, close) / tick
    f["std20_ticks"] = dtick.rolling(STD_WINDOW).std()
    f["range_ticks_lag1"] = (high - low) / tick

    # ---------- D. ปริมาณ + ปฏิทิน ----------
    vol_ma = volume.rolling(VOLUME_MA_WINDOW).mean()
    f["vol_ratio"] = np.log(volume / vol_ma.replace(0, np.nan))

    f["dow"] = df.index.dayofweek

    month = df.index.to_period("M")
    quarter = df.index.to_period("Q")
    idx_series = df.index.to_series()
    is_month_end = idx_series.groupby(month).transform("max") == idx_series
    is_quarter_end = idx_series.groupby(quarter).transform("max") == idx_series
    f["is_month_end"] = (is_month_end | is_quarter_end).astype(int)

    return f


def build_features(df, verbose=True):
    """
    ฟังก์ชันที่ควรเรียกใช้จริง
    = build_raw_features แล้ว shift(1) ทั้งตาราง

    คืน DataFrame ที่ปลอดภัย ใช้เทรนได้เลย
    """
    raw = build_raw_features(df)
    shifted = raw.shift(1)
    shifted.columns = [f"{c}_prev" for c in shifted.columns]

    if verbose:
        print(f"[features] สร้าง {shifted.shape[1]} features (หน่วย tick) "
              f"shift(1) แล้ว = ใช้ข้อมูลถึงวัน t-1 เท่านั้น")
    return shifted


def verify_no_leak(df, features, sample_idx=100):
    """
    ตรวจสอบเชิงโครงสร้างว่า shift ทำงานจริง
    เทียบว่า features แถว t == raw indicator แถว t-1 จริงไหม
    """
    raw = build_raw_features(df)
    row_t = features.iloc[sample_idx]
    row_prev = raw.iloc[sample_idx - 1]

    for col in raw.columns:
        a = row_t[f"{col}_prev"]
        b = row_prev[col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert np.isclose(a, b, equal_nan=True), (
            f"LEAK! คอลัมน์ {col}: features แถว {sample_idx} = {a} "
            f"แต่ raw แถว {sample_idx-1} = {b}"
        )

    print("[features] verify_no_leak ผ่าน: feature แถว t = ข้อมูลวัน t-1 จริง")
    return True
