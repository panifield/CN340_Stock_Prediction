"""
features.py — งาน A (คู่/คี่)
============================
สร้าง Feature สำหรับงาน A เท่านั้น (มี feature กลุ่ม parity เพิ่มจากงานอื่น)

*** กฎเหล็กของไฟล์นี้ ***
Feature ของแถววันที่ t ต้องคำนวณจากข้อมูล "ถึงวันที่ t-1 เท่านั้น"
ห้ามมีข้อมูลของวันที่ t หลุดเข้ามาแม้แต่นิดเดียว

วิธีที่ใช้:
  1. คำนวณ indicator ทั้งหมดตามปกติ (ใช้ข้อมูลถึงวัน t)
  2. shift(1) ทั้งตาราง ทีเดียวตอนท้าย
  => แถว t จะได้ค่า indicator ของวัน t-1

การ shift ทีเดียวตอนท้ายปลอดภัยกว่าการไล่ shift ทีละคอลัมน์
เพราะลืมไม่ได้
"""

import numpy as np
import pandas as pd

from config import (
    LAG_DAYS, MA_WINDOWS, VOL_WINDOWS, RSI_PERIOD,
    USE_DAY_OF_WEEK, USE_PARITY_LAGS, PARITY_LAG_DAYS,
)
from rounding import to_int_baht, parity


# ---------------------------------------------------------------
# Technical indicators (เขียนเอง ไม่ต้องลง TA-Lib)
# ---------------------------------------------------------------

def rsi(close, period=RSI_PERIOD):
    """Relative Strength Index (0-100)"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close, fast=12, slow=26, signal=9):
    """คืน (macd_line, signal_line, histogram)"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def bollinger_position(close, window=20, n_std=2):
    """
    ตำแหน่งราคาใน Bollinger Band
    0 = ขอบล่าง, 1 = ขอบบน
    """
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    width = (upper - lower).replace(0, np.nan)
    return (close - lower) / width


# ---------------------------------------------------------------
# ตัวสร้าง feature หลัก
# ---------------------------------------------------------------

def build_raw_features(df):
    """
    สร้าง feature ทั้งหมดโดย "ยังไม่ shift"
    (ฟังก์ชันนี้ยังมีข้อมูลวัน t อยู่ อย่าเอาไปเทรนตรงๆ)
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]

    f = pd.DataFrame(index=df.index)

    # --- ราคาดิบ ---
    f["close"] = close
    f["open"] = open_
    f["high"] = high
    f["low"] = low
    f["volume"] = volume

    # --- ผลตอบแทน (return) : ตัวสำคัญที่สุด เพราะเป็น stationary ---
    for lag in LAG_DAYS:
        f[f"ret_{lag}d"] = close.pct_change(lag)

    # --- รูปทรงแท่งเทียน (normalize ด้วยราคา -> ไม่มีปัญหา scale) ---
    f["hl_range"] = (high - low) / close
    f["oc_change"] = (close - open_) / open_
    f["close_pos_in_range"] = (close - low) / (high - low).replace(0, np.nan)

    # --- เส้นค่าเฉลี่ย: ใช้ "อัตราส่วน" ไม่ใช่ค่าดิบ ---
    # เหตุผล: ค่าดิบของ MA จะโตตามราคา ทำให้ tree model extrapolate ไม่ได้
    for w in MA_WINDOWS:
        sma = close.rolling(w).mean()
        ema = close.ewm(span=w, adjust=False).mean()
        f[f"close_over_sma{w}"] = close / sma
        f[f"close_over_ema{w}"] = close / ema
    if len(MA_WINDOWS) >= 2:
        a, b = MA_WINDOWS[0], MA_WINDOWS[-1]
        f[f"sma{a}_over_sma{b}"] = (close.rolling(a).mean()
                                    / close.rolling(b).mean())

    # --- ความผันผวน ---
    ret1 = close.pct_change()
    for w in VOL_WINDOWS:
        f[f"volatility_{w}d"] = ret1.rolling(w).std()

    # --- โมเมนตัม ---
    f["rsi"] = rsi(close)
    m_line, m_sig, m_hist = macd(close)
    f["macd"] = m_line / close          # normalize ด้วยราคา
    f["macd_signal"] = m_sig / close
    f["macd_hist"] = m_hist / close
    f["bb_position"] = bollinger_position(close)

    # --- ปริมาณซื้อขาย ---
    f["volume_change"] = volume.pct_change()
    f["volume_over_ma20"] = volume / volume.rolling(20).mean()

    # --- วันในสัปดาห์ ---
    if USE_DAY_OF_WEEK:
        dow = df.index.dayofweek
        for d in range(5):
            f[f"dow_{d}"] = (dow == d).astype(int)

    # --- feature เฉพาะงานคู่/คี่ ---
    # parity มัน "เหนียว" (วันนี้มักเท่ากับเมื่อวาน ถ้าราคาขยับน้อยกว่า 1 บาท)
    # ต้องใส่เข้าไปให้โมเดลเห็น ไม่งั้นโมเดลจับ pattern นี้ไม่ได้เลย
    if USE_PARITY_LAGS:
        int_price = to_int_baht(close)
        par = parity(int_price)
        f["parity_now"] = par
        f["int_price_change"] = int_price.diff()
        for lag in PARITY_LAG_DAYS:
            if lag == 1:
                continue  # parity_lag1 == parity_now (shift(0)) ซ้ำกัน ข้าม
            f[f"parity_lag{lag}"] = par.shift(lag - 1)
        # ระยะห่างจากจุดที่จะพลิก parity (เศษสตางค์)
        f["frac_part"] = close - np.floor(close)

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
        print(f"[features] สร้าง {shifted.shape[1]} features "
              f"(shift(1) แล้ว = ใช้ข้อมูลถึงวัน t-1 เท่านั้น)")
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

    # ตรวจซ้ำ: ราคาปิดของวัน t ต้องไม่เท่ากับ feature ตัวไหนเลย
    close_t = df["Close"].iloc[sample_idx]
    close_prev = df["Close"].iloc[sample_idx - 1]
    assert np.isclose(features["close_prev"].iloc[sample_idx], close_prev)
    if not np.isclose(close_t, close_prev):
        assert not np.isclose(features["close_prev"].iloc[sample_idx], close_t)

    print("[features] verify_no_leak ผ่าน: feature แถว t = ข้อมูลวัน t-1 จริง")
    return True
