"""
features.py — งาน B (ราคาปิด / return)
======================================
สร้าง Feature สำหรับงาน B เท่านั้น (ไม่มี feature กลุ่ม parity ของงาน A)

*** กฎเหล็กของไฟล์นี้ ***
Feature ที่มาจากตลาด (ราคา/ปริมาณ) ของแถววันที่ t ต้องคำนวณจากข้อมูล
"ถึงวันที่ t-1 เท่านั้น" ห้ามมีข้อมูลตลาดของวันที่ t หลุดเข้ามา

แบ่ง feature เป็น 2 กลุ่ม (A5):
  1. market   : ข้อมูลตลาดของวัน t ยังไม่รู้ก่อนทำนาย -> shift(1) ทั้งตาราง
                ทีเดียวตอนท้าย (ลืมไม่ได้) ชื่อคอลัมน์ลงท้าย _prev
  2. calendar : ปฏิทินของวัน t รู้ล่วงหน้าอยู่แล้ว -> ไม่ shift
                ชื่อคอลัมน์ไม่มี _prev

ทำไมต้องแยก: ถ้า shift dow ไปด้วย ทุกแถวจะได้ dow ของ "วันซื้อขายก่อนหน้า"
ในวันทำการปกติเป็นแค่การ relabel (จ.->อ. ...) แต่แถวที่มีวันหยุดคั่น
mapping จะแตกและให้ค่าที่ผิดจริง

ไม่มี feature ราคาดิบ (close/open/high/low/volume) (C1):
ราคาไม่ stationary ขณะที่ feature อื่นเป็นอัตราส่วนหรือ return ทั้งหมด
ต้นไม้ทำนายนอกช่วงที่เห็นตอน train ไม่ได้ และ StandardScaler แก้ไม่ได้
เพราะเป็นการแปลงเชิงเส้น -- เหตุผลเดียวกับที่ทำนาย return แทนราคาดิบ
"""

import numpy as np
import pandas as pd

from config import (
    LAG_DAYS, MA_WINDOWS, VOL_WINDOWS, RSI_PERIOD, USE_DAY_OF_WEEK,
)


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

def build_market_features(df):
    """
    feature ที่มาจากราคา/ปริมาณ -- ต้อง shift(1)
    (ฟังก์ชันนี้ยังมีข้อมูลวัน t อยู่ อย่าเอาไปเทรนตรงๆ)
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]

    f = pd.DataFrame(index=df.index)

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

    # --- ปริมาณซื้อขาย (อัตราส่วน ไม่ใช่ค่าดิบ) ---
    f["volume_change"] = volume.pct_change()
    f["volume_over_ma20"] = volume / volume.rolling(20).mean()

    return f


def build_calendar_features(df):
    """feature ปฏิทิน -- รู้ล่วงหน้า ไม่ต้อง shift"""
    c = pd.DataFrame(index=df.index)
    dow = df.index.dayofweek
    for d in range(5):
        c[f"dow_{d}"] = (dow == d).astype(int)
    return c


def build_features(df, verbose=True):
    """
    ฟังก์ชันที่ควรเรียกใช้จริง
    = market features shift(1) + calendar features (ไม่ shift)

    คืน DataFrame ที่ปลอดภัย ใช้เทรนได้เลย
    """
    market = build_market_features(df).shift(1)
    market.columns = [f"{c}_prev" for c in market.columns]

    if USE_DAY_OF_WEEK:
        calendar = build_calendar_features(df)       # ไม่ shift ไม่มี _prev
        out = pd.concat([market, calendar], axis=1)
    else:
        out = market

    if verbose:
        print(f"[features] สร้าง {out.shape[1]} features "
              f"(market {market.shape[1]} ตัว shift(1) แล้ว + "
              f"calendar {out.shape[1] - market.shape[1]} ตัวไม่ shift)")
    return out


def verify_no_leak(df, features, sample_idx=100):
    """
    ตรวจสอบเชิงโครงสร้าง 2 แบบ
      market  : features แถว t == raw indicator แถว t-1
      calendar: dow แถว t == วันในสัปดาห์ของวัน t เอง
    """
    raw = build_market_features(df)
    row_t, row_prev = features.iloc[sample_idx], raw.iloc[sample_idx - 1]

    checked = 0
    for col in raw.columns:                      # market: แถว t = raw แถว t-1
        name = f"{col}_prev"
        if name not in features.columns:
            continue
        a, b = row_t[name], row_prev[col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert np.isclose(a, b), (
            f"LEAK! คอลัมน์ {col}: features แถว {sample_idx} = {a} "
            f"แต่ raw แถว {sample_idx - 1} = {b}"
        )
        checked += 1
    assert checked > 0, "ไม่ได้ตรวจ market feature เลยสักตัว"

    for d in range(5):                           # calendar: แถว t = วันของ t
        name = f"dow_{d}"
        if name in features.columns:
            expected = int(features.index[sample_idx].dayofweek == d)
            assert row_t[name] == expected, f"dow ไม่ตรงวัน: {name}"

    print(f"[features] verify_no_leak ผ่าน ({checked} market cols + calendar)")
    return True
