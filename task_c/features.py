"""
features.py — งาน C (ขึ้น/ลง)
============================
สร้าง Feature สำหรับงาน C เท่านั้น (ไม่มี feature กลุ่ม parity ของงาน A)

*** กฎเหล็กของไฟล์นี้ ***
Feature ของแถววันที่ t ต้องคำนวณจากข้อมูล "ถึงวันที่ t-1 เท่านั้น"
ห้ามมีข้อมูลของวันที่ t หลุดเข้ามาแม้แต่นิดเดียว

ไม่มี raw close/open/high/low/volume: ราคาและ volume ดิบเป็น
non-stationary จึงใช้เฉพาะ feature ที่เป็นอัตราส่วนหรือผลตอบแทนแทน

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
    LAG_DAYS, MA_WINDOWS, VOL_WINDOWS, RSI_PERIOD, ATR_PERIOD,
    USE_DAY_OF_WEEK, WINSOR_LOWER_QUANTILE, WINSOR_UPPER_QUANTILE,
    WINSOR_MIN_PERIODS,
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


def normalized_atr(high, low, close, period=ATR_PERIOD):
    """Average True Range / close เพื่อให้เทียบข้ามระดับราคาได้"""
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(period).mean() / close.replace(0, np.nan)


def expanding_winsorize(series):
    """Clip outlier โดยใช้ percentile ของข้อมูลปัจจุบันและอดีตเท่านั้น."""
    lower = series.expanding(min_periods=WINSOR_MIN_PERIODS).quantile(
        WINSOR_LOWER_QUANTILE
    )
    upper = series.expanding(min_periods=WINSOR_MIN_PERIODS).quantile(
        WINSOR_UPPER_QUANTILE
    )
    return series.clip(lower=lower, upper=upper)


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
    bb_ma = close.rolling(20).mean()
    bb_width = 4 * close.rolling(20).std()
    f["bb_width"] = bb_width / bb_ma.replace(0, np.nan)
    f["atr_norm"] = normalized_atr(high, low, close)

    # --- Momentum / breakout ---
    direction = np.sign(close.diff())
    streak_group = direction.ne(direction.shift()).cumsum()
    # ขนาดของ streak ต้องอ่านคู่กับ prev_direction (1=ขึ้น, -1=ลง)
    f["up_streak"] = direction.groupby(streak_group).cumcount() + 1
    f["prev_direction"] = direction
    f["dist_from_high20"] = close / close.rolling(20).max() - 1
    f["dist_from_low20"] = close / close.rolling(20).min() - 1

    # --- ปริมาณซื้อขาย ---
    # ป้องกัน Volume(t-1)=0 ซึ่งจะทำให้ pct_change เป็น inf
    safe_volume = volume.replace(0, np.nan)
    f["volume_change"] = safe_volume.pct_change(fill_method=None)
    f["volume_change"] = f["volume_change"].replace([np.inf, -np.inf], np.nan)
    f["volume_over_ma20"] = volume / volume.rolling(20).mean()

    # --- วันในสัปดาห์ ---
    if USE_DAY_OF_WEEK:
        dow = df.index.dayofweek
        for d in range(5):
            f[f"dow_{d}"] = (dow == d).astype(int)

    # Winsorize เฉพาะตัวแปรที่เป็นการเปลี่ยนแปลง/return เพื่อลดผล outlier
    # Quantile แบบ expanding ไม่ใช้ข้อมูลในอนาคต และทั้งตารางจะถูก shift(1)
    # ใน build_features อีกชั้นหนึ่งก่อนถึงโมเดล
    winsor_cols = [f"ret_{lag}d" for lag in LAG_DAYS] + ["volume_change"]
    for col in winsor_cols:
        f[col] = expanding_winsorize(f[col])

    return f.replace([np.inf, -np.inf], np.nan)


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

    # raw ราคา/volume ต้องไม่กลับเข้ามาโดยไม่ตั้งใจ และห้ามมี inf เข้าสู่โมเดล
    forbidden = {"close_prev", "open_prev", "high_prev", "low_prev", "volume_prev"}
    assert forbidden.isdisjoint(features.columns), "พบ raw feature ที่ไม่ควรใช้"
    assert not np.isinf(features.to_numpy(dtype=float)).any(), "พบ inf ใน features"

    print("[features] verify_no_leak ผ่าน: feature แถว t = ข้อมูลวัน t-1 จริง "
          "และไม่มี raw price/volume หรือ inf")
    return True
