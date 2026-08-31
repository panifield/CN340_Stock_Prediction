"""
data_loader.py
==============
โหลดข้อมูลราคาหุ้น

- ใช้ yfinance ดึงจาก Yahoo Finance
- โหลดครั้งแรกแล้วเก็บเป็น csv ไว้ (ครั้งต่อไปไม่ต้องโหลดใหม่ เร็วขึ้นเยอะ)
- มีโหมดข้อมูลจำลองไว้เทสต์โค้ดตอนไม่มีเน็ต (ห้ามใช้ในรายงาน)
"""

import os
import numpy as np
import pandas as pd

from config import (
    START_DATE, END_DATE, CACHE_DIR, USE_SYNTHETIC_DATA, RANDOM_STATE
)

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(ticker):
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = ticker.replace("^", "").replace(".", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{START_DATE}_{END_DATE}.csv")


def download_from_yahoo(ticker, start=START_DATE, end=END_DATE):
    """ดึงข้อมูลจริงจาก Yahoo Finance"""
    import yfinance as yf

    # yfinance ตีความ end แบบ exclusive (ไม่รวมวันนั้น)
    # บวก 1 วันเพื่อให้ END_DATE ที่ตั้งไว้ถูกรวมอยู่ในข้อมูลจริง
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(
        ticker, start=start, end=end_exclusive,
        auto_adjust=False, progress=False,
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"โหลด {ticker} ไม่ได้ / ไม่มีข้อมูล")

    # yfinance รุ่นใหม่คืน MultiIndex column ต้องแบนก่อน
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[REQUIRED_COLS].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def make_synthetic(ticker="FAKE", n=2500, start_price=35.0, tick=0.25,
                   seed=RANDOM_STATE):
    """
    สร้างข้อมูลจำลอง (random walk + บังคับให้ราคาอยู่บน tick grid)
    ใช้เทสต์ pipeline เท่านั้น *** ห้ามใช้ในรายงาน ***
    """
    rng = np.random.default_rng(seed)

    rets = rng.normal(0.0003, 0.015, n)
    close = start_price * np.exp(np.cumsum(rets))
    close = np.round(close / tick) * tick          # บังคับลง tick grid

    noise = lambda scale: rng.normal(0, scale, n)
    open_ = np.round((close * (1 + noise(0.004))) / tick) * tick
    high = np.maximum(open_, close) * (1 + np.abs(noise(0.005)))
    low = np.minimum(open_, close) * (1 - np.abs(noise(0.005)))
    high = np.round(high / tick) * tick
    low = np.round(low / tick) * tick
    volume = rng.integers(1_000_000, 50_000_000, n)

    idx = pd.bdate_range(start="2015-01-02", periods=n, name="Date")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low,
         "Close": close, "Volume": volume},
        index=idx,
    )


def load_stock(ticker, use_cache=True, verbose=True):
    """
    ฟังก์ชันหลักที่ไฟล์อื่นเรียกใช้
    คืน DataFrame คอลัมน์ Open/High/Low/Close/Volume index เป็นวันที่
    """
    if USE_SYNTHETIC_DATA:
        if verbose:
            print(f"[data] !! ใช้ข้อมูลจำลองสำหรับ {ticker} "
                  f"(ห้ามใช้ในรายงาน) !!")
        # สร้างหุ้นราคาสูง/ต่ำต่างกัน เพื่อทดสอบว่างาน A ให้ผลต่างกันจริง
        if "HIGH" in ticker.upper():
            return make_synthetic(ticker, start_price=140.0, tick=0.50,
                                  seed=RANDOM_STATE + 1)
        return make_synthetic(ticker, start_price=18.0, tick=0.10,
                              seed=RANDOM_STATE)

    path = _cache_path(ticker)
    if use_cache and os.path.exists(path):
        if verbose:
            print(f"[data] อ่านจาก cache: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        if verbose:
            print(f"[data] กำลังโหลด {ticker} จาก Yahoo Finance ...")
        df = download_from_yahoo(ticker)
        df.to_csv(path)
        if verbose:
            print(f"[data] บันทึก cache ไว้ที่ {path}")

    df = clean(df, verbose=verbose)
    if verbose:
        print(f"[data] {ticker}: {len(df)} แถว "
              f"({df.index[0].date()} ถึง {df.index[-1].date()})")
    return df


def clean(df, verbose=True):
    """ทำความสะอาดข้อมูลเบื้องต้น"""
    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    before = len(df)
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    if verbose and len(df) < before:
        print(f"[data] ตัดแถวเสีย {before - len(df)} แถว")

    # วันที่ Volume = 0 มักเป็นวันหยุด/ข้อมูลผิด
    zero_vol = (df["Volume"] == 0).sum()
    if verbose and zero_vol > 0:
        print(f"[data] เตือน: มี {zero_vol} วันที่ Volume = 0")

    return df
