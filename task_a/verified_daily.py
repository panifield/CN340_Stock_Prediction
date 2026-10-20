"""
verified_daily.py — ราคารายวันที่ตรวจแล้ว สำหรับ predict_live ของ task_a / task_a2 / task_c
=========================================================================================

ประวัติมาจาก task_b/raw_data/ (Investing + Yahoo ที่ task_b ตรวจ OHLC ทุกวัน) ส่วน Yahoo ใช้แค่
ต่อวันที่ใหม่กว่าไฟล์ -- เดิมแต่ละ task ดึงประวัติเต็มจาก Yahoo เอง แต่ Yahoo 1d ปี 2024 มีแถวเสีย
(Volume=0 ราคาปิดค้าง ดู KNOWN_ISSUES.md 2026-10-17) ทำให้ป้ายคำตอบตอนเทรนผิด

ห้าม import โมดูลอื่นในโปรเจกต์: task_c โหลดไฟล์นี้ด้วย importlib จาก path ตรงๆ เพราะ task_c
มี config.py / features.py / data_loader.py ชื่อซ้ำกับ task_a
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "task_b" / "raw_data"
OVERLAP_DAYS = 10
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_verified_history(ticker, start=None):
    """OHLCV จาก task_b/raw_data/ (Volume เป็นหุ้น เหมือน Yahoo -- ไฟล์เก็บเป็นพันหุ้น)"""
    raw = pd.read_csv(RAW_DATA_DIR / f"{ticker.replace('.BK', '')}_10Y_Cleaned.csv")
    df = pd.DataFrame({
        "Open": pd.to_numeric(raw["Open"]),
        "High": pd.to_numeric(raw["High"]),
        "Low": pd.to_numeric(raw["Low"]),
        "Close": pd.to_numeric(raw["Price"]),
        "Volume": pd.to_numeric(raw["Vol. ('000)"]) * 1000,
    })
    df.index = pd.DatetimeIndex(pd.to_datetime(raw["Date"], format="%m/%d/%Y"), name="Date")
    if df.index.duplicated().any() or not df.index.is_monotonic_increasing:
        raise ValueError(f"raw_data ของ {ticker} มีวันซ้ำหรือเรียงไม่ถูก")
    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start)]
    return df


def fetch_verified_daily(ticker, start, end_exclusive):
    """ประวัติจาก raw_data + แถวใหม่จาก Yahoo ถึงก่อน end_exclusive (ยังไม่ clean)

    ดึงย้อนทับไฟล์ไว้ OVERLAP_DAYS วัน: ผลว่างจึงแปลว่าดึงไม่ได้จริง (ไม่ใช่แค่ไม่มีวันใหม่)
    และใช้ตรวจว่า Yahoo ตรงกับ raw_data · แถวใหม่ที่ Volume=0 (อาการเดียวกับแถวเสียปี 2024) ตัดทิ้ง
    """
    end_exclusive = pd.Timestamp(end_exclusive)
    if end_exclusive.tz is not None:
        end_exclusive = end_exclusive.tz_localize(None)
    hist = load_verified_history(ticker, start)
    hist = hist.loc[hist.index < end_exclusive]
    last = hist.index[-1]
    if last + pd.Timedelta(days=1) >= end_exclusive:
        return hist   # raw_data ครอบคลุมถึงวันที่ขอแล้ว (เช่น backtest ช่วงในอดีต) ไม่ต้องใช้ Yahoo

    df = yf.download(ticker, start=(last - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d"),
                     end=end_exclusive.strftime("%Y-%m-%d"),
                     auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"ดึงข้อมูลสดของ {ticker} ไม่ได้")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[COLUMNS].copy()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name="Date")

    overlap = df.index.intersection(hist.index)
    diff = (df.loc[overlap, "Close"] - hist.loc[overlap, "Close"]).abs()
    if (diff > 1e-6).any():
        bad = diff[diff > 1e-6]
        print(f"[live] !! Yahoo ราคาปิดไม่ตรง raw_data {len(bad)} วันในช่วงซ้อนทับ "
              f"({', '.join(str(d.date()) for d in bad.index)}) -- ใช้ค่าจาก raw_data")

    new = df.loc[df.index > last]

    # yfinance บางครั้งคืนแถวของวันล่าสุดมาแล้ว แต่ Close ยังเป็น NaN (ยังไม่ backfill แม้ตลาดปิด
    # ไปแล้ว) -- ผู้เรียก clean() ทิ้งเงียบๆ ทำให้ data_cutoff ถอยไปโดยไม่มีใครรู้ตัว เตือนตรงนี้
    n_trailing_nan = 0
    for v in new["Close"].iloc[::-1]:
        if pd.isna(v):
            n_trailing_nan += 1
        else:
            break
    if n_trailing_nan:
        dropped = new.index[-n_trailing_nan:]
        print(f"[live] !! yfinance ยังไม่มีราคาปิดของ {n_trailing_nan} วันล่าสุด "
              f"({dropped[0].date()} -> {dropped[-1].date()}, Close=NaN แถวเหล่านี้ "
              f"มักเกิดจาก backfill ช้าหลังตลาดปิด) -> data_cutoff จะถอยไปใช้วันก่อนหน้า"
              f"ที่มีราคาปิดจริง")

    zero_vol = new.index[new["Close"].notna() & (new["Volume"].fillna(0) <= 0)]
    if len(zero_vol):
        print(f"[live] !! Yahoo มีแถว Volume=0 {len(zero_vol)} วัน "
              f"({', '.join(str(d.date()) for d in zero_vol)}) -- ตัดทิ้ง ไม่ใช้")
        new = new.drop(zero_vol)

    return pd.concat([hist, new])
