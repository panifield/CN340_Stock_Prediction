"""
check_settrade_atc_match.py — Task C (ตรวจสอบเท่านั้น ไม่ใช่ pipeline)
=======================================================================
คำถามที่ค้างอยู่ใน KNOWN_ISSUES.md (task_c ข้อ 2): แท่ง 1h ที่ Hour == 16
ใน data_cache/{ticker}_1h_730d.csv (มาจาก yfinance) ที่ intraday_task2.py
ใช้เป็น target ("Close 16:00") ตรงกับราคาปิดทางการ (ATC) จริงหรือไม่ —
เคยมีแต่ task_a2 ที่ตรวจเรื่องนี้แล้ว (check_settrade_atc_match.py ของมัน
เทียบแท่ง Settrade 1m ล่าสุดของวัน กับราคาปิดรายวัน) task_c ยังไม่เคยตรวจเลย

วิธี: เทียบแบบ same-source กับ Settrade เอง (ตัดปัญหาความต่างข้ามแหล่งข้อมูล
ยกเว้นฝั่ง yfinance ที่เราต้องการตรวจอยู่แล้ว) -- ดึงราคาปิดรายวัน
(interval=1d) จาก Settrade มาเป็น ground truth แล้วเทียบกับแท่ง 16:00
ของ yfinance ที่ cache ไว้

ต้องตั้ง SETTRADE_APP_ID, SETTRADE_APP_SECRET ก่อนรัน
รัน:
    python check_settrade_atc_match.py
"""

import os

import numpy as np
import pandas as pd
from settrade_v2.user import Investor

from config import TICKERS, INTRADAY_CLOSE_HOUR

investor = Investor(
    app_id=os.environ["SETTRADE_APP_ID"],
    app_secret=os.environ["SETTRADE_APP_SECRET"],
    broker_id=os.environ.get("SETTRADE_BROKER_ID", "023"),
    app_code=os.environ.get("SETTRADE_APP_CODE", "ALGO_EQ"),
)
market = investor.MarketData()


def _hourly_path(ticker):
    safe = ticker.replace("^", "").replace(".", "_")
    return os.path.join("data_cache", f"{safe}_1h_730d.csv")


def load_close_hour_bars(ticker, close_hour=INTRADAY_CLOSE_HOUR):
    """แท่ง Hour == close_hour ของทุกวันใน data_cache/{ticker}_1h_730d.csv"""
    path = _hourly_path(ticker)
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df["Date"] = df["Datetime"].dt.normalize().dt.date
    df["Hour"] = df["Datetime"].dt.hour
    hits = df[df["Hour"] == close_hour].sort_values("Datetime")
    hits = hits.drop_duplicates("Date", keep="last")
    return pd.Series(hits["Close"].to_numpy(), index=hits["Date"].to_numpy())


def load_daily_close_settrade(symbol, start, end):
    """ราคาปิดรายวันจาก Settrade เอง (interval=1d) -- ground truth ของ ATC"""
    data = market.get_candlestick(symbol=symbol, interval="1d",
                                  start=start, end=end, limit=1000)
    n = len(data.get("time", []))
    if n == 0:
        return pd.Series(dtype=float)
    dt = pd.to_datetime(data["time"], unit="s", utc=True) \
        .tz_convert("Asia/Bangkok")
    return pd.Series(data["close"], index=dt.date)


print("=" * 78)
print(f"  เทียบแท่ง {INTRADAY_CLOSE_HOUR}:00 (yfinance 1h, cache) "
      "กับราคาปิดรายวันทางการ (Settrade)")
print("=" * 78)

for ticker in TICKERS:
    symbol = ticker.replace(".BK", "")
    print(f"\n[{ticker}] อ่าน {_hourly_path(ticker)} ...")

    close_hour_bars = load_close_hour_bars(ticker)
    if close_hour_bars.empty:
        print(f"  !! ไม่มีแท่ง {INTRADAY_CLOSE_HOUR}:00 เลยในไฟล์ cache")
        continue
    print(f"  ได้ {len(close_hour_bars)} วันที่มีแท่ง {INTRADAY_CLOSE_HOUR}:00 "
          f"({close_hour_bars.index.min()} -> {close_hour_bars.index.max()})")

    start = (pd.Timestamp(close_hour_bars.index.min())
            - pd.Timedelta(days=5)).strftime("%Y-%m-%dT00:00:00+07:00")
    end = (pd.Timestamp(close_hour_bars.index.max())
          + pd.Timedelta(days=1)).strftime("%Y-%m-%dT23:59:59+07:00")
    print(f"  ดึงราคาปิดรายวันจาก Settrade ({symbol}) ตั้งแต่ {start} ถึง {end} ...")
    daily_close = load_daily_close_settrade(symbol, start, end)
    if daily_close.empty:
        print("  !! Settrade ไม่คืนข้อมูลราคาปิดรายวันเลย เทียบไม่ได้")
        continue

    common = sorted(set(close_hour_bars.index) & set(daily_close.index))
    print(f"  วันที่เทียบได้: {len(common)} วัน")
    if not common:
        print("  !! ไม่มีวันทับซ้อนกัน เทียบไม่ได้")
        continue

    a = close_hour_bars.loc[common].to_numpy(dtype=float)
    b = daily_close.loc[common].to_numpy(dtype=float)
    match = np.isclose(a, b, atol=0.01)
    match_pct = match.mean() * 100
    diff = a - b

    print(f"  ตรงกัน: {match_pct:.1f}% ({match.sum()}/{len(common)})")
    print(f"  diff stats: mean={diff.mean():+.4f}  std={diff.std():.4f}  "
          f"max_abs={np.abs(diff).max():.2f}")

    mismatches = [d for d in common
                 if not np.isclose(close_hour_bars.loc[d], daily_close.loc[d], atol=0.01)]
    print(f"  จำนวนวันไม่ตรง: {len(mismatches)}")
    for d in mismatches[:10]:
        print(f"    {d}: yfinance {INTRADAY_CLOSE_HOUR}:00={close_hour_bars.loc[d]:.2f}  "
              f"settrade daily_close={daily_close.loc[d]:.2f}  "
              f"diff={close_hour_bars.loc[d] - daily_close.loc[d]:+.2f}")
    if len(mismatches) > 10:
        print(f"    ... อีก {len(mismatches) - 10} วัน")

print("\nเสร็จสิ้น — ส่ง output กลับมาดูได้เลย (ไม่มี secret ปนอยู่)")
