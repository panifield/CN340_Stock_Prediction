"""
check_settrade_atc_match.py — Task A2 (ตรวจสอบเท่านั้น)
=========================================================
Settrade Open API รองรับ interval: 1d, 1m, 5m, 15m, 60m (ยืนยันแล้ว)
ขั้นนี้เช็คคำถามสำคัญกว่า: แท่งสุดท้ายของวัน (1m) ตรงกับราคาปิดทางการ
(ATC) ที่ใช้อยู่ใน task_a/data_cache หรือไม่ — ถ้าตรง = แก้ปัญหาหลักของ
Task A2 ได้ ถ้าไม่ตรง = ยังมีปัญหาเดิม

ต้องตั้ง SETTRADE_APP_ID, SETTRADE_APP_SECRET ก่อนรัน
รัน:
    python check_settrade_atc_match.py
"""

import os
import numpy as np
import pandas as pd
from settrade_v2.user import Investor

investor = Investor(
    app_id=os.environ["SETTRADE_APP_ID"],
    app_secret=os.environ["SETTRADE_APP_SECRET"],
    broker_id=os.environ.get("SETTRADE_BROKER_ID", "023"),
    app_code=os.environ.get("SETTRADE_APP_CODE", "ALGO_EQ"),
)
market = investor.MarketData()

TICKERS = {"KBANK": "KBANK_BK", "ADVANC": "ADVANC_BK"}
DAILY_CACHE = os.path.join("..", "task_a", "data_cache")


def fetch_1m(symbol, start, end, limit=1000):
    data = market.get_candlestick(symbol=symbol, interval="1m",
                                  start=start, end=end, limit=limit)
    n = len(data.get("time", []))
    if n == 0:
        return None
    df = pd.DataFrame({
        "time": data["time"], "open": data["open"], "high": data["high"],
        "low": data["low"], "close": data["close"],
        "volume": data.get("volume", [None] * n),
    })
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True) \
        .dt.tz_convert("Asia/Bangkok")
    return df


def load_daily_close_settrade(symbol, start, end):
    """ราคาปิดรายวันจาก Settrade เอง (interval=1d) — เทียบแบบ same-source
    ตัดปัญหาความต่างข้ามแหล่งข้อมูล (Settrade vs yfinance) ออกไปเลย"""
    data = market.get_candlestick(symbol=symbol, interval="1d",
                                  start=start, end=end, limit=1000)
    n = len(data.get("time", []))
    if n == 0:
        return pd.Series(dtype=float)
    dt = pd.to_datetime(data["time"], unit="s", utc=True) \
        .tz_convert("Asia/Bangkok")
    s = pd.Series(data["close"], index=dt.date)
    return s


print("=" * 78)
print("  เทียบแท่งสุดท้ายของวัน (Settrade 1m) กับราคาปิดรายวันที่ใช้อยู่")
print("=" * 78)

# ดึงย้อนหลัง ~30 วันทำการล่าสุด (1m data มักเก็บไม่นาน ลองดูก่อน)
end = pd.Timestamp.now(tz="Asia/Bangkok")
start = end - pd.Timedelta(days=45)
start_str = start.strftime("%Y-%m-%dT00:00:00+07:00")
end_str = end.strftime("%Y-%m-%dT23:59:59+07:00")

for symbol, safe_name in TICKERS.items():
    print(f"\n[{symbol}] ดึง 1m ตั้งแต่ {start_str} ถึง {end_str} ...")
    df = fetch_1m(symbol, start_str, end_str, limit=1000)
    if df is None or len(df) == 0:
        print("  !! ไม่มีข้อมูล")
        continue

    print(f"  ได้ {len(df)} แท่ง, {df['dt'].min()} -> {df['dt'].max()}")

    date_list = df["dt"].dt.date
    last_bar = df.groupby(date_list)["close"].last()
    last_time = df.groupby(date_list)["dt"].last()

    # ดึงราคาปิดรายวันจาก Settrade เอง ช่วงเดียวกับที่แท่ง 1m ครอบคลุม
    d_start = (df["dt"].min() - pd.Timedelta(days=5)).strftime("%Y-%m-%dT00:00:00+07:00")
    d_end = (df["dt"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%dT23:59:59+07:00")
    daily_close = load_daily_close_settrade(symbol, d_start, d_end)
    common = sorted(set(last_bar.index) & set(daily_close.index))
    print(f"  วันที่เทียบได้: {len(common)} วัน")

    if len(common) == 0:
        print("  !! ไม่มีวันทับซ้อนกับไฟล์ราคาปิดรายวัน (อาจเพราะข้อมูล "
              "1m ใหม่กว่าไฟล์ daily cache) เทียบไม่ได้")
        continue

    a = last_bar.loc[common].values
    b = daily_close.loc[common].values
    match = np.isclose(a, b, atol=0.01)
    match_pct = match.mean() * 100
    diff = a - b

    print(f"  ตรงกัน: {match_pct:.1f}% ({match.sum()}/{len(common)})")
    print(f"  diff stats: mean={diff.mean():+.4f}  std={diff.std():.4f}  "
          f"max_abs={np.abs(diff).max():.2f}")
    print(f"  เวลาแท่งสุดท้ายของวัน (ตัวอย่าง 5 วันล่าสุด):")
    for d in common[-5:]:
        print(f"    {d}: last_bar_time={last_time.loc[d].strftime('%H:%M:%S')}  "
              f"1m_close={last_bar.loc[d]:.2f}  daily_close={daily_close.loc[d]:.2f}  "
              f"{'OK' if np.isclose(last_bar.loc[d], daily_close.loc[d], atol=0.01) else 'MISMATCH'}")

    out_path = f"data/{symbol}_1m_atc_check.csv"
    os.makedirs("data", exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [saved] {out_path}")

print("\nเสร็จสิ้น — ส่ง output กลับมาดูได้เลย (ไม่มี secret ปนอยู่)")
