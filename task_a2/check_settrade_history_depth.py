"""
check_settrade_history_depth.py — Task A2 (ตรวจสอบเท่านั้น)
=============================================================
เช็คว่า Settrade เก็บข้อมูล intraday (1m, 5m) ย้อนหลังได้ไกลแค่ไหนจริงๆ
โดยขอช่วงวันที่เก่าๆ (1 เดือน, 3 เดือน, 1 ปี, 3 ปีก่อน) ทีละช่วงสั้นๆ
(ไม่ชนกับ limit 1000) แล้วดูว่า server มีข้อมูลให้ไหม

รัน:
    python check_settrade_history_depth.py
"""

import os
import pandas as pd
from settrade_v2.user import Investor

investor = Investor(
    app_id=os.environ["SETTRADE_APP_ID"],
    app_secret=os.environ["SETTRADE_APP_SECRET"],
    broker_id=os.environ.get("SETTRADE_BROKER_ID", "023"),
    app_code=os.environ.get("SETTRADE_APP_CODE", "ALGO_EQ"),
)
market = investor.MarketData()

NOW = pd.Timestamp.now(tz="Asia/Bangkok")

# ขอทีละ 3 วันปฏิทิน (สั้นพอไม่ชน limit 1000 แท่งของ 1m แน่นอน)
WINDOWS_AGO = {
    "7 วันก่อน": pd.Timedelta(days=7),
    "1 เดือนก่อน": pd.Timedelta(days=30),
    "3 เดือนก่อน": pd.Timedelta(days=90),
    "6 เดือนก่อน": pd.Timedelta(days=180),
    "1 ปีก่อน": pd.Timedelta(days=365),
    "2 ปีก่อน": pd.Timedelta(days=730),
    "3 ปีก่อน": pd.Timedelta(days=1095),
}


def try_window(symbol, interval, center):
    start = (center - pd.Timedelta(days=3)).strftime("%Y-%m-%dT00:00:00+07:00")
    end = (center + pd.Timedelta(days=3)).strftime("%Y-%m-%dT23:59:59+07:00")
    try:
        data = market.get_candlestick(symbol=symbol, interval=interval,
                                      start=start, end=end, limit=1000)
    except Exception as e:
        return None, f"ERROR: {type(e).__name__}: {e}"
    n = len(data.get("time", []))
    if n == 0:
        return 0, "ไม่มีข้อมูล"
    dt = pd.to_datetime(data["time"], unit="s", utc=True) \
        .tz_convert("Asia/Bangkok")
    return n, f"{n} แท่ง, {dt.min()} -> {dt.max()}"


print("=" * 78)
print("  เช็คความลึกของข้อมูลย้อนหลัง — interval 1m และ 5m, symbol=KBANK")
print("=" * 78)

for interval in ["1m", "5m"]:
    print(f"\n--- interval={interval} ---")
    for label, delta in WINDOWS_AGO.items():
        center = NOW - delta
        n, msg = try_window("KBANK", interval, center)
        print(f"  {label:15s} (~{center.date()}): {msg}")

print("\nเสร็จสิ้น — ส่ง output กลับมาดูได้เลย (ไม่มี secret ปนอยู่)")
