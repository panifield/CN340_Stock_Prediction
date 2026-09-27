"""
check_settrade_intervals.py — Task A2 (ตรวจสอบเท่านั้น)
=========================================================
ทดสอบว่า Settrade Open API รองรับ interval ระดับ intraday ไหม
(ไม่มีเอกสารสาธารณะที่ยืนยันชัดเจน ต้องยิงจริงเพื่อดูว่า server ตอบอะไร)

ต้องตั้ง environment variable ก่อนรัน:
    SETTRADE_APP_ID, SETTRADE_APP_SECRET
    (SETTRADE_BROKER_ID, SETTRADE_APP_CODE มีค่า default ให้แล้ว)

รัน:
    python check_settrade_intervals.py
"""

import os
from settrade_v2.user import Investor

investor = Investor(
    app_id=os.environ["SETTRADE_APP_ID"],
    app_secret=os.environ["SETTRADE_APP_SECRET"],
    broker_id=os.environ.get("SETTRADE_BROKER_ID", "023"),
    app_code=os.environ.get("SETTRADE_APP_CODE", "ALGO_EQ"),
)
market = investor.MarketData()

# ลอง interval หลายแบบที่เป็นไปได้ — ไม่รู้ syntax ที่ถูกต้องแน่ชัด
# เลยลองทุกแบบที่ API เทคนิคชาร์ตทั่วไปมักใช้
CANDIDATE_INTERVALS = [
    "1d", "1D", "day",
    "1m", "1min", "1",
    "5m", "5min", "5",
    "15m", "15min", "15",
    "60m", "60min", "60", "1h", "1H",
]

print("=" * 70)
print("  ทดสอบ interval ที่ Settrade Open API รองรับ (symbol=KBANK)")
print("=" * 70)

results = []
for interval in CANDIDATE_INTERVALS:
    try:
        data = market.get_candlestick(
            symbol="KBANK",
            interval=interval,
            limit=5,
        )
        n = len(data.get("time", []))
        ok = n > 0
        results.append((interval, "OK" if ok else "เรียกผ่านแต่ไม่มีข้อมูล (n=0)", n))
        print(f"  interval={interval!r:10s} -> {'OK' if ok else 'no data'} "
              f"(n={n})")
        if ok:
            # โชว์ timestamp ตัวอย่าง 2 อันแรก เพื่อดูว่าเป็นรายวันหรือ intraday
            import pandas as pd
            times = pd.to_datetime(data["time"], unit="s", utc=True) \
                .tz_convert("Asia/Bangkok")
            print(f"      ตัวอย่างเวลา: {list(times[:2])}")
    except Exception as e:
        results.append((interval, f"ERROR: {type(e).__name__}: {e}", 0))
        print(f"  interval={interval!r:10s} -> ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("  สรุป — ส่ง output ทั้งหมดนี้กลับมาให้ดูได้เลย (ไม่มี secret ปนอยู่)")
print("=" * 70)
