"""
collect_settrade_daily.py — Task A2 (เก็บสะสมข้อมูล)
=====================================================
ดึง intraday (1m + 5m) ของ KBANK/ADVANC จาก Settrade แล้วเก็บสะสมไว้
เพราะ Settrade เก็บ retention แค่ ~1-2 สัปดาห์ ถ้าไม่เก็บเองทุกวัน
ข้อมูลจะหายไปเรื่อยๆ

*** รันด้วยมือหลังตลาดปิดทุกวัน (ยังไม่ตั้ง scheduler) ***
    cd task_a2
    python collect_settrade_daily.py

อ่าน credential จาก .env (ไฟล์ .env ต้องอยู่ที่ root โปรเจกต์หรือ
task_a2/ เอง) รูปแบบ:
    SETTRADE_APP_ID=xxxx
    SETTRADE_APP_SECRET=xxxx
    SETTRADE_BROKER_ID=023
    SETTRADE_APP_CODE=ALGO_EQ

บันทึกลง task_a2/data/settrade_log/{SYMBOL}_{interval}_{YYYY-MM-DD}.csv
รันซ้ำวันเดิมจะเขียนทับไฟล์ของวันนั้นเท่านั้น (idempotent ต่อวัน)
"""

import os
import sys

import pandas as pd
from dotenv import load_dotenv
from settrade_v2.user import Investor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
LOG_DIR = os.path.join(HERE, "data", "settrade_log")
os.makedirs(LOG_DIR, exist_ok=True)

# หา .env ทั้งที่ root โปรเจกต์และใน task_a2/ เอง (root มาก่อน)
load_dotenv(os.path.join(ROOT, ".env"))
load_dotenv(os.path.join(HERE, ".env"), override=False)

SYMBOLS = ["KBANK", "ADVANC"]
INTERVALS = ["1m", "5m"]


def get_investor():
    missing = [k for k in ("SETTRADE_APP_ID", "SETTRADE_APP_SECRET")
              if not os.environ.get(k)]
    if missing:
        print(f"!! ไม่พบ environment variable: {', '.join(missing)}")
        print("   ตั้งใน .env หรือ export ก่อนรัน (ดู docstring ของไฟล์นี้)")
        sys.exit(1)

    return Investor(
        app_id=os.environ["SETTRADE_APP_ID"],
        app_secret=os.environ["SETTRADE_APP_SECRET"],
        broker_id=os.environ.get("SETTRADE_BROKER_ID", "023"),
        app_code=os.environ.get("SETTRADE_APP_CODE", "ALGO_EQ"),
    )


def fetch_today(market, symbol, interval, today):
    start = today.strftime("%Y-%m-%dT00:00:00+07:00")
    end = today.strftime("%Y-%m-%dT23:59:59+07:00")
    try:
        data = market.get_candlestick(symbol=symbol, interval=interval,
                                      start=start, end=end, limit=1000)
    except Exception as e:
        print(f"    !! {symbol}/{interval}: ERROR {type(e).__name__}: {e}")
        return None

    n = len(data.get("time", []))
    if n == 0:
        print(f"    {symbol}/{interval}: ไม่มีข้อมูล (วันหยุด/ตลาดยังไม่ปิด?)")
        return None

    df = pd.DataFrame({
        "time": data["time"], "open": data["open"], "high": data["high"],
        "low": data["low"], "close": data["close"],
        "volume": data.get("volume", [None] * n),
    })
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True) \
        .dt.tz_convert("Asia/Bangkok")
    return df


def main():
    today = pd.Timestamp.now(tz="Asia/Bangkok").normalize()
    date_str = today.strftime("%Y-%m-%d")

    print("=" * 70)
    print(f"  เก็บ intraday จาก Settrade — วันที่ {date_str}")
    print("=" * 70)

    investor = get_investor()
    market = investor.MarketData()

    n_saved = 0
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            print(f"  ดึง {symbol}/{interval} ...", end=" ", flush=True)
            df = fetch_today(market, symbol, interval, today)
            if df is None:
                continue
            path = os.path.join(LOG_DIR, f"{symbol}_{interval}_{date_str}.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"OK ({len(df)} แท่ง) -> {path}")
            n_saved += 1

    print(f"\nเสร็จสิ้น — บันทึกไป {n_saved} ไฟล์")
    if n_saved == 0:
        print("!! ไม่ได้ข้อมูลเลยสักไฟล์ — เช็คว่าวันนี้ตลาดเปิดไหม "
              "หรือรันก่อนตลาดปิด (ควรรันหลัง ~17:00 น.)")


if __name__ == "__main__":
    main()
