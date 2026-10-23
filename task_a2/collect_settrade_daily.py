"""
collect_settrade_daily.py — Task A2 (เก็บสะสมข้อมูล)
=====================================================
ดึง intraday (1m + 5m) ของ KBANK/ADVANC จาก Settrade แล้วเก็บสะสมไว้
เพราะ Settrade เก็บ retention แค่ ~1-2 สัปดาห์ ถ้าไม่เก็บเองทุกวัน
ข้อมูลจะหายไปเรื่อยๆ

รันหลังตลาดปิด (scheduled task SettradeDailyCollector เรียก run_collector.bat 17:30 ทุกวัน)
    cd task_a2
    python collect_settrade_daily.py              # วันนี้ + เติมวันที่ขาดย้อนหลัง 14 วัน
    python collect_settrade_daily.py --days 0     # วันนี้อย่างเดียว

เติมย้อนหลัง (2026-10-24): เครื่อง sleep/ใช้แบตตอน 17:30 ทำให้พลาดหลายวัน (21-22 ต.ค.) แต่ Settrade
ยังเก็บแท่งไว้ ~1-2 สัปดาห์ ทุกครั้งที่รันจึงลองดึงวันทำการ (จ.-ศ.) ย้อนหลังที่ยังไม่มีไฟล์ด้วย
ไฟล์ที่มีแล้วไม่ดึงซ้ำ ยกเว้นวันนี้ (เขียนทับได้ เผื่อรันก่อนตลาดปิด) · วันหยุดได้ "ไม่มีข้อมูล" ข้ามไป

credential: environment variable ของผู้ใช้ หรือ .env

อ่าน credential จาก .env (ไฟล์ .env ต้องอยู่ที่ root โปรเจกต์หรือ
task_a2/ เอง) รูปแบบ:
    SETTRADE_APP_ID=xxxx
    SETTRADE_APP_SECRET=xxxx
    SETTRADE_BROKER_ID=023
    SETTRADE_APP_CODE=ALGO_EQ

บันทึกลง task_a2/data/settrade_log/{SYMBOL}_{interval}_{YYYY-MM-DD}.csv
รันซ้ำวันเดิมจะเขียนทับไฟล์ของวันนั้นเท่านั้น (idempotent ต่อวัน)
"""

import argparse
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


def file_path(symbol, interval, day):
    return os.path.join(LOG_DIR, f"{symbol}_{interval}_{day.strftime('%Y-%m-%d')}.csv")


def days_to_fetch(today, lookback_days):
    """วันนี้ (ถ้าเป็นวันธรรมดา) + วันธรรมดาย้อนหลังที่ยังไม่มีไฟล์ครบทุก symbol/interval"""
    out = []
    for k in range(lookback_days, -1, -1):
        day = today - pd.Timedelta(days=k)
        if day.dayofweek >= 5:
            continue
        have_all = all(os.path.exists(file_path(s, i, day)) for s in SYMBOLS for i in INTERVALS)
        if k == 0 or not have_all:
            out.append(day)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="เติมวันที่ขาดย้อนหลังกี่วัน (0 = วันนี้อย่างเดียว)")
    ap.add_argument("--log", help="เขียน output ต่อท้ายไฟล์นี้ (scheduled task รันผ่าน pythonw.exe ที่ไม่มี console)")
    args = ap.parse_args()
    if args.log:
        # ไม่มี console window -> ไม่โดน Ctrl+C/ปิดหน้าต่างฆ่ากลางทาง (0xC000013A ใน scheduler_run.log เดิม)
        sys.stdout = sys.stderr = open(args.log, "a", encoding="utf-8", buffering=1)
    today = pd.Timestamp.now(tz="Asia/Bangkok").normalize()
    date_str = today.strftime("%Y-%m-%d")

    print("=" * 70)
    print(f"  เก็บ intraday จาก Settrade — วันที่ {date_str} (เติมย้อนหลัง {args.days} วัน)")
    print("=" * 70)

    investor = get_investor()
    market = investor.MarketData()

    n_saved = 0
    for day in days_to_fetch(today, args.days):
        for symbol in SYMBOLS:
            for interval in INTERVALS:
                path = file_path(symbol, interval, day)
                if day != today and os.path.exists(path):
                    continue
                print(f"  ดึง {symbol}/{interval} {day.date()} ...", end=" ", flush=True)
                df = fetch_today(market, symbol, interval, day)
                if df is None:
                    continue
                df.to_csv(path, index=False, encoding="utf-8-sig")
                print(f"OK ({len(df)} แท่ง) -> {path}")
                n_saved += 1

    print(f"\nเสร็จสิ้น — บันทึกไป {n_saved} ไฟล์")
    if n_saved == 0:
        print("!! ไม่ได้ข้อมูลเลยสักไฟล์ — เช็คว่าวันนี้ตลาดเปิดไหม "
              "หรือรันก่อนตลาดปิด (ควรรันหลัง ~17:00 น.)")


if __name__ == "__main__":
    main()
