"""
tools/make_locked_daily.py — สร้างไฟล์ราคาปิดรายวันที่ "ล็อก" สำหรับ Phase 1D รอบ 2 (ครั้งเดียว)
==========================================================================================

    python tools/make_locked_daily.py            # เขียน raw_data_daily/<TICKER>_1d_locked.csv (ไม่ทับถ้ามีแล้ว)
    python tools/make_locked_daily.py --check    # ตรวจว่า raw_data/ ปัจจุบันยังให้ไฟล์ล็อกเดียวกันเป๊ะ (ไม่เขียน)

ที่มา: raw_data/<TICKER>_10Y_Cleaned.csv (Investing.com + Yahoo ที่ตรวจ OHLC ทับของเดิมเป๊ะ) ตัดที่วันที่ <= LOCKED_END
เหตุผลที่ไม่ใช้ Yahoo 1d ดึงสดมาเป็นประวัติ (PHASE1D_PLAN_R2.md ข้อ 3):
  Yahoo 1d ที่ดึงเมื่อ 2026-10-17 มีแถวเสียในปี 2024 (Volume = 0 ราคาปิดค้าง KBANK 65 วัน / ADVANC 47 วัน)
  ขณะที่ไฟล์นี้ตรงกับ cache เก่าของ task_a (ดึงก่อนหน้านั้น) 100% ในช่วงเดียวกัน
คอลัมน์: Date (YYYY-MM-DD), Close, Volume · SHA-256 ของไฟล์ผลลัพธ์ถูกล็อกใน daily_close.EXPECTED_SHA256
ไฟล์ raw_data/ ไม่ถูกแก้ (อ่านอย่างเดียว) · ไฟล์นี้อ่าน data_loader ได้เพราะเป็นเครื่องมือสร้างไฟล์ ไม่ใช่โมดูล Phase 1D
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from daily_close import DAILY_DIR, LOCKED_END, locked_path  # noqa: E402
from data_loader import load_stock                          # noqa: E402

TICKERS = ["KBANK.BK", "ADVANC.BK"]


def build(ticker):
    df = load_stock(ticker, verbose=False)
    df = df.loc[:LOCKED_END]
    out = pd.DataFrame({"Close": df["Close"], "Volume": df["Volume"].astype("int64")})
    out.index = pd.DatetimeIndex(out.index).normalize()
    out.index.name = "Date"
    return out


def to_bytes(df):
    return df.to_csv(lineterminator="\n", date_format="%Y-%m-%d").encode("utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    DAILY_DIR.mkdir(exist_ok=True)
    rc = 0
    for t in TICKERS:
        b = to_bytes(build(t))
        sha = hashlib.sha256(b).hexdigest()
        path = locked_path(t)
        if args.check:
            same = path.exists() and path.read_bytes() == b
            print(f"{t}: {'ตรง' if same else 'ไม่ตรง'} · sha256 {sha}")
            rc |= 0 if same else 1
            continue
        if path.exists():
            print(f"{t}: มีไฟล์แล้ว ({path.name}) -- ไม่ทับ · sha256 ใหม่ {sha}")
            rc |= 0 if path.read_bytes() == b else 1
            continue
        path.write_bytes(b)
        print(f"{t}: เขียน {path} · {len(b.splitlines()) - 1} แถว · sha256 {sha}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
