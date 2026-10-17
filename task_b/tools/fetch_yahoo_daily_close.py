"""
tools/fetch_yahoo_daily_close.py — ตัวเชื่อมราคาปิดรายวัน: Yahoo 1d -> live snapshot (Phase 1D รอบ 2)
=====================================================================================================

    python tools/fetch_yahoo_daily_close.py                    # ดึงทุกหุ้น -> raw_data_daily_live/<YYYYMMDD>/
    python tools/fetch_yahoo_daily_close.py --no-save --out-dir <dir>   # ดึงแล้วเขียน CSV อย่างเดียว (ทดสอบรูปแบบ)

yfinance interval=1d, auto_adjust=False (ราคาจริง ไม่ปรับปันผล) · ดึงตั้งแต่ DAILY_START เพื่อให้ทับไฟล์ล็อก
(raw_data_daily/, ถึง 2026-09-25) ได้ — แถวที่ทับกันใช้ค่าจากไฟล์ล็อกและรายงานแถวที่ไม่ตรง
บันทึก "ทั้งหมดที่ Yahoo คืน" byte-for-byte (รวมแถวที่ Volume=0 / Close ว่าง) พร้อม SHA-256 + meta
การกรองแถวผีทำตอนใช้ (daily_close.usable) ไม่ใช่ตอนบันทึก — เพื่อให้เห็นหลักฐานของสิ่งที่ Yahoo ส่งมาจริง

ไม่ดึงประวัติเต็ม 10 ปี: Yahoo 1d ที่ดึงสดมีแถวเสียปี 2024 (ดู raw_data_daily/README.md) ประวัติมาจากไฟล์ล็อกเท่านั้น
downloaded_at ที่บันทึก = เวลา "เริ่ม" ดาวน์โหลด (ค่าระมัดระวัง เหมือน fetch_yahoo_intraday.py)
"""

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daily_close import DailyCloseError, load_snapshot, save_snapshot      # noqa: E402
from intraday_1600 import BANGKOK                                          # noqa: E402

TICKERS = ["KBANK.BK", "ADVANC.BK"]
DAILY_START = "2026-09-01"           # ทับไฟล์ล็อก (สิ้นสุด 2026-09-25) ~3 สัปดาห์ เพื่อตรวจความสอดคล้อง
COLUMNS = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]


class FetchError(RuntimeError):
    pass


def download_daily(ticker, start=DAILY_START):
    import yfinance as yf
    df = yf.download(ticker, start=start, interval="1d", auto_adjust=False,
                     progress=False, threads=False)
    if df is None or len(df) == 0:
        raise FetchError(f"{ticker}: Yahoo ไม่คืนข้อมูลรายวัน")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def to_snapshot_format(df):
    """index ชื่อ Date (naive, เที่ยงคืน) · คอลัมน์เรียงตาม COLUMNS · เก็บทุกแถวที่ Yahoo คืน"""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise FetchError(f"คอลัมน์ขาด {missing}")
    out = df[COLUMNS].copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "Date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def main(argv=None, downloader=download_daily):
    ap = argparse.ArgumentParser(description="ดึงราคาปิดรายวันจาก Yahoo แล้วบันทึกเป็น live snapshot")
    ap.add_argument("--no-save", action="store_true", help="ไม่บันทึก snapshot (เขียน CSV ลง --out-dir อย่างเดียว)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--start", default=DAILY_START)
    args = ap.parse_args(argv)
    if args.no_save and args.out_dir is None:
        ap.error("--no-save ต้องใช้กับ --out-dir")

    started = datetime.now(BANGKOK).replace(microsecond=0)
    try:
        frames = {t: to_snapshot_format(downloader(t, args.start)) for t in TICKERS}   # ดึงครบก่อนค่อยบันทึก
    except FetchError as e:
        print(f"!! หยุด -- ไม่ได้บันทึก snapshot ใด ๆ\n   {e}")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = args.out_dir or Path(tmp)
        out_dir.mkdir(parents=True, exist_ok=True)
        for t, df in frames.items():
            path = out_dir / f"{t.replace('.', '_')}_1d_yahoo_{started:%Y%m%d_%H%M%S}.csv"
            df.to_csv(path, encoding="utf-8", date_format="%Y-%m-%d")
            ghost = int((df["Volume"].fillna(0) <= 0).sum())
            print(f"=== {t}: {len(df)} แถว · แถวล่าสุด {df.index[-1].date()} · แถว Volume<=0 หรือว่าง {ghost}")
            if args.no_save:
                print(f"  เขียน {path}")
                continue
            try:
                dst = save_snapshot(path, t, "yahoo", started.isoformat())
            except DailyCloseError as e:
                print(f"!! {t}: {e}")
                return 1
            _, meta = load_snapshot(dst)
            print(f"  snapshot {dst.relative_to(dst.parents[2])} · downloaded_at {meta['downloaded_at'].isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
