"""
tools/fetch_yahoo_intraday.py — ตัวเชื่อม (connector) ข้อมูลรายชั่วโมง: Yahoo -> live snapshot (Phase 1D)
======================================================================================================

    python tools/fetch_yahoo_intraday.py                    # ดึงทุกหุ้น -> บันทึก snapshot ใน raw_data_intraday_live/
    python tools/fetch_yahoo_intraday.py --no-save --out-dir <dir>   # ดึงแล้วเขียน CSV อย่างเดียว (ทดสอบรูปแบบ)

แหล่งเดียวกับไฟล์ล็อก raw_data_intraday/ (yfinance interval=1h, auto_adjust=False) -> source = "yahoo"
ซึ่งเป็นแหล่งเดียวที่ predict_1600.py ยอมรับ · รูปแบบ CSV ตรงกับไฟล์ล็อก:
    Datetime(+07:00, เวลาเริ่มแท่ง),Adj Close,Close,High,Low,Open,Volume

downloaded_at ที่บันทึก = เวลา "เริ่ม" ดาวน์โหลด (ค่าที่ระมัดระวังกว่า: ถ้าเริ่มก่อน 16:00 guard จะปฏิเสธ
แม้ดาวน์โหลดเสร็จหลัง 16:00) · บันทึกผ่าน live_1600.save_snapshot (byte-for-byte + SHA + meta)
"""

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intraday_1600 import BANGKOK                                      # noqa: E402
from live_1600 import SnapshotError, load_snapshot, save_snapshot      # noqa: E402

TICKERS = ["KBANK.BK", "ADVANC.BK"]
PERIOD = "60d"               # พอให้ lag/relvol (20 วัน) ต่อกับไฟล์ล็อกได้ครบ
COLUMNS = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]


class FetchError(RuntimeError):
    pass


def download_hourly(ticker, period=PERIOD):
    import yfinance as yf
    df = yf.download(ticker, period=period, interval="1h", auto_adjust=False,
                     prepost=False, progress=False, threads=False)
    if df is None or len(df) == 0:
        raise FetchError(f"{ticker}: Yahoo ไม่คืนข้อมูลรายชั่วโมง")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def to_locked_format(df):
    """ให้ตรงรูปแบบไฟล์ล็อก: index ชื่อ Datetime เป็น +07:00 · คอลัมน์เรียงตาม COLUMNS"""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise FetchError(f"คอลัมน์ขาด {missing}")
    out = df[COLUMNS].copy()
    idx = pd.DatetimeIndex(out.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    out.index = idx.tz_convert(BANGKOK)
    out.index.name = "Datetime"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out["Volume"] = out["Volume"].fillna(0).astype("int64")
    return out


def main(argv=None, downloader=download_hourly):
    ap = argparse.ArgumentParser(description="ดึงข้อมูลรายชั่วโมงจาก Yahoo แล้วบันทึกเป็น live snapshot")
    ap.add_argument("--no-save", action="store_true", help="ไม่บันทึก snapshot (เขียน CSV ลง --out-dir อย่างเดียว)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--period", default=PERIOD)
    args = ap.parse_args(argv)
    if args.no_save and args.out_dir is None:
        ap.error("--no-save ต้องใช้กับ --out-dir")

    started = datetime.now(BANGKOK).replace(microsecond=0)
    try:
        frames = {t: to_locked_format(downloader(t, args.period)) for t in TICKERS}   # ดึงครบก่อนค่อยบันทึก
    except FetchError as e:
        print(f"!! หยุด -- ไม่ได้บันทึก snapshot ใด ๆ\n   {e}")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = args.out_dir or Path(tmp)
        out_dir.mkdir(parents=True, exist_ok=True)
        for t, df in frames.items():
            path = out_dir / f"{t.replace('.', '_')}_1h_yahoo_{started:%Y%m%d_%H%M%S}.csv"
            df.to_csv(path, encoding="utf-8")
            last = df.index[-1]
            print(f"=== {t}: {len(df)} แท่ง · แท่งล่าสุด {last.isoformat()}")
            if args.no_save:
                print(f"  เขียน {path}")
                continue
            try:
                dst = save_snapshot(path, t, "yahoo", started.isoformat())
            except SnapshotError as e:
                print(f"!! {t}: {e}")
                return 1
            _, meta = load_snapshot(dst)
            print(f"  snapshot {dst.relative_to(dst.parents[2])} · downloaded_at {meta['downloaded_at'].isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
