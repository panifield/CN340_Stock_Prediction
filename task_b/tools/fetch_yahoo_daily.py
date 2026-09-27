"""
tools/fetch_yahoo_daily.py — ตัวเชื่อม (connector) ข้อมูลรายวัน: Yahoo -> ไฟล์รูปแบบ Investing.com
=================================================================================================

    python tools/fetch_yahoo_daily.py                     # ดึง + ตรวจ + เขียนไฟล์ลง .staging/
    python tools/fetch_yahoo_daily.py --check-only        # ดึง + ตรวจ ไม่เขียนไฟล์
    python tools/append_investing.py .staging/yahoo_<YYYYMMDD>/*.csv --source yahoo

ทำไมใช้ Yahoo ได้: Investing.com ไม่มี API จึง automate ไม่ได้ · ตรวจแล้ว (2026-09-28) ว่า
Open/High/Low/Close รายวันของ Yahoo ตรงกับ raw_data/ **เป๊ะ 100%** ทุกวันที่ทับกัน
(2,453 วัน × 2 หุ้น) · Volume ตรงเป๊ะ 2,433 วันแรก แต่ 20 แถวที่ append จาก Investing (2026-08-31 ->
2026-09-25) ต่างได้ เพราะ Investing ปัดเป็น 3 หลัก (`8.65M`) และ Yahoo แก้ Volume ย้อนหลังได้
-> ตัวเชื่อมนี้ตรวจราคา "เป๊ะ" และรายงาน Volume เป็นคำเตือนเท่านั้น

ตัวเชื่อมนี้ไม่เขียน raw_data/ เอง -- ส่งต่อให้ append_investing.py ตามกติกาใน INTEGRATION.md ข้อ 2

ขั้นตอน (หยุดทันทีถ้ามี error -- ไม่เขียนไฟล์):
  1. อ่านวันสุดท้ายของ raw_data/ แล้วดึง Yahoo 1d ย้อนหลัง OVERLAP_DAYS วันปฏิทินจากวันนั้น
  2. ตัดแถวของวันนี้ทิ้งถ้ายังไม่ถึง FINAL_AFTER (แท่งวันนี้ยังไม่จบ / ยังไม่นิ่ง) และตัด "แถวผีวันหยุด"
     ของ Yahoo (Volume = 0 และ O=H=L=C -- เช่น 2026-08-12 วันแม่) พร้อมรายงานวันที่ตัด
  3. ตรวจช่วงทับ: วันที่ต้องตรงกันทั้งสองฝั่ง และ OHLC ต้องเท่ากันเป๊ะ (ไม่งั้น error)
  4. แถวใหม่: ห้ามมี NaN / Volume <= 0 ที่ไม่ใช่แถวผี / วันเสาร์-อาทิตย์
  5. เขียนไฟล์ต่อหุ้นในรูปแบบ Investing (header เดียวกัน · Vol. หน่วย K แบบเต็มความละเอียด ·
     Change % คำนวณจากราคาปิดวันก่อน) -> exit 0 · ไม่มีวันใหม่ -> exit 3
"""

import argparse
import sys
from datetime import datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BASE_DIR, RAW_DATA_DIR, RAW_DATA_SUFFIX, TICKERS   # noqa: E402
from append_investing import INPUT_HEADER, read_canonical              # noqa: E402

BANGKOK = timezone(timedelta(hours=7))
OVERLAP_DAYS = 45                 # วันปฏิทินที่ดึงทับของเดิมเพื่อตรวจเทียบ (~30 วันทำการ)
FINAL_AFTER = time(18, 0)         # แท่งรายวันของวันนี้ถือว่าจบหลังเวลานี้ (+07:00)
STAGING_DIR = BASE_DIR / ".staging"
EXIT_NO_NEW = 3


class FetchError(RuntimeError):
    """เงื่อนไขที่ต้องหยุด -- ไม่เขียนไฟล์ใด ๆ"""


def download_daily(ticker, start, end_exclusive):
    """Yahoo 1d (ไม่ adjust) -> DataFrame index = วันที่ (naive) คอลัมน์ Open/High/Low/Close/Volume"""
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end_exclusive, interval="1d",
                     auto_adjust=False, progress=False, threads=False)
    if df is None or len(df) == 0:
        raise FetchError(f"{ticker}: Yahoo ไม่คืนข้อมูล ({start} -> {end_exclusive})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Bangkok").tz_localize(None)
    df.index = idx.normalize()
    return df


def drop_unfinished(df, now):
    """ตัดแถววันนี้ (และอนาคต) ถ้ายังไม่ถึง FINAL_AFTER"""
    today = pd.Timestamp(now.date())
    if now.time() >= FINAL_AFTER:
        return df[df.index <= today]
    return df[df.index < today]


def drop_phantom(df):
    """แถวผีวันหยุดของ Yahoo: Volume = 0 และ O=H=L=C -> คืน (df ที่ตัดแล้ว, วันที่ตัด)"""
    flat = (df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"])
    ph = (df["Volume"] == 0) & flat
    return df[~ph], [str(d.date()) for d in df.index[ph]]


def _num(s):
    return float(s)


def verify_overlap(ticker, yahoo, old):
    """ช่วงทับ: วันที่ต้องตรง · OHLC เป๊ะ · คืนข้อความเตือนเรื่อง Volume"""
    lo = yahoo.index.min()
    old_w = old[old.index >= lo]
    y_w = yahoo[yahoo.index <= old.index[-1]]
    only_old = old_w.index.difference(y_w.index)
    only_y = y_w.index.difference(old_w.index)
    if len(only_old) or len(only_y):
        raise FetchError(f"{ticker}: วันที่ในช่วงทับไม่ตรงกัน · มีแต่ raw_data {[str(d.date()) for d in only_old]}"
                         f" · มีแต่ Yahoo {[str(d.date()) for d in only_y]}")
    if len(y_w) == 0:
        raise FetchError(f"{ticker}: ไม่มีช่วงทับให้ตรวจเทียบ (raw_data เก่าเกิน {OVERLAP_DAYS} วัน?) "
                         "-- ดึงช่วงยาวขึ้นด้วย --overlap-days")
    for col_old, col_y in [("Price", "Close"), ("Open", "Open"), ("High", "High"), ("Low", "Low")]:
        a = old_w[col_old].map(_num)
        b = y_w[col_y].astype(float)
        bad = (a - b).abs() > 1e-9
        if bad.any():
            d = bad[bad].index[0]
            raise FetchError(f"{ticker}: {col_old} ไม่ตรงกับ raw_data ใน {int(bad.sum())} วัน "
                             f"(เช่น {d.date()}: raw_data {a[d]} / Yahoo {b[d]}) -- หยุด ห้ามใช้ Yahoo รอบนี้")
    vk_old = old_w["Vol. ('000)"].map(_num)
    vk_y = y_w["Volume"].astype(float) / 1000
    rel = ((vk_y - vk_old).abs() / vk_old.where(vk_old > 0)).fillna(0)
    warns = []
    if (rel > 0.005).any():
        warns.append(f"Volume ต่างจาก raw_data เกิน 0.5% ใน {int((rel > 0.005).sum())}/{len(rel)} วันที่ทับ "
                     f"(สูงสุด {rel.max():.1%}) -- คาดไว้แล้ว: Investing ปัด 3 หลัก และ Yahoo แก้ Volume ย้อนหลังได้")
    return len(y_w), warns


def validate_new(ticker, new):
    if new.isna().any().any():
        raise FetchError(f"{ticker}: มีค่าว่างในแถวใหม่ {[str(d.date()) for d in new[new.isna().any(axis=1)].index]}")
    if (new.index.dayofweek > 4).any():
        raise FetchError(f"{ticker}: มีวันเสาร์-อาทิตย์ในแถวใหม่")
    if not new.index.is_unique:
        raise FetchError(f"{ticker}: วันที่ซ้ำในแถวใหม่")
    zero = new[new["Volume"] <= 0]
    if len(zero):
        raise FetchError(f"{ticker}: Volume <= 0 ใน {[str(d.date()) for d in zero.index]} "
                         "แต่ราคาไม่แบน -- ไม่ใช่แถวผีวันหยุดแบบปกติ หยุด ตรวจด้วยตาก่อน")


def _dec(x):
    """float -> Decimal สั้นสุดที่ปัดกลับเป็น float เดิม"""
    return Decimal(repr(float(x)))


def investing_rows(new, prev_close):
    """แปลงเป็นแถวรูปแบบ Investing (เรียงใหม่->เก่า เหมือนไฟล์ดาวน์โหลดจริง)"""
    rows, prev = [], _dec(prev_close)
    for d, r in new.iterrows():
        close = _dec(r["Close"])
        chg = ((close / prev - 1) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        vol_k = Decimal(int(r["Volume"])) / 1000
        vol_s = format(vol_k.normalize(), "f") + "K"
        rows.append([d.strftime("%m/%d/%Y")] + [f"{_dec(r[c]):.2f}" for c in ("Close", "Open", "High", "Low")]
                    + [vol_s, f"{chg}%"])
        prev = close
    return list(reversed(rows))


def write_investing_csv(path, rows):
    lines = [",".join(f'"{h}"' for h in INPUT_HEADER)] + [",".join(f'"{v}"' for v in r) for r in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plan_fetch(now, raw_dir=RAW_DATA_DIR, tickers=TICKERS, overlap_days=OVERLAP_DAYS, downloader=download_daily):
    """ดึง + ตรวจทุกหุ้นก่อน -- คืนแผนต่อหุ้น (ยังไม่เขียนอะไร)"""
    plans = []
    for tk in tickers:
        short = tk.replace(".BK", "")
        _, old = read_canonical(Path(raw_dir) / f"{short}{RAW_DATA_SUFFIX}")
        last = old.index[-1]
        start = (last - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")
        end_excl = (pd.Timestamp(now.date()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        y, phantom = drop_phantom(drop_unfinished(downloader(tk, start, end_excl), now))
        n_overlap, warns = verify_overlap(short, y, old)
        if phantom:
            warns.append(f"ตัดแถวผีวันหยุดของ Yahoo (Volume 0, ราคาแบน): {phantom}")
        new = y[y.index > last]
        validate_new(short, new)
        plans.append({"ticker": short, "last": last, "new": new, "n_overlap": n_overlap, "warnings": warns,
                      "prev_close": float(old["Price"].iloc[-1])})
    ends = {p["ticker"]: (p["new"].index[-1] if len(p["new"]) else p["last"]) for p in plans}
    if len(set(ends.values())) > 1:
        raise FetchError(f"วันสุดท้ายของแต่ละหุ้นไม่เท่ากัน {ends} -- ต้องอัปเดตทุกหุ้นพร้อมกัน")
    return plans


def main(argv=None):
    ap = argparse.ArgumentParser(description="ดึงข้อมูลรายวันจาก Yahoo แล้วเขียนเป็นไฟล์รูปแบบ Investing.com")
    ap.add_argument("--check-only", action="store_true", help="ดึง + ตรวจ ไม่เขียนไฟล์")
    ap.add_argument("--out-dir", type=Path, default=None, help="default = task_b/.staging/yahoo_<YYYYMMDD>/")
    ap.add_argument("--overlap-days", type=int, default=OVERLAP_DAYS)
    args = ap.parse_args(argv)

    now = datetime.now(BANGKOK)
    try:
        plans = plan_fetch(now, overlap_days=args.overlap_days)
    except FetchError as e:
        print(f"!! หยุด -- ไม่ได้เขียนไฟล์ใด ๆ\n   {e}")
        return 1

    for p in plans:
        n = p["new"]
        print(f"=== {p['ticker']}: ทับของเดิม {p['n_overlap']} วัน · OHLC ตรงเป๊ะ · "
              f"แถวใหม่ {len(n)}" + (f" ({n.index[0].date()} -> {n.index[-1].date()})" if len(n) else ""))
        for w in p["warnings"]:
            print(f"  ?? เตือน: {w}")

    if not any(len(p["new"]) for p in plans):
        print(f"\nไม่มีวันใหม่หลัง {plans[0]['last'].date()} (วันหยุด หรือยังไม่ถึง {FINAL_AFTER:%H:%M})")
        return EXIT_NO_NEW
    if args.check_only:
        print("\n[check-only] ผ่านการตรวจ -- ไม่ได้เขียนไฟล์ใด ๆ")
        return 0

    out = args.out_dir or STAGING_DIR / f"yahoo_{now:%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    for p in plans:
        n = p["new"]
        name = f"{p['ticker']} Yahoo daily {n.index[0]:%Y%m%d}-{n.index[-1]:%Y%m%d}.csv"
        write_investing_csv(out / name, investing_rows(n, p["prev_close"]))
        print(f"เขียน {out / name}")
    print(f"\nต่อไป: python tools/append_investing.py {out}/*.csv --source yahoo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
