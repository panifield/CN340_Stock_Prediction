"""
tools/check_raw_update.py — ตรวจไฟล์ raw_data ชุดใหม่ก่อน commit (§1.6)
=========================================================================

    python tools/check_raw_update.py <โฟลเดอร์เก่า> <โฟลเดอร์ใหม่>
    python tools/check_raw_update.py raw_data_backup_20260828 raw_data

ตรวจ:
  1. ทุก ticker: inner join บน Date ของช่วงที่ทับกัน แล้วเทียบ
     Open / High / Low / Close / Volume ให้ "เท่ากันเป๊ะ"
     -> ต่างแม้แถวเดียว: พิมพ์แถวนั้นทั้งหมดแล้ว exit(1)
  2. รายงานจำนวนแถวเก่า/ใหม่/ที่เพิ่ม, วันสุดท้ายเก่า -> ใหม่, วันที่เพิ่มทั้งหมด
  3. เตือนถ้าเจอวันที่ซ้ำ หรือวันที่ไม่เรียงจากเก่าไปใหม่

ทำไมต้องตรวจช่วงทับ: fit_live_models() เทรนจากข้อมูล "ทั้งหมด"
ถ้า investing.com revise ราคาย้อนหลัง โมเดล live จะเปลี่ยนโดยไม่รู้ตัว
ส่วน regression check ด้วย main.py --dev ตรวจได้แค่ถึง val_end จึงไม่พอ

option --cross-check raw_data_intraday/ (§6.7):
  เทียบราคาปิดของ "วันที่เพิ่มใหม่" กับ close_bar16 จากไฟล์รายชั่วโมง (Yahoo)
  วันไหนต่างเกิน ~1 บาท พิมพ์ออกมาให้ตรวจด้วยตา
  -> เป็นคำเตือนเท่านั้น ไม่กระทบ exit code (รู้อยู่แล้วว่าสองแหล่งต่างกัน)
  *** ไม่เอาข้อมูล Yahoo ไปเติม raw_data/ เด็ดขาด -- ใช้เป็นเครื่องตรวจทานอย่างเดียว ***

exit code: 0 = ผ่าน, 1 = ข้อมูลเก่าถูกแก้ / ไฟล์หาย / อ่านไม่ได้
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ใช้นิยามรูปแบบไฟล์ตัวเดียวกับ data_loader -- ห้ามเขียนรูปแบบซ้ำ
from config import RAW_DATA_SUFFIX
from data_loader import INVESTING_DATE_FORMAT, INVESTING_VOLUME_COL

CROSS_CHECK_WARN_BAHT = 1.0

COMPARE = {"Open": "Open", "High": "High", "Low": "Low",
           "Close": "Price", "Volume": INVESTING_VOLUME_COL}


def read_raw(path):
    """อ่านไฟล์ investing.com ตามลำดับในไฟล์ (ยังไม่ sort -- จะได้ตรวจลำดับได้)"""
    raw = pd.read_csv(path)
    missing = [c for c in ["Date", *COMPARE.values()] if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} ไม่มีคอลัมน์ {missing}")
    df = pd.DataFrame({k: pd.to_numeric(raw[v]) for k, v in COMPARE.items()})
    df.index = pd.to_datetime(raw["Date"], format=INVESTING_DATE_FORMAT)
    df.index.name = "Date"
    return df


def warn_order(name, df):
    """เตือนวันที่ซ้ำ / ไม่ monotonic -- เป็นคำเตือน ไม่ fail"""
    dup = df.index[df.index.duplicated()]
    if len(dup):
        print(f"  !! เตือน {name}: วันที่ซ้ำ {len(dup)} วัน: "
              f"{[str(d.date()) for d in dup.unique()]}")
    if not df.index.is_monotonic_increasing:
        bad = [str(df.index[i].date()) for i in range(1, len(df))
               if df.index[i] <= df.index[i - 1]]
        print(f"  !! เตือน {name}: วันที่ไม่เรียงจากเก่าไปใหม่ {len(bad)} จุด "
              f"(เช่น {bad[:5]})")


def cross_check(new_path, added, intraday_dir):
    """
    เทียบ Close ของวันที่เพิ่มใหม่กับ close_bar16 (Yahoo) -- คำเตือนเท่านั้น (§6.7)
    ไม่คืนค่า pass/fail: สองแหล่งต่างกันเฉลี่ย 0.33-0.69 บาทอยู่แล้ว (§6.2)
    """
    from intraday_probe import read_hourly, bars_by_hour   # อยู่ใน tools/ เดียวกัน

    stem = new_path.name.replace(RAW_DATA_SUFFIX, "")
    files = sorted(intraday_dir.glob(f"{stem}_BK_1h*.csv"))
    if not files:
        print(f"  [cross-check] ไม่พบไฟล์รายชั่วโมงของ {stem} ใน {intraday_dir} -- ข้าม")
        return
    c16 = bars_by_hour(read_hourly(files[0]))[16].dropna()
    new = read_raw(new_path)
    new = new[~new.index.duplicated()]
    days = added.intersection(c16.index)
    if not len(days):
        print(f"  [cross-check] วันที่เพิ่มใหม่ไม่ทับกับ {files[0].name} -- ไม่มีอะไรให้เทียบ")
        return
    diff = (new.loc[days, "Close"] - c16.loc[days]).abs()
    far = diff[diff > CROSS_CHECK_WARN_BAHT]
    print(f"  [cross-check] วันที่เพิ่มใหม่ {len(days)} วัน vs close_bar16 ({files[0].name}): "
          f"mean|ต่าง| {diff.mean():.4f} บาท · ต่างเกิน {CROSS_CHECK_WARN_BAHT:g} บาท "
          f"{len(far)} วัน")
    for d, v in far.items():
        print(f"    ?? เตือน {d.date()}: investing {new.loc[d, 'Close']} "
              f"vs close_bar16 {c16.loc[d]} (ต่าง {v:.2f} บาท) -- ตรวจด้วยตา")


def check_ticker(old_path, new_path, intraday_dir=None):
    """คืน True ถ้าผ่าน (--cross-check เป็นคำเตือน ไม่กระทบผล)"""
    print(f"\n=== {old_path.name} ===")
    if not new_path.exists():
        print(f"  !! ไม่พบไฟล์ใหม่ {new_path}")
        return False

    old, new = read_raw(old_path), read_raw(new_path)
    warn_order("ไฟล์เก่า", old)
    warn_order("ไฟล์ใหม่", new)

    # ตัดวันที่ซ้ำ (เก็บตัวแรก) ก่อน join -- ได้เตือนไปแล้วด้านบน
    old_u = old[~old.index.duplicated()].sort_index()
    new_u = new[~new.index.duplicated()].sort_index()

    both = old_u.join(new_u, how="inner", lsuffix="_old", rsuffix="_new")
    diff_mask = pd.Series(False, index=both.index)
    for c in COMPARE:
        diff_mask |= both[f"{c}_old"] != both[f"{c}_new"]

    added = new_u.index.difference(old_u.index)
    removed = old_u.index.difference(new_u.index)

    print(f"  แถว      : เก่า {len(old)}  ->  ใหม่ {len(new)}  "
          f"(เพิ่ม {len(added)}, ทับกัน {len(both)})")
    print(f"  วันสุดท้าย: {old_u.index[-1].date()}  ->  {new_u.index[-1].date()}")
    print(f"  วันที่เพิ่ม: {[str(d.date()) for d in added] if len(added) else '(ไม่มี)'}")
    if len(removed):
        print(f"  !! เตือน: วันที่มีในไฟล์เก่าแต่หายจากไฟล์ใหม่ {len(removed)} วัน: "
              f"{[str(d.date()) for d in removed]}")

    if intraday_dir is not None:
        cross_check(new_path, added, intraday_dir)

    if diff_mask.any():
        print(f"  !! ข้อมูลช่วงที่ทับกันถูกแก้ {int(diff_mask.sum())} แถว:")
        cols = [f"{c}_{s}" for c in COMPARE for s in ("old", "new")]
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(both.loc[diff_mask, cols].to_string())
        return False

    print("  ผ่าน: ช่วงที่ทับกันเท่ากันเป๊ะทุกแถวทุกคอลัมน์")
    return True


def main():
    p = argparse.ArgumentParser(description="ตรวจ raw_data ชุดใหม่เทียบชุดเก่า")
    p.add_argument("old_dir", type=Path, help="โฟลเดอร์ raw_data เดิม (backup)")
    p.add_argument("new_dir", type=Path, help="โฟลเดอร์ raw_data ใหม่")
    p.add_argument("--cross-check", type=Path, default=None, metavar="INTRADAY_DIR",
                   help="เทียบวันที่เพิ่มใหม่กับ close_bar16 ใน raw_data_intraday/ "
                        "(คำเตือนเท่านั้น)")
    args = p.parse_args()

    files = sorted(args.old_dir.glob(f"*{RAW_DATA_SUFFIX}"))
    if not files:
        print(f"!! ไม่พบไฟล์ *{RAW_DATA_SUFFIX} ใน {args.old_dir}")
        sys.exit(1)

    ok = all([check_ticker(f, args.new_dir / f.name, args.cross_check) for f in files])
    print("\n" + ("ผ่านทุกไฟล์" if ok else "!! ไม่ผ่าน -- ห้าม commit ข้อมูลชุดนี้ หาสาเหตุก่อน"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
