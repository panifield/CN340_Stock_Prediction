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


def check_ticker(old_path, new_path):
    """คืน True ถ้าผ่าน"""
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
    args = p.parse_args()

    files = sorted(args.old_dir.glob(f"*{RAW_DATA_SUFFIX}"))
    if not files:
        print(f"!! ไม่พบไฟล์ *{RAW_DATA_SUFFIX} ใน {args.old_dir}")
        sys.exit(1)

    ok = all([check_ticker(f, args.new_dir / f.name) for f in files])
    print("\n" + ("ผ่านทุกไฟล์" if ok else "!! ไม่ผ่าน -- ห้าม commit ข้อมูลชุดนี้ หาสาเหตุก่อน"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
