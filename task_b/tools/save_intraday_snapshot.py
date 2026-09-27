"""
tools/save_intraday_snapshot.py — เก็บไฟล์รายชั่วโมงที่ดาวน์โหลดมา เป็น live snapshot (Phase 1D)
===============================================================================================

    python tools/save_intraday_snapshot.py KBANK.BK <ไฟล์.csv> --source yahoo
    python tools/save_intraday_snapshot.py ADVANC.BK <ไฟล์.csv> --source yahoo --downloaded-at 2026-09-29T16:02:10+07:00

จุดเชื่อมสำหรับตัวดึงข้อมูล (เช่น connector กลาง): ดึงข้อมูลเสร็จ -> เขียน CSV -> เรียกสคริปต์นี้
ไฟล์ถูกคัดลอก byte-for-byte เข้า raw_data_intraday_live/<YYYYMMDD>/ พร้อม .meta.json
(ticker, source, downloaded_at, sha256) · ห้ามแก้ไฟล์หลังบันทึก (predict_1600 ตรวจ SHA)

รูปแบบ CSV: Datetime(+07:00, เวลาเริ่มแท่ง),Adj Close,Close,High,Low,Open,Volume
--downloaded-at: เวลาที่ดาวน์โหลดจริง (default = ตอนนี้) -- ต้องเป็นเวลาจริง ห้ามใส่ย้อนหลังเพื่อให้ผ่าน guard
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_1600 import SnapshotError, load_snapshot, save_snapshot   # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description="บันทึก live snapshot รายชั่วโมง")
    p.add_argument("ticker", help="KBANK.BK หรือ ADVANC.BK")
    p.add_argument("file", type=Path)
    p.add_argument("--source", required=True, help="แหล่งข้อมูล เช่น yahoo")
    p.add_argument("--downloaded-at", default=None, help="ISO +07:00 (default = ตอนนี้)")
    args = p.parse_args(argv)
    try:
        dst = save_snapshot(args.file, args.ticker, args.source, args.downloaded_at)
    except SnapshotError as e:
        print(f"!! {e}")
        return 1
    bars, meta = load_snapshot(dst)
    print(f"[snapshot] {dst}\n  source={meta['source']} downloaded_at={meta['downloaded_at'].isoformat()}"
          f"\n  sha256={meta['sha256']}\n  แท่ง {len(bars)} · วันสุดท้าย {bars['date'].max().date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
