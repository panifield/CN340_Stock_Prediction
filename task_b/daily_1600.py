"""
daily_1600.py — ขั้นตอนประจำวันของโมเดล 16:00 (Phase 1D Mode A) ครบในคำสั่งเดียว
===============================================================================

    python daily_1600.py --phase predict --official --commit --push   # 16:00–16:25 วันทำการ (ใช้จริง)
    python daily_1600.py --phase outcome --official --commit --push   # หลัง 17:00: ผลจริง + ตรวจแท่ง 15:00
    python daily_1600.py --phase predict                              # dry-run (ทดสอบ ไม่เขียน production log)

ขั้นตอน:
  predict : tools/fetch_yahoo_intraday.py -> predict_1600.py --latest-snapshot (--dry-run | --official)
            -> tools/check_snapshot_1600.py
  outcome : tools/fetch_yahoo_intraday.py -> tools/check_snapshot_1600.py (bar15_changed_vs_later)
            -> record_outcomes.py (เฉพาะ --official)
  --commit: commit snapshot + ผล · --push: push ต่อ

ไม่ข้าม guard ใดของ predict_1600.py (เวลา 16:00–16:30 · commit แล้ว · snapshot หลัง 16:00 · ห้ามซ้ำ)
วันเสาร์-อาทิตย์/วันหยุดใน set_holidays.txt -> ข้ามทั้งหมด
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from daily_next_day import HOLIDAYS_PATH, StepError, git, read_holidays, run
from intraday_1600 import BANGKOK, BASE_DIR

TOOLS = BASE_DIR / "tools"
PY = sys.executable


def is_trading_day(day, holidays_path=HOLIDAYS_PATH):
    """คืน (เปิดทำการไหม, ข้อความเตือน)"""
    holidays, through = read_holidays(holidays_path)
    if day.dayofweek > 4 or day in holidays:
        return False, None
    warn = None
    if day > through:
        warn = (f"{day.date()} เกิน confirmed_through ({through.date()}) ของ {Path(holidays_path).name} "
                "-- ถ้าวันนี้เป็นวันหยุด ผลวันนี้จะไม่ผ่านเกณฑ์ (ไม่มีแท่งของวันนี้)")
    return True, warn


def main(argv=None):
    ap = argparse.ArgumentParser(description="ขั้นตอนประจำวันของโมเดล 16:00")
    ap.add_argument("--phase", choices=["predict", "outcome"], required=True)
    ap.add_argument("--official", action="store_true",
                    help="ทำนายแบบ official (predict_1600.py ตรวจเวลา/commit/snapshot เอง)")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--holidays", type=Path, default=HOLIDAYS_PATH)
    args = ap.parse_args(argv)
    if args.push and not args.commit:
        ap.error("--push ต้องใช้กับ --commit")

    now = datetime.now(BANGKOK)
    today = pd.Timestamp(now.date())
    try:
        open_, warn = is_trading_day(today, args.holidays)
        if not open_:
            print(f"{today.date()} ไม่ใช่วันทำการ -- ข้าม")
            return 0
        if warn:
            print(f"?? เตือน: {warn}")
        if args.phase == "predict" and not (16 <= now.hour and (now.hour, now.minute) < (16, 30)):
            print(f"?? เตือน: ตอนนี้ {now:%H:%M} อยู่นอกช่วง 16:00–16:30 -- official จะถูกปฏิเสธ")
        if args.phase == "outcome" and now.hour < 17:
            raise StepError(f"phase outcome ต้องรันหลัง 17:00 (ตอนนี้ {now:%H:%M})")

        run("1 fetch", [PY, TOOLS / "fetch_yahoo_intraday.py"])
        if args.phase == "predict":
            mode = "--official" if args.official else "--dry-run"
            run("2 predict", [PY, "predict_1600.py", "--target-date", today.date().isoformat(),
                              mode, "--latest-snapshot"])
        run("3 availability", [PY, TOOLS / "check_snapshot_1600.py", "--date", today.date().isoformat()])
        if args.phase == "outcome" and args.official:
            run("4 outcomes", [PY, "record_outcomes.py"])

        if args.commit:
            paths = ["raw_data_intraday_live", "results"]
            git("add", *paths)
            if git("diff", "--cached", "--quiet", check=False).returncode != 0:
                git("commit", "-m", f"1600 {args.phase}{' official' if args.official else ' dry-run'}: {today.date()}")
            if args.push:
                git("push")
    except (StepError, subprocess.SubprocessError) as e:
        print(f"\n!! {e}")
        return 1
    print("\nเสร็จ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
