"""
deploy/precheck.py -- ตรวจก่อนรันงานจริง (เรียกจาก deploy/railway_run.sh ตอน DRY_RUN=0 -- ไม่เขียนอะไรเลย)
=========================================================================================================

    python deploy/precheck.py next_day --now "2026-10-14 08:30"
    python deploy/precheck.py t1600    --now "2026-10-14 16:02"

ทำไมต้องมี: ตัวคุม (automation/run_*.py) รัน task หลายตัวต่อกันและ commit ทีละตัว
  1) next_day: Yahoo บางเช้ายังส่ง Close ของวันล่าสุดเป็น NaN -> task_a/task_c ถอย data_cutoff ไปใช้ข้อมูลเก่ากว่า 1 วันทำการ
     "แล้วเขียน log ต่อ" แต่ task_b หยุดตามที่ออกแบบ => ได้ log ครึ่ง ๆ กลาง ๆ และรันซ้ำไม่ได้ (task_a ชน guard กันทำนายซ้ำ
     ก่อนจะถึง task_b) -> ตรวจข้อมูลให้พร้อมก่อน ถ้ายังไม่พร้อมก็ไม่รันอะไรเลย แล้วให้ cron ลองใหม่ในรอบถัดไป
  2) cron ถี่หลายรอบ/วัน -> รอบหลังต้องรู้ว่า "วันนี้ทำเสร็จแล้ว" และไม่รันซ้ำ

exit code:  0 = พร้อม รันต่อได้ · 10 = วันนี้ทำเสร็จครบทุก task แล้ว · 11 = ข้อมูลยังไม่พร้อม
            12 = ทำไปแล้วบางส่วน (ต้องมีคนดู ห้ามรันซ้ำอัตโนมัติ) · 13 = ตรวจไม่ได้ (error)
ใช้ฟังก์ชันคำนวณวันของ automation/run_next_day.py ตัวเดียวกัน (ไม่เขียนซ้ำ) เพื่อให้ target ตรงกับที่ตัวคุมใช้เสมอ
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "automation"))
import run_next_day as RND  # noqa: E402  (read_holidays / next_trading_day / MARKET_OPEN_HOUR)

EXIT_READY, EXIT_DONE, EXIT_NOT_READY, EXIT_PARTIAL, EXIT_ERROR = 0, 10, 11, 12, 13

# โหมด -> [(ไฟล์ log จริง, prediction_type ที่กรอง หรือ None)] ของทุก task ที่ตัวคุมต้องเขียน
LOGS = {
    "next_day": [("task_a/results/prediction_log.csv", None),
                 ("task_c/results/prediction_log.csv", None),
                 ("task_b/results/prediction_log.csv", "next_day")],
    "t1600": [("task_a2/results/prediction_log.csv", None),
              ("task_b/results/prediction_log.csv", "same_day_1600"),
              ("task_c/results/prediction_log_1600.csv", None)],
}


def target_and_cutoff(mode, now, holidays, through):
    """คืน (target_date, expected_cutoff) -- next_day ตรงกับ run_next_day.main() · t1600 target = วันนี้"""
    today = now.normalize().tz_localize(None) if now.tzinfo else now.normalize()
    if mode == "t1600":
        return today, None
    reference = today - pd.Timedelta(days=1) if now.hour < RND.MARKET_OPEN_HOUR else today
    target = RND.next_trading_day(reference, holidays, through)
    return target, RND.expected_cutoff(target, holidays)


def log_status(root, mode, target):
    """คืน (จำนวน task ที่มีแถว official ของ target แล้ว, จำนวน task ทั้งหมด, รายชื่อที่มีแล้ว)"""
    have = []
    for rel, ptype in LOGS[mode]:
        p = Path(root) / rel
        if not p.exists():
            continue
        d = pd.read_csv(p, usecols=lambda c: c in {"target_date", "is_dry_run", "prediction_type"})
        d = d[~d["is_dry_run"].astype(str).str.lower().isin(["true", "1"])]
        if ptype is not None:
            d = d[d["prediction_type"] == ptype]
        if (d["target_date"].astype(str) == target.date().isoformat()).any():
            have.append(rel.split("/")[0] + ("" if ptype is None else f"[{ptype}]"))
    return len(have), len(LOGS[mode]), have


# ตรวจ Yahoo ใช้ฟังก์ชันตัวเดียวกับ automation/run_next_day.py (preflight ของตัวคุมเอง) -- ไม่เขียนซ้ำ
download_daily = RND.download_daily
yahoo_ready = RND.yahoo_ready


def main(argv=None, root=ROOT, downloader=download_daily):
    ap = argparse.ArgumentParser(description="ตรวจก่อนรันงานจริง (ไม่เขียนอะไร)")
    ap.add_argument("mode", choices=["next_day", "t1600"])
    ap.add_argument("--now", required=True, help='เวลาไทย "YYYY-MM-DD HH:MM"')
    ap.add_argument("--no-yahoo", action="store_true", help="ข้ามการตรวจ Yahoo")
    a = ap.parse_args(argv)

    now = pd.Timestamp(a.now)
    try:
        holidays, through = RND.read_holidays(Path(root) / "task_b" / "set_holidays.txt")
        target, cutoff = target_and_cutoff(a.mode, now, holidays, through)
    except Exception as e:                          # noqa: BLE001
        print(f"[precheck] คำนวณวันไม่ได้: {e}")
        return EXIT_ERROR

    n_have, n_all, have = log_status(root, a.mode, target)
    print(f"[precheck] mode={a.mode} target={target.date()}"
          + (f" expected_cutoff={cutoff.date()}" if cutoff is not None else "")
          + f" · log จริงของ target มีแล้ว {n_have}/{n_all} task {have}")
    if n_have == n_all:
        print("[precheck] วันนี้ทำเสร็จครบทุก task แล้ว -- ไม่รันซ้ำ")
        return EXIT_DONE
    if n_have > 0:
        print(f"[precheck] !! ทำไปแล้วบางส่วน ({have}) -- รันซ้ำอัตโนมัติไม่ได้ (task แรกจะชน guard กันทำนายซ้ำ "
              "แล้ว task ที่เหลือไม่ถูกรัน) ต้องมีคนดูและรัน task ที่เหลือเอง")
        return EXIT_PARTIAL

    if a.mode == "next_day" and not a.no_yahoo:
        ok, problems = yahoo_ready(cutoff, downloader)
        if not ok:
            print("[precheck] ข้อมูลของวัน cutoff ยังไม่พร้อม:")
            for p in problems:
                print(f"           - {p}")
            return EXIT_NOT_READY
        print(f"[precheck] Yahoo มี Close ของ {cutoff.date()} ครบทั้ง 2 หุ้น")
    return EXIT_READY


if __name__ == "__main__":
    sys.exit(main())
