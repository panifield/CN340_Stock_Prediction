"""
tests/test_run_next_day.py — automation/run_next_day.py

    python tests/test_run_next_day.py

ไม่ใช้อินเทอร์เน็ต ไม่แตะ git

*** เหตุผลที่มีไฟล์นี้ ***
automation/ ไม่มี test มาก่อน (2026-10-01) — พบบั๊กจริงตอนทดสอบ dry-run รอบ
08:30: target_date เพี้ยนไป 1 วันทำการเมื่อรันข้ามเที่ยงคืนตอนเช้ามืด (เช่น
automation/register_next_day.ps1 ตั้งไว้ 08:30) เพราะ target เดิมคำนวณจาก
"วันนี้ตามนาฬิกา" เสมอ ไม่ได้สนใจว่าตลาดของวันนี้เปิดไปหรือยัง
test_target_date_matches_regardless_of_run_time คือ regression test ของบั๊กนั้น
โดยตรง
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_next_day as R  # noqa: E402

HOLIDAYS, THROUGH = R.read_holidays(R.TASK_B_HOLIDAYS)


def _target_for(now_str):
    now = pd.Timestamp(now_str, tz="Asia/Bangkok")
    today = now.normalize().tz_localize(None)
    ref = today - pd.Timedelta(days=1) if now.hour < R.MARKET_OPEN_HOUR else today
    return R.next_trading_day(ref, HOLIDAYS, THROUGH)


def test_target_date_matches_regardless_of_run_time():
    """รอบเย็นของวันอังคาร กับรอบเช้าของวันพุธถัดมา ต้องได้ target_date เดียวกัน
    (ทำนายวันเดียวกัน ด้วยข้อมูลปิดตลาดวันอังคารเดียวกัน) -- เดิมรอบเช้าจะเพี้ยน
    ไปอีก 1 วันทำการ (แก้ 2026-10-01)"""
    evening = _target_for("2026-09-29 18:30")   # เย็นวันอังคาร
    next_morning = _target_for("2026-09-30 08:30")  # เช้าวันพุธถัดมา
    assert evening == next_morning == pd.Timestamp("2026-09-30")


def test_target_date_morning_after_weekend_skips_to_monday():
    morning = _target_for("2026-10-05 08:30")   # เช้าวันจันทร์ (หลังเสาร์-อาทิตย์)
    assert morning == pd.Timestamp("2026-10-05")


def test_target_date_evening_still_next_trading_day():
    evening = _target_for("2026-09-29 19:00")
    assert evening == pd.Timestamp("2026-09-30")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
