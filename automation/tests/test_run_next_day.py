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


def _daily(close, day="2026-10-12"):
    return lambda tk: pd.DataFrame({"Close": [close]}, index=pd.DatetimeIndex([day]))


def test_expected_cutoff_skips_weekend_and_holidays():
    """cutoff = วันทำการสุดท้ายก่อน target (ข้าม ส-อา และวันหยุด SET เช่น 10-13, 10-16)"""
    f = lambda t: str(R.expected_cutoff(pd.Timestamp(t), HOLIDAYS).date())
    assert f("2026-10-14") == "2026-10-12"      # 10-13 วันนวมินทรมหาราช
    assert f("2026-10-19") == "2026-10-15"      # จันทร์: ถอยข้าม ส-อา และ 10-16 (หยุดพิเศษ)
    assert f("2026-10-06") == "2026-10-05"


def test_yahoo_ready_blocks_nan_missing_and_errors():
    """regression: 2026-10-13 Yahoo ส่ง Close ของวันล่าสุดเป็น NaN หลายชั่วโมง -> ต้องไม่ปล่อยให้ task_a/c ทำนายด้วยข้อมูลเก่า"""
    c = pd.Timestamp("2026-10-12")
    assert R.yahoo_ready(c, _daily(231.0)) == (True, [])
    ok, why = R.yahoo_ready(c, _daily(float("nan")))
    assert not ok and len(why) == 2 and "NaN" in why[0]
    ok, why = R.yahoo_ready(c, _daily(231.0, day="2026-10-09"))
    assert not ok and "ยังไม่มีแถวของ 2026-10-12" in why[0]
    ok, why = R.yahoo_ready(c, lambda tk: (_ for _ in ()).throw(ConnectionError("x")))
    assert not ok and "ดึงข้อมูลไม่ได้" in why[0]


class _Patched:
    """แทน run/commit_log/yahoo_ready/already_logged ชั่วคราว -- เก็บชื่อขั้นที่ถูกเรียกไว้ใน .calls

    ต้องแทน already_logged ด้วย ไม่งั้น test อ่าน prediction log จริงในเรโป แล้วผลเปลี่ยนไปตาม
    ว่าวันที่รัน test มี log ของ target แล้วหรือยัง (เจอจริง 2026-10-15)
    """

    def __init__(self, yahoo_ok=True, logged=()):
        self.calls, self.yahoo_ok, self.logged = [], yahoo_ok, set(logged)

    def __enter__(self):
        self._orig = (R.run, R.commit_log, R.yahoo_ready, R.already_logged)
        R.run = lambda step, args, cwd: self.calls.append(step)
        R.commit_log = lambda step, cwd, *a, **k: self.calls.append(f"commit:{cwd.name}")
        R.yahoo_ready = lambda cutoff, *a, **k: (
            (True, []) if self.yahoo_ok else (False, ["KBANK.BK: Close ของ x ยังเป็น NaN"]))
        R.already_logged = lambda path, target, ptype=None: Path(path).parts[-3] in self.logged
        return self

    def __exit__(self, *exc):
        R.run, R.commit_log, R.yahoo_ready, R.already_logged = self._orig


def test_commit_run_stops_before_any_task_when_yahoo_not_ready():
    """--commit + ข้อมูลไม่พร้อม -> หยุดก่อนเรียก task ใด ๆ (ไม่มี log ครึ่งเดียว) · --dry-run ไม่หยุด"""
    with _Patched(yahoo_ok=False) as p:
        assert R.main(["--commit"]) == 1 and p.calls == []                 # หยุดก่อนรันอะไร
        p.calls.clear()
        assert R.main(["--commit", "--no-preflight"]) == 0 and "task_a" in p.calls   # ข้ามได้ตั้งใจเท่านั้น
        p.calls.clear()
        assert R.main(["--dry-run"]) == 0 and "task_a" in p.calls          # dry-run เดินต่อ


def test_resume_skips_tasks_that_already_have_log():
    """รอบก่อนถูกฆ่ากลางทาง (ได้ log task_a/task_c แล้ว) -> รอบถัดไปต้องข้ามสองตัวนั้น
    แล้วเดินไป task_b ให้ครบ ไม่ใช่ไปชน guard กันทำนายซ้ำแล้วหยุด (เหตุจริง 2026-10-15)"""
    with _Patched(logged={"task_a", "task_c"}) as p:
        assert R.main(["--commit"]) == 0
        assert p.calls == ["task_b"], p.calls


def test_fresh_day_runs_every_task_in_order():
    with _Patched() as p:
        assert R.main(["--commit"]) == 0
        assert p.calls == ["task_a", "commit:task_a", "task_c", "commit:task_c", "task_b"], p.calls


def test_dry_run_never_skips_even_when_official_log_exists():
    """dry-run เขียนคนละไฟล์ -- ต้องรันได้ซ้ำไม่จำกัด แม้ log official ของ target มีแล้ว"""
    with _Patched(logged={"task_a", "task_c"}) as p:
        assert R.main(["--dry-run"]) == 0
        assert p.calls == ["task_a", "task_c", "task_b"], p.calls


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
