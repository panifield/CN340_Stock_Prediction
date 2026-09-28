"""
tests/test_predict_live.py — predict_live.py

    python tests/test_predict_live.py

ไม่ใช้อินเทอร์เน็ต ไม่เขียนไฟล์นอก temp dir

*** เหตุผลที่มีไฟล์นี้ ***
task_a2 ไม่มี test มาก่อน (2026-09-29) — check_has_1500_bar/check_time_window
ถูกแยกออกจาก predict_one_ticker() ในวันเดียวกัน (behavior เดิมทุกอย่าง แค่
แยกให้ test ได้โดยไม่ต้องดึงข้อมูลสดจริง) ไฟล์นี้ทดสอบทั้งสองฟังก์ชันนั้น
"""

import sys
import tempfile
from datetime import timezone, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import predict_live as PL   # noqa: E402

T = pd.Timestamp
BANGKOK = timezone(timedelta(hours=7))


def _raises(fn, *a, exc=RuntimeError, match="", **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        assert match in str(e), f"ข้อความ error ไม่ตรง: {e}"
        return
    raise AssertionError(f"ต้อง raise {exc.__name__}")


def _bars_index(*hhmm_list, date="2026-09-28"):
    return pd.DatetimeIndex([T(f"{date} {hhmm}", tz=BANGKOK) for hhmm in hhmm_list])


class _FakeBars:
    """แค่ .index ก็พอสำหรับ check_has_1500_bar (ไม่ได้ใช้ค่าอื่นของ df)"""
    def __init__(self, index):
        self.index = index


# ---------------------------------------------------------------
# check_has_1500_bar
# ---------------------------------------------------------------

def test_check_has_1500_bar_passes_official_when_bar_exists():
    bars = _FakeBars(_bars_index("10:00", "14:00", "15:00"))
    now = T("2026-09-28 16:05", tz=BANGKOK)
    PL.check_has_1500_bar(bars, T("2026-09-28").date(), now, allow_incomplete=False)


def test_check_has_1500_bar_raises_official_when_missing():
    bars = _FakeBars(_bars_index("10:00", "14:00"))   # ไม่มี 15:00
    now = T("2026-09-28 14:30", tz=BANGKOK)
    _raises(PL.check_has_1500_bar, bars, T("2026-09-28").date(), now, False,
           match="ยังไม่มีแท่ง 15:00")


def test_check_has_1500_bar_allows_incomplete_when_dry_run():
    bars = _FakeBars(_bars_index("10:00"))
    now = T("2026-09-28 11:00", tz=BANGKOK)
    PL.check_has_1500_bar(bars, T("2026-09-28").date(), now, allow_incomplete=True)


# ---------------------------------------------------------------
# check_time_window
# ---------------------------------------------------------------

def test_check_time_window_passes_within_1600_1630():
    PL.check_time_window(T("2026-09-28 16:15", tz=BANGKOK), allow_incomplete=False)


def test_check_time_window_raises_before_1600():
    _raises(PL.check_time_window, T("2026-09-28 15:59", tz=BANGKOK), False,
           match="16:00-16:30")


def test_check_time_window_raises_at_or_after_1630():
    _raises(PL.check_time_window, T("2026-09-28 16:30", tz=BANGKOK), False,
           match="16:00-16:30")


def test_check_time_window_bypassed_by_dry_run():
    PL.check_time_window(T("2026-09-28 01:00", tz=BANGKOK), allow_incomplete=True)


# ---------------------------------------------------------------
# check_no_duplicate / append_log / commit tracking
# ---------------------------------------------------------------

def test_check_no_duplicate_raises_on_existing_key():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-09-28",
                       "is_dry_run": False}]).to_csv(log, index=False)
        new_rows = pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-09-28"}])
        _raises(PL.check_no_duplicate, new_rows, log, match="มีคำทำนาย")


def test_append_log_rejects_schema_mismatch():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        pd.DataFrame([{"a": 1, "b": 2}]).to_csv(log, index=False)
        rows = pd.DataFrame([{"a": 1, "b": 2, "c": 3}])
        _raises(PL.append_log, rows, log, ["a", "b", "c"], match="schema")


def test_get_code_commit_returns_nonempty_hash_in_this_repo():
    commit = PL.get_code_commit()
    assert len(commit) >= 7, f"คาดว่าได้ commit hash แต่ได้ {commit!r}"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
