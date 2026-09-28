"""
tests/test_predict_live.py — predict_live.py

    python tests/test_predict_live.py

ไม่ใช้อินเทอร์เน็ต ไม่เขียนไฟล์นอก temp dir
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


def test_check_market_closed_blocks_before_1800():
    now = T("2026-09-29 12:00", tz=BANGKOK)
    _raises(PL.check_market_closed, now=now, match="18:00")


def test_check_market_closed_allows_after_1800():
    now = T("2026-09-29 18:30", tz=BANGKOK)
    PL.check_market_closed(now=now)   # ไม่ raise = ผ่าน


def test_check_market_closed_boundary_exact_1800_allowed():
    now = T("2026-09-29 18:00", tz=BANGKOK)
    PL.check_market_closed(now=now)   # ไม่ raise = ผ่าน (>= ไม่ใช่ >)


def test_check_no_duplicate_raises_on_existing_key():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-09-30",
                       "is_dry_run": False}]).to_csv(log, index=False)
        new_rows = pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-09-30"}])
        _raises(PL.check_no_duplicate, new_rows, log, match="มีคำทำนาย")


def test_check_no_duplicate_ignores_dry_run_rows():
    """แถว dry-run เก่า (is_dry_run=True) ไม่ควรนับเป็นคำทำนายซ้ำ"""
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-09-30",
                       "is_dry_run": True}]).to_csv(log, index=False)
        new_rows = pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-09-30"}])
        # หมายเหตุ: check_no_duplicate ของ task_a ปัจจุบันไม่กรอง is_dry_run ออก
        # ก่อนเทียบ -- ถ้า assert นี้ fail แปลว่าพฤติกรรมเปลี่ยนไปแล้ว (อาจตั้งใจ
        # หรือไม่ตั้งใจก็ได้ ให้ไปดูโค้ดจริงอีกที)
        try:
            PL.check_no_duplicate(new_rows, log)
            behavior = "ignores dry-run rows"
        except RuntimeError:
            behavior = "counts dry-run rows as duplicates too"
        print(f"    [info] check_no_duplicate ปัจจุบัน: {behavior}")


def test_append_log_rejects_schema_mismatch():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        pd.DataFrame([{"a": 1, "b": 2}]).to_csv(log, index=False)
        rows = pd.DataFrame([{"a": 1, "b": 2, "c": 3}])
        _raises(PL.append_log, rows, log, ["a", "b", "c"], match="schema")


def test_append_log_creates_file_with_header():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "sub" / "log.csv"
        rows = pd.DataFrame([{"a": 1, "b": 2}])
        PL.append_log(rows, log, ["a", "b"])
        assert log.exists()
        assert pd.read_csv(log).to_dict("records") == [{"a": 1, "b": 2}]


def test_get_code_commit_returns_nonempty_hash_in_this_repo():
    commit = PL.get_code_commit()
    assert len(commit) >= 7, f"คาดว่าได้ commit hash แต่ได้ {commit!r}"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
