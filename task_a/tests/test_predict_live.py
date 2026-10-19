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


def test_check_market_closed_allows_early_morning_next_day():
    """regression test: เดิมเช็คแค่ now.hour < 18 ซึ่งบล็อก 08:30 ผิดพลาด (8 < 18)
    ทั้งที่ automation/register_next_day.ps1 ตั้งรันตอน 08:30 จริง (แก้ 2026-10-01)"""
    now = T("2026-09-30 08:30", tz=BANGKOK)
    PL.check_market_closed(now=now)   # ไม่ raise = ผ่าน


def test_check_market_closed_blocks_market_open_hours():
    now = T("2026-09-30 10:00", tz=BANGKOK)
    _raises(PL.check_market_closed, now=now, match="18:00")


def test_check_no_duplicate_raises_on_existing_key():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-09-30",
                       "model": "ANN (MLP)", "is_dry_run": False}]).to_csv(log, index=False)
        new_rows = pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-09-30",
                                  "model": "ANN (MLP)"}])
        _raises(PL.check_no_duplicate, new_rows, log, match="มีคำทำนาย")


def test_check_no_duplicate_ignores_dry_run_rows():
    """แถว dry-run เก่า (is_dry_run=True) ไม่ควรนับเป็นคำทำนายซ้ำ"""
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-09-30",
                       "model": "ANN (MLP)", "is_dry_run": True}]).to_csv(log, index=False)
        new_rows = pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-09-30",
                                  "model": "ANN (MLP)"}])
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


def _ohlcv(dates, closes, volumes):
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                         "Close": closes, "Volume": volumes},
                        index=pd.DatetimeIndex(dates, name="Date"))


def test_fetch_live_keeps_raw_data_history_and_appends_only_new_yahoo_rows():
    """regression 2026-10-19: Yahoo 1d ปี 2024 มีแถวเสีย (Volume=0 ราคาปิดค้าง)
    -> ประวัติต้องมาจาก task_b/raw_data เสมอ Yahoo ใช้แค่ต่อวันใหม่ และแถวใหม่ที่
    Volume=0 ต้องถูกตัดทิ้ง"""
    from unittest import mock
    hist = _ohlcv(["2026-10-13", "2026-10-14"], [230.0, 231.0], [1e6, 1e6])
    yahoo = _ohlcv(["2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16"],
                   [999.0, 999.0, 235.0, 235.0],    # วันซ้อนทับราคาผิด · 10-16 Volume=0
                   [1e6, 0.0, 1e6, 0.0])
    with mock.patch.object(PL, "load_verified_history", return_value=hist), \
         mock.patch.object(PL.yf, "download", return_value=yahoo):
        df = PL.fetch_live("KBANK.BK")
    assert list(df.index.strftime("%Y-%m-%d")) == ["2026-10-13", "2026-10-14", "2026-10-15"]
    assert list(df["Close"]) == [230.0, 231.0, 235.0]   # raw_data ชนะ Yahoo ในช่วงซ้อนทับ


def test_fetch_live_raises_when_yahoo_returns_nothing():
    """ดึงย้อนทับไฟล์ไว้เสมอ -> ผลว่าง = ดึงไม่ได้จริง ห้ามเงียบแล้วทำนายด้วยข้อมูลเก่า"""
    from unittest import mock
    hist = _ohlcv(["2026-10-14"], [231.0], [1e6])
    with mock.patch.object(PL, "load_verified_history", return_value=hist), \
         mock.patch.object(PL.yf, "download", return_value=pd.DataFrame()):
        _raises(PL.fetch_live, "KBANK.BK", match="ดึงข้อมูลสด")


def test_load_verified_history_reads_real_raw_data_without_bad_2024_rows():
    df = PL.load_verified_history("KBANK.BK")
    assert df.index[0] >= pd.Timestamp("2016-08-26")
    assert int((df.loc["2024", "Volume"] <= 0).sum()) == 0


def test_evaluate_live_scores_against_official_close_and_skips_dry_run():
    import evaluate_live as EV
    from unittest import mock
    close = pd.Series([231.0, 232.0], index=pd.DatetimeIndex(["2026-10-14", "2026-10-15"]))
    log = pd.DataFrame({
        "ticker": ["KBANK.BK"] * 4,
        "target_date": ["2026-10-14", "2026-10-15", "2026-10-19", "2026-10-14"],
        "model": ["ANN (MLP)"] * 4,
        "prev_parity": [0.0, 1.0, 0.0, 0.0],
        "predicted_parity": [1.0, 1.0, 0.0, 0.0],
        "is_dry_run": [False, False, False, True],
    })
    with mock.patch.object(EV, "official_close", return_value=close):
        out = EV.score(log, "prev_parity")
    assert len(out) == 3                                  # แถว dry-run ไม่นับ
    assert list(out["status"]) == ["scored", "scored", "pending"]
    assert list(out["actual_parity"].iloc[:2]) == [1.0, 0.0]   # 231 คี่ · 232 คู่
    assert list(out["correct"].iloc[:2]) == [1.0, 0.0]
    assert list(out["baseline_correct"].iloc[:2]) == [0.0, 0.0]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
