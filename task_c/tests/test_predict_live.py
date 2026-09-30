"""
tests/test_predict_live.py — predict_live.py, predict_live_1600.py, intraday_task2.py

    python tests/test_predict_live.py

ไม่ใช้อินเทอร์เน็ต (mock yfinance ตรง load_daily_close) ไม่เขียนไฟล์นอก temp dir

*** เหตุผลที่มีไฟล์นี้ ***
task_c ไม่มี test เลยมาก่อน (2026-09-29) ระหว่างแก้บั๊ก target ของ Task 2
(ดู KNOWN_ISSUES.md) พบว่าไม่มีอะไรกันการถอยกลับไปใช้ตรรกะเดิม (แท่ง 1h ที่
Hour==16) โดยไม่ตั้งใจ — test_build_task2_dataset_prefers_daily_close_over_hourly_bar
คือ regression test ของบั๊กนั้นโดยตรง
"""

import sys
import tempfile
from datetime import timezone, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import predict_live as PL                    # noqa: E402
import predict_live_1600 as PL16              # noqa: E402
import intraday_task2 as T2                   # noqa: E402

T = pd.Timestamp
BANGKOK = timezone(timedelta(hours=7))


def _raises(fn, *a, exc=RuntimeError, match="", **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        assert match in str(e), f"ข้อความ error ไม่ตรง: {e}"
        return
    raise AssertionError(f"ต้อง raise {exc.__name__}")


# ---------------------------------------------------------------
# predict_live.py (next_day)
# ---------------------------------------------------------------

def test_check_market_closed_blocks_before_1800():
    now = T("2026-09-29 12:00", tz=BANGKOK)
    _raises(PL.check_market_closed, now=now, match="18:00")


def test_check_market_closed_allows_after_1800():
    now = T("2026-09-29 18:30", tz=BANGKOK)
    PL.check_market_closed(now=now)   # ไม่ raise = ผ่าน


def test_check_no_duplicate_raises_on_existing_key():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-09-30",
                       "is_dry_run": False}]).to_csv(log, index=False)
        new_rows = pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-09-30"}])
        _raises(PL.check_no_duplicate, new_rows, log, match="มีคำทำนาย")


def test_check_no_duplicate_passes_for_new_key():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-09-30",
                       "is_dry_run": False}]).to_csv(log, index=False)
        new_rows = pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-10-01"}])
        PL.check_no_duplicate(new_rows, log)   # ไม่ raise = ผ่าน


def test_check_no_duplicate_passes_when_log_missing():
    PL.check_no_duplicate(pd.DataFrame([{"ticker": "X", "target_date": "2026-01-01"}]),
                          Path("/nonexistent/log.csv"))


def test_append_log_rejects_schema_mismatch():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        pd.DataFrame([{"a": 1, "b": 2}]).to_csv(log, index=False)
        rows = pd.DataFrame([{"a": 1, "b": 2, "c": 3}])
        _raises(PL.append_log, rows, log, ["a", "b", "c"], match="schema")


# ---------------------------------------------------------------
# intraday_task2.py / predict_live_1600.py (t1600, Task 2)
# ---------------------------------------------------------------

def _make_hourly(dates_hours_closes):
    """dates_hours_closes: [(date_str, {hour: close}), ...] -> DataFrame แบบ load_hourly()"""
    rows = []
    for date_str, hours in dates_hours_closes:
        for hour, close in hours.items():
            rows.append({
                "Datetime": pd.Timestamp(f"{date_str} {hour:02d}:00", tz=BANGKOK),
                "Open": close, "High": close + 0.5, "Low": close - 0.5,
                "Close": close, "Volume": 1000,
            })
    df = pd.DataFrame(rows)
    df["Date"] = df["Datetime"].dt.normalize()
    df["Hour"] = df["Datetime"].dt.hour
    return df


def test_build_task2_dataset_prefers_daily_close_over_hourly_bar():
    """
    regression test ของบั๊กที่แก้ 2026-09-29: ก่อนหน้านี้ actual_close มาจาก
    แท่ง 1h ที่ Hour==16 (ตรวจกับ Settrade แล้วพบว่าตรง ATC จริงแค่ ~40-46%)
    ตอนนี้ต้องใช้ daily_close ที่ส่งเข้ามาแทนเสมอถ้ามี ไม่ใช่แท่ง 1h
    """
    hourly = _make_hourly([
        ("2024-01-02", {10: 100, 14: 101, 15: 102, 16: 103}),  # แท่ง 1h บอกว่า close=103
        ("2024-01-03", {10: 104, 14: 105, 15: 106, 16: 107}),
    ])
    # daily_close ตั้งใจให้ต่างจากแท่ง 1h ชัดๆ เพื่อพิสูจน์ว่าใช้ตัวนี้จริง
    daily_close = pd.Series(
        {pd.Timestamp("2024-01-02"): 999.0, pd.Timestamp("2024-01-03"): 888.0})

    X, y, details = T2.build_task2_dataset(hourly, daily_close=daily_close)
    assert details["actual_close"].loc[pd.Timestamp("2024-01-02", tz=BANGKOK)] == 999.0
    assert details["actual_close"].loc[pd.Timestamp("2024-01-03", tz=BANGKOK)] == 888.0
    # cutoff_price ต้องยังมาจากแท่ง 1h ตามปกติ (แค่ target ที่เปลี่ยน ไม่ใช่ feature)
    assert details["cutoff_price"].loc[pd.Timestamp("2024-01-02", tz=BANGKOK)] == 102


def test_build_task2_dataset_falls_back_to_hourly_bar_without_daily_close():
    """backward-compat: daily_close=None ต้องยังใช้แท่ง 1h ที่ Hour==close_hour ได้เหมือนเดิม"""
    hourly = _make_hourly([
        ("2024-01-02", {10: 100, 14: 101, 15: 102, 16: 103}),
        ("2024-01-03", {10: 104, 14: 105, 15: 106, 16: 107}),
    ])
    X, y, details = T2.build_task2_dataset(hourly, daily_close=None)
    assert details["actual_close"].loc[pd.Timestamp("2024-01-02", tz=BANGKOK)] == 103


def test_build_task2_dataset_skips_days_missing_from_daily_close():
    hourly = _make_hourly([
        ("2024-01-02", {10: 100, 14: 101, 15: 102, 16: 103}),
        ("2024-01-03", {10: 104, 14: 105, 15: 106, 16: 107}),
    ])
    daily_close = pd.Series({pd.Timestamp("2024-01-02"): 999.0})   # ขาดวัน 01-03
    X, y, details = T2.build_task2_dataset(hourly, daily_close=daily_close)
    assert len(details) == 1
    assert pd.Timestamp("2024-01-03", tz=BANGKOK) not in details.index


def test_load_daily_close_uses_cache_without_hitting_network():
    ticker = "FAKE.BK"
    with tempfile.TemporaryDirectory() as d, mock.patch(
            "os.getcwd", return_value=d):
        import os
        cwd = os.getcwd()
        cache_dir = Path(cwd) / "data_cache"
        cache_dir.mkdir()
        cached = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
        cached.to_csv(cache_dir / f"{ticker.replace('.', '_')}_daily_close.csv")

        real_path = str(cache_dir / f"{ticker.replace('.', '_')}_daily_close.csv")
        with mock.patch.object(T2, "_daily_close_path", return_value=real_path), \
             mock.patch("yfinance.download", side_effect=AssertionError(
                 "ไม่ควรเรียก yfinance ถ้า cache ครอบคลุมช่วงที่ต้องการแล้ว")):
            s = T2.load_daily_close(ticker, "2024-01-02", "2024-01-03")
    assert list(s.values) == [100.0, 101.0]


# ---------------------------------------------------------------
# predict_live_1600.py -- guard เวลา + duplicate/log ใช้ของเดียวกับ predict_live.py
# ---------------------------------------------------------------

def test_predict_live_1600_check_no_duplicate_raises():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-09-28",
                       "model": "ANN (MLP)", "is_dry_run": False}]).to_csv(log, index=False)
        new_rows = pd.DataFrame([{"ticker": "KBANK.BK", "target_date": "2026-09-28",
                                  "model": "ANN (MLP)"}])
        _raises(PL16.check_no_duplicate, new_rows, log, match="มีคำทำนาย")


def test_daily_volume_history_excludes_today_and_sums_full_day():
    hourly = _make_hourly([
        ("2024-01-02", {10: 100, 14: 101, 15: 102, 16: 103}),
        ("2024-01-03", {10: 104, 14: 105, 15: 106, 16: 107}),
    ])
    vol = PL16._daily_volume_history(hourly, pd.Timestamp("2024-01-03", tz=BANGKOK))
    assert list(vol.index) == [pd.Timestamp("2024-01-02", tz=BANGKOK)]
    assert vol.iloc[0] == 4000   # 4 แท่ง x 1000


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
