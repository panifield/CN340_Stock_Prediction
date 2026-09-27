"""
tests/test_daily.py — daily_next_day.py / daily_1600.py (วันทำการ) · tools/check_env.py ·
                      tools/fetch_yahoo_intraday.py (รูปแบบ)

    python tests/test_daily.py

ไม่ใช้อินเทอร์เน็ต · ไม่เขียนไฟล์นอก temp dir
"""

import sys
import tempfile
from importlib import metadata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import check_env as E              # noqa: E402
import daily_1600 as D16           # noqa: E402
import daily_next_day as D         # noqa: E402
import fetch_yahoo_intraday as FI  # noqa: E402

T = pd.Timestamp


def _raises(fn, *a, match=""):
    try:
        fn(*a)
    except D.StepError as e:
        assert match in str(e), f"ข้อความ error ไม่ตรง: {e}"
        return
    raise AssertionError("ต้อง raise StepError")


def test_real_holiday_file_parses():
    hol, through = D.read_holidays()
    assert T("2026-08-12") in hol and through >= T("2026-09-25")


def test_next_trading_day_skips_weekend_and_holiday():
    hol = {T("2026-04-13"), T("2026-04-14"), T("2026-04-15")}
    assert D.next_trading_day("2026-04-10", hol, T("2026-12-31")) == T("2026-04-16")
    assert D.next_trading_day("2026-09-25", set(), T("2026-12-31")) == T("2026-09-28")


def test_next_trading_day_beyond_confirmed_stops():
    _raises(D.next_trading_day, "2026-09-25", set(), T("2026-09-25"), match="confirmed_through")


def test_holiday_file_requires_confirmed_through():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "h.txt"
        p.write_text("2026-01-01  # x\n", encoding="utf-8")
        _raises(D.read_holidays, p, match="confirmed_through")


def test_check_env_detects_mismatch_and_missing():
    def fake(name):
        if name == "xgboost":
            raise metadata.PackageNotFoundError(name)
        return {"numpy": "2.1.1", "pandas": "2.2.0"}.get(name, "0")
    probs = E.check({"numpy": "2.1.1", "pandas": "2.3.3", "xgboost": "3.4.1"}, version=fake)
    assert len(probs) == 2 and any("pandas" in p for p in probs) and any("xgboost" in p for p in probs)


def test_pins_file_is_exact():
    assert set(E.read_pins()) == {"numpy", "pandas", "scikit-learn", "xgboost"}


def test_daily_1600_trading_day():
    assert D16.is_trading_day(T("2026-09-26"))[0] is False          # เสาร์
    assert D16.is_trading_day(T("2026-08-12"))[0] is False          # วันหยุดในไฟล์
    ok, warn = D16.is_trading_day(T("2026-09-25"))
    assert ok and warn is None
    ok, warn = D16.is_trading_day(T("2030-01-07"))
    assert ok and "confirmed_through" in warn


def test_intraday_format_matches_locked_file():
    """แปลงจากรูปแบบ yfinance (UTC index, คอลัมน์สลับ) -> ต้องได้ header/บรรทัดเดียวกับไฟล์ล็อก"""
    idx = pd.DatetimeIndex(["2026-09-25 08:00", "2026-09-25 09:00"], tz="UTC")
    raw = pd.DataFrame({"Open": [252.0, 252.0], "High": [253.0, 253.0], "Low": [252.0, 252.0],
                        "Close": [252.0, 253.0], "Adj Close": [252.0, 253.0], "Volume": [680903, 376062]}, index=idx)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "x.csv"
        FI.to_locked_format(raw).to_csv(f, encoding="utf-8")
        got = f.read_text(encoding="utf-8").splitlines()
    locked = (ROOT / "raw_data_intraday" / "KBANK_BK_1h_730d.csv").read_text(encoding="utf-8-sig").splitlines()
    assert got[0] == locked[0]
    assert got[1:] == locked[-2:], (got, locked[-2:])


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
