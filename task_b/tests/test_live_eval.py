"""
tests/test_live_eval.py — record_outcomes.py / report_live.py (ข้อมูลสมมติ ค่าคำนวณมือ)

    python tests/test_live_eval.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import record_outcomes as R    # noqa: E402
import report_live as L        # noqa: E402


def _log():
    base = {"generated_at": "2026-09-27T20:00:00+07:00", "prediction_type": "next_day",
            "ticker": "KBANK.BK", "model": "RF", "config_tag": "x", "code_commit": "c",
            "is_dry_run": False, "prev_close": 100.0, "data_cutoff": "2026-09-25"}
    rows = [
        {**base, "target_date": "2026-09-28", "predicted_return": 0.01, "predicted_close": 101.0},
        {**base, "target_date": "2026-09-29", "data_cutoff": "2026-09-28", "prev_close": 102.0,
         "predicted_return": -0.01, "predicted_close": 100.98},
        {**base, "target_date": "2026-09-30", "data_cutoff": "2026-09-29",   # ยังไม่มีผลจริง
         "prev_close": 102.0, "predicted_return": 0.0, "predicted_close": 102.0},
        {**base, "target_date": "2026-09-28", "is_dry_run": True,            # dry-run
         "predicted_return": 0.5, "predicted_close": 150.0},
        {**base, "target_date": "2026-09-28", "prediction_type": "same_day_1600",
         "predicted_return": 0.0, "predicted_close": 100.0},
    ]
    return pd.DataFrame(rows)


def _closes():
    idx = pd.to_datetime(["2026-09-25", "2026-09-28", "2026-09-29"])
    return {"KBANK.BK": pd.Series([100.0, 102.0, 102.0], index=idx)}


def _outcomes(include_dry_run=False):
    return R.compute_outcomes(_log(), _closes(), {"KBANK.BK": "src"}, {"KBANK.BK": "sha"},
                              "now", include_dry_run)


def test_outcome_values():
    o = _outcomes()
    assert len(o) == 2                                   # ไม่มี dry-run, same_day_1600, วันที่ยังไม่มีผล
    d1 = o[o["target_date"] == "2026-09-28"].iloc[0]
    # actual = 102/100 − 1 = 0.02 · error = 0.01 − 0.02 = −0.01 · naive = 0.02 -> win
    assert np.isclose(d1["actual_return"], 0.02) and np.isclose(d1["error_return"], -0.01)
    assert np.isclose(d1["abs_error_baht"], 1.0) and d1["beat_naive"] == "win"
    assert d1["prev_close_match"]
    d2 = o[o["target_date"] == "2026-09-29"].iloc[0]
    # actual = 102/102 − 1 = 0 · error = −0.01 · naive = 0 -> lose
    assert np.isclose(d2["actual_return"], 0.0) and d2["beat_naive"] == "lose"
    assert d2["prev_close_match"]                        # close 09-28 = 102 = prev_close


def test_dry_run_and_revision_flag():
    assert len(_outcomes(include_dry_run=True)) == 3
    log = _log()
    log.loc[0, "prev_close"] = 99.0                      # ไม่ตรง raw_data -> ติดธง
    o = R.compute_outcomes(log, _closes(), {}, {}, "now")
    assert not o[o["target_date"] == "2026-09-28"].iloc[0]["prev_close_match"]


def test_append_only_no_duplicates():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "outcomes.csv"
        assert R.append_new(_outcomes(), p) == 2
        assert R.append_new(_outcomes(), p) == 0         # รันซ้ำ -> ไม่เพิ่ม
        assert len(pd.read_csv(p)) == 2


def test_summary_metrics():
    s = L.summarize(_outcomes(), cost=0.0034).iloc[0]
    assert s["n"] == 2
    # |e| = 0.01, 0.01 -> MAE 0.01 · naive |y| = 0.02, 0 -> 0.01 · relMAE 1.0
    assert np.isclose(s["MAE_return"], 0.01) and np.isclose(s["MAE_naive"], 0.01)
    assert np.isclose(s["relMAE"], 1.0)
    # R2_OOS = 1 − (0.0001+0.0001)/(0.0004+0) = 0.5
    assert np.isclose(s["R2_OOS"], 0.5)
    assert s["wins"] == 1 and s["losses"] == 1 and s["ties"] == 0
    assert np.isclose(s["win_rate_ex_ties"], 0.5)
    assert s["zero_move_days"] == 1 and np.isclose(s["dir_hit_rate_moved_days"], 1.0)
    assert s["long_signal_days"] == 1 and np.isclose(s["long_signal_up_rate"], 1.0)
    assert not s["enough_data"]


def test_coverage_missing_and_late():
    log = _log()
    log = log[~log["is_dry_run"] & (log["prediction_type"] == "next_day")].copy()
    log = log[log["target_date"] != "2026-09-29"]        # ขาดวัน 09-29
    log.loc[log["target_date"] == "2026-09-28", "generated_at"] = "2026-09-28T09:30:00+07:00"
    closes = {"KBANK.BK": pd.Series(1.0, index=pd.bdate_range("2026-09-25", "2026-09-30"))}
    c = L.coverage(log, closes).iloc[0]
    assert c["missing_dates"] == "2026-09-29" and c["late_predictions"] == 1


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
