"""
tests/test_close_r2.py — Phase 1D รอบ 2: predict_close / outcome same_day_close / daily_1600 --round

    python tests/test_close_r2.py

เขียนลง temp dir เท่านั้น · ไม่สร้าง official prediction · ไม่แตะ log จริง · ไม่ใช้อินเทอร์เน็ต
(test_dryrun_historical_end_to_end fit โมเดลจริงบนไฟล์ล็อก ราว 1–2 นาที)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import daily_1600 as DD        # noqa: E402
import predict_close as PC     # noqa: E402
import record_outcomes as R    # noqa: E402
import report_live as RL       # noqa: E402
import tune_1600 as T          # noqa: E402


def _raises(exc, fn, *a, match="", **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        assert match in str(e), f"ข้อความไม่ตรง: {e}"
        return
    raise AssertionError(f"ต้อง raise {exc.__name__}")


def _rows():
    return [{"generated_at": "2026-10-19T16:03:00+07:00", "prediction_type": "same_day_close",
             "ticker": "KBANK.BK", "target_date": "2026-10-19", "model": "XGBoost", "config_id": 1,
             "params": "{}", "predicted_return": 0.001, "close_bar15": 250.0,
             "predicted_close": 250.25, "last_feature_bar_start": "2026-10-19T15:00:00+07:00",
             "feature_window_end": "2026-10-19T16:00:00+07:00",
             "target_time": "2026-10-19T16:35:00+07:00",
             "snapshot_downloaded_at": "2026-10-19T16:01:00+07:00", "snapshot_path": "s.csv",
             "snapshot_source": "yahoo", "locked_mismatch_bars": 0, "train_days": 700,
             "n_features": 21, "dataset_sha256": "a" * 64, "code_commit": "c" * 40,
             "daily_close_path": "d.csv", "daily_close_source": "yahoo",
             "daily_close_sha256": "b" * 64, "daily_close_last_date": "2026-10-15",
             "daily_close_mismatch_rows": 0, "daily_close_dropped_rows": 1}]


def test_01_constants_and_target_time():
    assert PC.PREDICTION_TYPE == "same_day_close" and PC.CONFIG_TAG_CLOSE == "phase1d-r2-close"
    assert PC.SCORES_PATH.parent.name == "phase1d_r2" and PC.DRYRUN_PATH.name == "prediction_close_dryrun.csv"
    assert PC.SIDECAR_PATH.name == "prediction_log_close_meta.csv"       # ไม่ต่อท้ายไฟล์ sidecar รอบ 1
    assert PC.target_time("2026-10-19") == pd.Timestamp("2026-10-19T16:35:00+07:00")


def test_02_check_clean_covers_round2_files():
    PC.check_clean(lambda *a: "")
    calls = []
    PC.check_clean(lambda *a: calls.append(a) or "")
    flat = [x for c in calls for x in c]
    for must in ("PHASE1D_PLAN_R2.md", "results/phase1d_r2", "raw_data_daily", "results/prediction_log_close_meta.csv"):
        assert must in flat, must
    _raises(PC.GuardError, PC.check_clean, lambda *a: " M daily_close.py" if "*.py" in a else "", match="ยังไม่ commit")
    _raises(PC.GuardError, PC.check_clean,
            lambda *a: " M results/prediction_log_close_meta.csv" if "results/prediction_log_close_meta.csv" in a else "",
            match="log รอบก่อน")


def test_03_official_write_schema_and_duplicate():
    import predict_live
    assert PC.LOG_COLUMNS == predict_live.LOG_COLUMNS                  # schema production เดิม
    with tempfile.TemporaryDirectory() as d:
        log, side = Path(d) / "log.csv", Path(d) / "side.csv"
        PC.write_official(_rows(), log, side)
        got = pd.read_csv(log).iloc[0]
        assert got["prediction_type"] == "same_day_close" and got["config_tag"] == "phase1d-r2-close"
        assert got["prev_close"] == 250.0 and got["predicted_close"] == 250.25
        assert got["data_cutoff"] == "2026-10-19T16:00:00+07:00" and not bool(got["is_dry_run"])
        s = pd.read_csv(side)
        assert s.columns.tolist() == PC.SIDECAR_COLUMNS
        assert s.iloc[0]["daily_close_sha256"] == "b" * 64 and s.iloc[0]["target_time"].endswith("16:35:00+07:00")
        _raises(PC.GuardError, PC.check_no_duplicate, _rows(), log, match="ซ้ำ")
        old_type = [{**_rows()[0], "prediction_type": "same_day_1600"}]    # แถวรอบ 1 key ต่างกัน ไม่บล็อกรอบ 2
        PC.check_no_duplicate([{**old_type[0], "prediction_type": "same_day_1600"}], log)


def test_04_official_guards_window_future_and_daily_snapshot():
    _raises(PC.GuardError, PC.main, ["--target-date", "2026-09-25", "--official", "--latest-snapshot"],
            now="2026-09-25T16:45:00+07:00", match="16:00–16:30")
    _raises(ValueError, PC.main, ["--target-date", "2026-10-01", "--dry-run"],
            now="2026-09-28T16:05:00+07:00", match="อนาคต")
    orig = PC.check_clean
    PC.check_clean = lambda *a, **k: None                                # ข้าม git เพื่อทดสอบ guard ถัดไป
    try:
        _raises(PC.GuardError, PC.main, ["--target-date", "2026-09-25", "--official", "--snapshot", "KBANK.BK=x"],
                now="2026-09-25T16:10:00+07:00", match="ราคาปิดรายวัน")
        _raises(PC.GuardError, PC.main, ["--target-date", "2026-09-25", "--official"],
                now="2026-09-25T16:10:00+07:00", match="live snapshot")
    finally:
        PC.check_clean = orig


def test_05_outcome_same_day_close():
    base = {"generated_at": "2026-10-19T16:03:00+07:00", "prediction_type": "same_day_close",
            "ticker": "KBANK.BK", "target_date": "2026-10-19", "data_cutoff": "2026-10-19T16:00:00+07:00",
            "model": "XGBoost", "config_tag": "phase1d-r2-close", "code_commit": "c", "is_dry_run": False,
            "prev_close": 250.0, "predicted_return": 0.001, "predicted_close": 250.25}
    log = pd.DataFrame([base])
    closes = {"KBANK.BK": pd.Series([251.0], index=pd.to_datetime(["2026-10-19"]))}
    look = lambda tk, t: (999.0, 250.0, "snap.csv", "s" * 64)             # noqa: E731  (C16 ไม่ถูกใช้)
    o = R.compute_outcomes(log, closes, {"KBANK.BK": "raw_data/KBANK"}, {"KBANK.BK": "sha"}, "now", intraday=look)
    r = o.iloc[0]
    assert np.isclose(r["actual_close"], 251.0) and np.isclose(r["actual_return"], 0.004)   # ราคาปิดทางการ/C15 − 1
    assert np.isclose(r["error_return"], 0.001 - 0.004) and r["beat_naive"] == "win"
    assert r["prev_close_match"] and r["outcome_source"] == "raw_data/KBANK" and r["outcome_sha256"] == "sha"
    # C15 สุดท้ายต่างจากที่ log ไว้ (Yahoo ดีเลย์ตอน 16:02) -> บันทึกแต่ติดธง
    o2 = R.compute_outcomes(log, closes, {}, {}, "now", intraday=lambda *a: (999.0, 249.5, "s", "x"))
    assert not o2.iloc[0]["prev_close_match"] and np.isclose(o2.iloc[0]["actual_return"], 251.0 / 250.0 - 1)
    # ยังไม่มีราคาปิด / ยังไม่มี snapshot / ไม่มี lookup -> ข้าม (ยังรอผลจริง)
    assert R.compute_outcomes(log, {"KBANK.BK": closes["KBANK.BK"].iloc[:0]}, {}, {}, "now", intraday=look).empty
    assert R.compute_outcomes(log, closes, {}, {}, "now", intraday=lambda *a: None).empty
    assert R.compute_outcomes(log, closes, {}, {}, "now").empty
    assert "same_day_close" in R.SUPPORTED_TYPES and "same_day_1600" in R.SUPPORTED_TYPES


def test_06_normalize_dryrun_log_round2():
    row = {k: v for k, v in _rows()[0].items()}
    row["is_dry_run"] = True
    n = R.normalize_log(pd.DataFrame([row]))
    assert n.iloc[0]["config_tag"] == "phase1d-r2-close" and n.iloc[0]["prev_close"] == 250.0
    assert n.iloc[0]["predicted_close"] == 250.25 and n.iloc[0]["data_cutoff"] == "2026-10-19T16:00:00+07:00"


def test_07_report_window_same_day_close():
    base = {"prediction_type": "same_day_close", "ticker": "KBANK.BK", "model": "XGBoost", "is_dry_run": False}
    log = pd.DataFrame([
        {**base, "target_date": "2026-10-19", "generated_at": "2026-10-19T15:59:00+07:00"},
        {**base, "target_date": "2026-10-20", "generated_at": "2026-10-20T16:31:00+07:00"},
        {**base, "target_date": "2026-10-21", "generated_at": "2026-10-21T16:05:00+07:00"}])
    c = RL.coverage(log, {}).iloc[0]
    assert c["early_predictions"] == 1 and c["late_predictions"] == 1


def _run_daily(argv):
    calls = []
    orig_run, orig_trading = DD.run, DD.is_trading_day
    DD.run = lambda step, args, ok_codes=(0,): calls.append((step, [Path(str(a)).name for a in args])) or 0
    DD.is_trading_day = lambda *a, **k: (True, None)
    try:
        assert DD.main(argv) == 0
    finally:
        DD.run, DD.is_trading_day = orig_run, orig_trading
    return calls


def test_08_daily_1600_round_selects_predictor():
    c2 = _run_daily(["--phase", "predict"])                               # ค่าเริ่มต้น = r2
    steps = [s for s, _ in c2]
    assert steps == ["1 fetch", "1b fetch daily close", "2 predict", "3 availability"], steps
    predict_args = dict(c2)["2 predict"]
    assert "predict_close.py" in predict_args and "--dry-run" in predict_args
    c1 = _run_daily(["--phase", "predict", "--round", "r1"])
    assert [s for s, _ in c1] == ["1 fetch", "2 predict", "3 availability"]
    assert "predict_1600.py" in dict(c1)["2 predict"]
    co = _run_daily(["--phase", "outcome"])                               # outcome ไม่ดึงราคาปิดรายวัน
    assert "1b fetch daily close" not in [s for s, _ in co]


def test_09_dryrun_historical_end_to_end():
    """replay target ที่อยู่ในไฟล์ล็อก: X_live == X(t) ของ historical builder · ราคาปิดวัน t ไม่ถูกใช้ทำนาย"""
    with tempfile.TemporaryDirectory() as d:
        scores = Path(d) / "scores.csv"
        pd.DataFrame([{"model": fam, "config_id": 0, "config": str(T.GRIDS[fam][0]), "winner": True}
                      for fam in T.GRIDS]).to_csv(scores, index=False)
        orig = (PC.SCORES_PATH, PC.DRYRUN_PATH)
        PC.SCORES_PATH, PC.DRYRUN_PATH = scores, Path(d) / "dry.csv"
        try:
            rows = PC.main(["--target-date", "2026-09-24", "--dry-run"], now="2026-09-25T10:00:00+07:00")
            out = pd.read_csv(PC.DRYRUN_PATH)
        finally:
            PC.SCORES_PATH, PC.DRYRUN_PATH = orig
    assert len(rows) == 6 and len(out) == 6 and list(out.columns) == PC.COLUMNS
    assert set(out["prediction_type"]) == {"same_day_close"} and out["is_dry_run"].all()
    assert out["historical_x_match"].all() and set(out["daily_close_source"]) == {"locked"}
    assert out["daily_close_last_date"].eq("2026-09-23").all()             # ราคาปิดล่าสุดที่ใช้ = วันก่อน target
    assert np.allclose(out["predicted_close"], out["close_bar15"] * (1 + out["predicted_return"]))
    assert out["target_time"].str.endswith("16:35:00+07:00").all()


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
