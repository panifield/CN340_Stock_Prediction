"""
tests/test_live_1600.py — live snapshot / official guards ของ predict_1600 / outcome ของ same_day_1600

    python tests/test_live_1600.py

เขียนลง temp dir เท่านั้น · ไม่สร้าง official prediction · ไม่แตะ log จริง
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import intraday_1600 as I      # noqa: E402
import live_1600 as LV         # noqa: E402
import predict_1600 as P       # noqa: E402
import record_outcomes as R    # noqa: E402
import report_live as RL       # noqa: E402

LOCKED = ROOT / "raw_data_intraday" / "KBANK_BK_1h_730d.csv"


def _raises(exc, fn, *a, match="", **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        assert match in str(e), f"ข้อความไม่ตรง: {e}"
        return
    raise AssertionError(f"ต้อง raise {exc.__name__}")


def _save(d, when, source="yahoo", src=LOCKED):
    return LV.save_snapshot(src, "KBANK.BK", source, when, root=d)


def test_snapshot_save_load_and_tamper():
    with tempfile.TemporaryDirectory() as d:
        p = _save(d, "2026-09-25T16:05:00+07:00")
        assert p.read_bytes() == LOCKED.read_bytes()
        meta = json.loads(Path(str(p) + ".meta.json").read_text(encoding="utf-8"))
        assert meta["sha256"] == I.EXPECTED_SHA256["KBANK.BK"] and meta["source"] == "yahoo"
        bars, m = LV.load_snapshot(p)
        assert m["downloaded_at"] == pd.Timestamp("2026-09-25T16:05:00+07:00")
        _raises(LV.SnapshotError, _save, d, "2026-09-25T16:05:00+07:00", match="เขียนทับ")
        p.write_bytes(p.read_bytes() + b"x")
        _raises(LV.SnapshotError, LV.load_snapshot, p, match="SHA")


def test_validate_for_prediction():
    t = pd.Timestamp("2026-09-25")
    ok = {"source": "yahoo", "downloaded_at": pd.Timestamp("2026-09-25T16:00:00+07:00")}
    LV.validate_for_prediction(ok, t)
    _raises(LV.SnapshotError, LV.validate_for_prediction, {**ok, "source": "settrade"}, t,
            match="รอบสอง")
    _raises(LV.SnapshotError, LV.validate_for_prediction,
            {**ok, "downloaded_at": pd.Timestamp("2026-09-25T15:59:59+07:00")}, t, match="ก่อน 16:00")
    _raises(LV.SnapshotError, LV.validate_for_prediction,
            {**ok, "downloaded_at": pd.Timestamp("2026-09-26T16:10:00+07:00")}, t, match="คนละวัน")


def test_merge_prefers_locked_and_reports_mismatch():
    locked = I.load_ticker_bars("KBANK.BK")
    tail = locked[locked["date"] >= "2026-09-24"].copy()
    tail.loc[tail.index[0], "Close"] += 1.0                         # แท่งที่ทับแต่ค่าไม่ตรง
    ts = pd.Timestamp("2026-09-28 10:00", tz=I.BANGKOK)
    new = tail.iloc[[0]].copy()
    new["Datetime"] = ts.isoformat(sep=" ")
    snap = I.prepare_bars(pd.concat([tail, new])[["Datetime", "Adj Close", "Close", "High", "Low",
                                                  "Open", "Volume"]])
    bars, mism = LV.merge_with_locked(locked, snap)
    assert len(mism) == 1
    assert len(bars) == len(locked) + 1                            # เพิ่มเฉพาะแท่งใหม่
    first = tail.iloc[0]
    got = bars[(bars["date"] == first["date"]) & (bars["hour"] == first["hour"])]["Close"].iloc[0]
    assert got == locked.loc[tail.index[0], "Close"]               # ใช้ค่าจากไฟล์ล็อก


def test_snapshot_prediction_equals_historical():
    """snapshot = สำเนาไฟล์ล็อก -> bars และ X_live เท่ากับ historical เป๊ะ"""
    t = pd.Timestamp("2026-09-25")
    with tempfile.TemporaryDirectory() as d:
        snap = LV.load_snapshot(_save(d, "2026-09-25T16:05:00+07:00"))
        bars, info = LV.bars_for_prediction("KBANK.BK", t, snap)
        assert info["mismatches"] == 0 and info["source"] == "yahoo"
        live = I.build_1600_live_feature(bars, t)
        assert P.check_against_historical(I.load_ticker_bars("KBANK.BK"), t, live) is True
        assert LV.latest_snapshot_for("KBANK.BK", t, d)[1]["path"] == snap[1]["path"]
        _raises(LV.SnapshotError, LV.latest_snapshot_for, "KBANK.BK", pd.Timestamp("2026-09-24"), d)


def test_outcome_close_bar16_rules():
    t = pd.Timestamp("2026-09-25")
    locked = I.load_ticker_bars("KBANK.BK")
    day = locked[locked["date"] == t].set_index("hour")
    with tempfile.TemporaryDirectory() as d:
        early = LV.load_snapshot(_save(d, "2026-09-25T16:20:00+07:00"))
        assert LV.outcome_close_bar16("KBANK.BK", t, [early]) is None      # ก่อน 17:00 ใช้ไม่ได้
        late = LV.load_snapshot(_save(d, "2026-09-25T17:10:00+07:00"))
        c16, c15, src, sha = LV.outcome_close_bar16("KBANK.BK", t, [early, late])
        assert (c16, c15) == (day.loc[16, "Close"], day.loc[15, "Close"]) and "171000" in src
        c16b, _, src_b, _ = LV.outcome_close_bar16("KBANK.BK", t, [], locked)   # ไฟล์ล็อก
        assert c16b == c16 and "raw_data_intraday" in src_b


def test_time_window():
    t = pd.Timestamp("2026-09-29")
    P.check_time_window(t, pd.Timestamp("2026-09-29T16:00:00+07:00"))
    P.check_time_window(t, pd.Timestamp("2026-09-29T16:29:59+07:00"))
    for bad in ("2026-09-29T15:59:59+07:00", "2026-09-29T16:30:00+07:00", "2026-09-28T16:10:00+07:00"):
        _raises(P.GuardError, P.check_time_window, t, pd.Timestamp(bad))


def test_approval_file_rules():
    t = pd.Timestamp("2026-10-05")
    ok_git = lambda *a: ""                                          # noqa: E731
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "PHASE1D_LIVE_APPROVAL.md"
        _raises(P.GuardError, P.check_approval, t, f, ok_git, match="ไม่พบ")
        full = ("APPROVED_BY: x\nAVAILABILITY_EVIDENCE: results/dryrun/availability_1600.csv\n"
                "FIRST_OFFICIAL_DATE: 2026-10-05\nFREEZE_COMMIT: abc\n")
        f.write_text(full.replace("APPROVED_BY: x\n", ""), encoding="utf-8")
        _raises(P.GuardError, P.check_approval, t, f, ok_git, match="APPROVED_BY")
        f.write_text(full.replace("FREEZE_COMMIT: abc", "FREEZE_COMMIT:"), encoding="utf-8")
        _raises(P.GuardError, P.check_approval, t, f, ok_git, match="ว่าง")
        f.write_text(full, encoding="utf-8")
        _raises(P.GuardError, P.check_approval, pd.Timestamp("2026-10-02"), f, ok_git, match="ก่อน")
        assert P.check_approval(t, f, ok_git)["FIRST_OFFICIAL_DATE:"] == "2026-10-05"


def _rows():
    return [{"generated_at": "2026-10-05T16:03:00+07:00", "prediction_type": "same_day_1600",
             "ticker": "KBANK.BK", "target_date": "2026-10-05", "model": "XGBoost", "config_id": 1,
             "params": "{}", "predicted_return": 0.001, "close_bar15": 250.0,
             "predicted_close_bar16": 250.25, "last_feature_bar_start": "2026-10-05T15:00:00+07:00",
             "feature_window_end": "2026-10-05T16:00:00+07:00",
             "target_bar_start": "2026-10-05T16:00:00+07:00",
             "snapshot_downloaded_at": "2026-10-05T16:01:00+07:00", "snapshot_path": "s.csv",
             "snapshot_source": "yahoo", "locked_mismatch_bars": 0, "train_days": 700,
             "n_features": 21, "dataset_sha256": "a" * 64, "code_commit": "c" * 40}]


def test_official_write_schema_and_duplicate():
    import predict_live
    assert P.LOG_COLUMNS == predict_live.LOG_COLUMNS            # ใช้ schema production เดิม
    with tempfile.TemporaryDirectory() as d:
        log, side = Path(d) / "log.csv", Path(d) / "side.csv"
        P.write_official(_rows(), log, side)
        got = pd.read_csv(log).iloc[0]
        assert got["prev_close"] == 250.0 and got["predicted_close"] == 250.25
        assert got["data_cutoff"] == "2026-10-05T16:00:00+07:00"
        assert got["config_tag"] == P.CONFIG_TAG_1600 and not bool(got["is_dry_run"])
        assert pd.read_csv(side).columns.tolist() == P.SIDECAR_COLUMNS
        _raises(P.GuardError, P.check_no_duplicate, _rows(), log, match="ซ้ำ")


def test_official_refused_without_approval_and_future_date():
    _raises(P.GuardError, P.main, ["--target-date", "2026-09-25", "--official", "--latest-snapshot"],
            now="2026-09-25T16:05:00+07:00", match="ไม่พบ")
    _raises(ValueError, P.main, ["--target-date", "2026-10-01", "--dry-run"],
            now="2026-09-28T16:05:00+07:00", match="อนาคต")


def test_outcomes_same_day_1600():
    log = R.normalize_log(pd.DataFrame([{**_rows()[0], "is_dry_run": False}]))
    look = lambda tk, t: (251.0, 250.0, "snap.csv", "s" * 64)       # noqa: E731
    o = R.compute_outcomes(log, {}, {}, {}, "now", intraday=look)
    r = o.iloc[0]
    assert np.isclose(r["actual_return"], 0.004) and np.isclose(r["error_return"], 0.001 - 0.004)
    assert r["beat_naive"] == "win" and r["prev_close_match"] and r["outcome_source"] == "snap.csv"
    assert R.compute_outcomes(log, {}, {}, {}, "now").empty                 # ไม่มี lookup -> ข้าม
    assert R.compute_outcomes(log, {}, {}, {}, "now", intraday=lambda *a: None).empty


def test_report_window_1600():
    base = {"prediction_type": "same_day_1600", "ticker": "KBANK.BK", "model": "XGBoost",
            "is_dry_run": False}
    log = pd.DataFrame([
        {**base, "target_date": "2026-10-05", "generated_at": "2026-10-05T15:59:00+07:00"},
        {**base, "target_date": "2026-10-06", "generated_at": "2026-10-06T16:31:00+07:00"},
        {**base, "target_date": "2026-10-07", "generated_at": "2026-10-07T16:05:00+07:00"}])
    c = RL.coverage(log, {}).iloc[0]
    assert c["early_predictions"] == 1 and c["late_predictions"] == 1


def test_live_modules_do_not_import_data_loader():
    code = "import sys, live_1600, predict_1600; print('data_loader' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "False", r.stderr


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
