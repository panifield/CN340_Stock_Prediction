"""
tests/test_phase1d_r2.py — Phase 1D รอบ 2 (target = ราคาปิดทางการ) · PHASE1D_PLAN_R2.md

    python tests/test_phase1d_r2.py

ใช้ข้อมูลสังเคราะห์เท่านั้น (ยกเว้นตรวจไฟล์ล็อกจริงว่า SHA/รูปแบบถูก) -- ไม่ fit โมเดลบน target จริงของรอบ 2
test_phase1d.py เดิม (รอบ 1) ต้องผ่านเหมือนเดิม -- ไฟล์นี้ไม่แก้ไฟล์นั้น
(test_smoke_runs รัน tune_1600.py --smoke --round r2 ราว 3 นาที)
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import daily_close as D        # noqa: E402
import intraday_1600 as I      # noqa: E402
import tune_1600 as T          # noqa: E402

BKK = I.BANGKOK


def _bars(n_days=80, seed=1):
    return I.prepare_bars(T.synthetic_bars(n_days=n_days, seed=seed))


def _daily(bars, seed=1):
    return T.synthetic_daily(bars, seed=seed)


def _day(bars, k=40):
    return sorted(bars["date"].unique())[k]


def _mod(bars, date, hour, **vals):
    b = bars.copy()
    m = (b["date"] == date) & (b["hour"] == hour)
    assert m.sum() == 1
    for c, v in vals.items():
        b.loc[m, c] = v
    return b


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError(f"ควร raise {exc.__name__}")


# ---------------------------------------------------------------
# builder (รอบ 2)
# ---------------------------------------------------------------
def test_01_target_is_official_close_over_c15():
    bars, daily = _bars(), _daily(_bars())
    X, y, info, _ = I.build_dataset(bars, daily)
    t = X.index[10]
    c = bars[bars["date"] == t].set_index("hour")["Close"]
    assert np.isclose(y.loc[t], daily.loc[t] / c[15] - 1)
    assert np.isclose(info.loc[t, "C15"], c[15])
    assert X.shape[1] == 21 and list(X.columns) == I.FEATURE_COLS


def test_02_round2_never_reads_bar16():
    bars, daily = _bars(), _daily(_bars())
    X0, y0, _, _ = I.build_dataset(bars, daily)
    no16 = bars[bars["hour"] != 16].reset_index(drop=True)          # ไม่มีแท่ง 16 เลย
    X1, y1, _, rep = I.build_dataset(no16, daily)
    pd.testing.assert_frame_equal(X0, X1)
    pd.testing.assert_series_equal(y0, y1)
    t = X0.index[10]
    wild = bars.copy()
    wild.loc[wild["hour"] == 16, ["Open", "High", "Low", "Close"]] *= 3.0   # แท่ง 16 เพี้ยนทั้งหมด
    X2, y2, _, _ = I.build_dataset(wild, daily)
    pd.testing.assert_frame_equal(X0, X2)
    pd.testing.assert_series_equal(y0, y2)
    assert t in X0.index


def test_03_gap_uses_previous_official_close():
    bars, daily = _bars(), _daily(_bars())
    X, _, _, _ = I.build_dataset(bars, daily)
    t = X.index[10]
    o10 = bars[(bars["date"] == t) & (bars["hour"] == 10)]["Open"].iloc[0]
    prev = daily[daily.index < t].iloc[-1]                               # แถวก่อนหน้าใน daily = วันทำการก่อนหน้า
    assert np.isclose(X.loc[t, "gap"], o10 / prev - 1)
    X1, _, _, _ = I.build_dataset(bars)                                  # รอบ 1 ใช้ C16 ของ t_prev -> ค่าต่างกัน
    common = X.index.intersection(X1.index)
    assert not np.allclose(X1.loc[common, "gap"], X.loc[common, "gap"])


def test_04_missing_daily_close_drops_the_day():
    bars, daily = _bars(), _daily(_bars())
    X0, _, _, rep0 = I.build_dataset(bars, daily)
    t = X0.index[10]
    X1, _, _, rep1 = I.build_dataset(bars, daily.drop(t))
    assert t not in X1.index
    assert rep1["dropped_days"][str(t.date())] == ["daily_close"]
    assert rep1["n_eligible_labeled"] == rep0["n_eligible_labeled"] - 1


def test_05_round1_path_unchanged_without_daily():
    bars = _bars()
    X, y, _, _ = I.build_dataset(bars)
    t = X.index[10]
    c = bars[bars["date"] == t].set_index("hour")["Close"]
    assert np.isclose(y.loc[t], c[16] / c[15] - 1)                      # target รอบ 1 ยังเป็น C16/C15 − 1


# ---------------------------------------------------------------
# live builder (รอบ 2) -- causal
# ---------------------------------------------------------------
def test_06_live_ignores_target_day_and_future_official_close():
    bars, daily = _bars(), _daily(_bars())
    t = _day(bars)
    base = I.build_1600_live_feature(bars, t, daily)
    wild = daily.copy()
    wild[wild.index >= t] *= 2.0                                         # ราคาปิดวัน t และอนาคต เพี้ยน
    out = I.build_1600_live_feature(bars, t, wild)
    pd.testing.assert_frame_equal(base["X_live"], out["X_live"])
    pd.testing.assert_frame_equal(base["X_train"], out["X_train"])
    pd.testing.assert_series_equal(base["y_train"], out["y_train"])
    assert base["train_days"] == out["train_days"]


def test_07_live_uses_previous_close_for_gap():
    bars, daily = _bars(), _daily(_bars())
    t = _day(bars)
    base = I.build_1600_live_feature(bars, t, daily)
    prev_day = daily[daily.index < t].index[-1]
    changed = daily.copy()
    changed.loc[prev_day] = changed.loc[prev_day] * 1.1
    out = I.build_1600_live_feature(bars, t, changed)
    assert not np.isclose(base["X_live"]["gap"].iloc[0], out["X_live"]["gap"].iloc[0])
    o10 = bars[(bars["date"] == t) & (bars["hour"] == 10)]["Open"].iloc[0]
    assert np.isclose(out["X_live"]["gap"].iloc[0], o10 / changed.loc[prev_day] - 1)
    assert np.isclose(out["daily_prev_close"], changed.loc[prev_day])


def test_08_live_ignores_bar16_and_later_days():
    bars, daily = _bars(), _daily(_bars())
    t = _day(bars)
    base = I.build_1600_live_feature(bars, t, daily)
    wild = bars.copy()
    wild.loc[(wild["date"] > t) | ((wild["date"] == t) & (wild["hour"] >= 16)),
             ["Open", "High", "Low", "Close", "Volume"]] *= 5.0
    out = I.build_1600_live_feature(wild, t, daily)
    pd.testing.assert_frame_equal(base["X_live"], out["X_live"])
    pd.testing.assert_frame_equal(base["X_train"], out["X_train"])


def test_09_live_matches_historical_row():
    bars, daily = _bars(), _daily(_bars())
    X, _, _, _ = I.build_dataset(bars, daily)
    t = X.index[30]
    live = I.build_1600_live_feature(bars, t, daily)
    assert np.allclose(live["X_live"].iloc[0].to_numpy(), X.loc[t].to_numpy(), rtol=0, atol=1e-12)
    assert live["train_days"] == int((X.index < t).sum())


def test_10_live_without_previous_close_raises():
    bars, daily = _bars(), _daily(_bars())
    t = _day(bars)
    _raises(ValueError, I.build_1600_live_feature, bars, t, daily[daily.index >= t])   # ไม่มีราคาปิดก่อน t เลย


# ---------------------------------------------------------------
# daily_close.py
# ---------------------------------------------------------------
def _snap_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(date, close, vol=1000):
    return {"Date": date, "Adj Close": close, "Close": close, "High": close, "Low": close,
            "Open": close, "Volume": vol}


def test_11_locked_files_are_real_and_sha_checked():
    for t in ("KBANK.BK", "ADVANC.BK"):
        d = D.load_locked(t)
        assert d.index.max() == D.LOCKED_END and (d["Volume"] > 0).all() and d["Close"].notna().all()
        _raises(D.DailyCloseError, D.load_locked, t, expected_sha256="0" * 64)


def test_12_merge_prefers_locked_reports_mismatch_drops_ghost():
    locked = pd.DataFrame({"Close": [10.0, 11.0, 12.0], "Volume": [1, 1, 1]},
                          index=pd.to_datetime(["2026-09-23", "2026-09-24", "2026-09-25"]))
    snap = pd.DataFrame({"Close": [11.5, 12.0, 13.0, 99.0, 14.0], "Volume": [5, 5, 5, 0, 5]},
                        index=pd.to_datetime(["2026-09-24", "2026-09-25", "2026-09-28", "2026-09-29", "2026-09-30"]))
    close, mism, dropped = D.merge_daily(locked, snap)
    assert close.loc["2026-09-24"] == 11.0                        # ใช้ค่าไฟล์ล็อก
    assert list(mism.index) == [pd.Timestamp("2026-09-24")]      # รายงานแถวที่ไม่ตรง
    assert dropped == 1 and pd.Timestamp("2026-09-29") not in close.index   # แถวผี Volume=0 ไม่ถูกใช้
    assert close.loc["2026-09-28"] == 13.0 and close.loc["2026-09-30"] == 14.0
    assert list(close.index) == sorted(close.index)


def test_13_snapshot_roundtrip_sha_and_tamper():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src.csv"
        _snap_csv(src, [_row("2026-10-14", 230.0), _row("2026-10-15", 231.0)])
        dst = D.save_snapshot(src, "KBANK.BK", "yahoo", "2026-10-16T16:02:30+07:00", root=d)
        df, meta = D.load_snapshot(dst)
        assert meta["sha256"] and meta["source"] == "yahoo" and df.loc["2026-10-15", "Close"] == 231.0
        dst.write_text(dst.read_text() + "\n2026-10-16,1,1,1,1,1,1\n")
        _raises(D.DailyCloseError, D.load_snapshot, dst)           # แก้หลังบันทึก -> SHA ไม่ตรง
        _raises(D.DailyCloseError, D.save_snapshot, src, "KBANK.BK", "yahoo",
                "2026-10-16T16:02:30+07:00", d)                    # ห้ามเขียนทับ snapshot เดิม


def test_14_latest_snapshot_same_day_only_and_validation():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src.csv"
        _snap_csv(src, [_row("2026-10-15", 231.0)])
        D.save_snapshot(src, "KBANK.BK", "yahoo", "2026-10-15T16:02:00+07:00", root=d)
        D.save_snapshot(src, "KBANK.BK", "yahoo", "2026-10-16T16:02:05+07:00", root=d)
        D.save_snapshot(src, "KBANK.BK", "yahoo", "2026-10-16T16:20:00+07:00", root=d)
        df, meta = D.latest_snapshot_for("KBANK.BK", "2026-10-16", root=d)
        assert meta["downloaded_at"].strftime("%H:%M:%S") == "16:20:00"          # ล่าสุดของวัน target
        _raises(D.DailyCloseError, D.latest_snapshot_for, "KBANK.BK", "2026-10-19", root=d)
        D.validate_for_prediction(meta, "2026-10-16")
        _raises(D.DailyCloseError, D.validate_for_prediction, meta, "2026-10-19")      # คนละวัน
        bad = dict(meta, source="settrade")
        _raises(D.DailyCloseError, D.validate_for_prediction, bad, "2026-10-16")       # แหล่งไม่อนุญาต


def test_15_previous_trading_day_guard_uses_holiday_calendar():
    holidays = {pd.Timestamp("2026-10-13"), pd.Timestamp("2026-10-16")}
    assert D.previous_trading_day("2026-10-19", holidays) == pd.Timestamp("2026-10-15")   # ข้ามเสาร์-อาทิตย์ + หยุด 16
    assert D.previous_trading_day("2026-10-14", holidays) == pd.Timestamp("2026-10-12")   # ข้ามหยุด 13
    close = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-10-14", "2026-10-15"]))
    assert D.require_previous_close(close, "2026-10-19", holidays) == pd.Timestamp("2026-10-15")
    _raises(D.DailyCloseError, D.require_previous_close, close.iloc[:1], "2026-10-19", holidays)


def test_16_daily_close_for_prediction_with_snapshot():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src.csv"
        _snap_csv(src, [_row("2026-09-25", 252.0), _row("2026-10-14", 230.0), _row("2026-10-15", 231.0),
                        _row("2026-10-16", 999.0, vol=0)])            # แถวผี/แถววันนี้ที่ยังไม่สมบูรณ์
        D.save_snapshot(src, "KBANK.BK", "yahoo", "2026-10-16T16:02:00+07:00", root=d)
        snap = D.latest_snapshot_for("KBANK.BK", "2026-10-16", root=d)
        close, info = D.daily_close_for_prediction("KBANK.BK", "2026-10-16", snap, check_previous=False)
        assert close.loc["2026-10-15"] == 231.0 and pd.Timestamp("2026-10-16") not in close.index
        assert info["source"] == "yahoo" and info["dropped_rows"] == 1 and info["last_date"] == "2026-10-15"
        # ไม่มีราคาปิดวันทำการก่อนหน้า (15 ต.ค.) -> หยุด
        src2 = Path(d) / "src2.csv"
        _snap_csv(src2, [_row("2026-10-14", 230.0)])
        D.save_snapshot(src2, "KBANK.BK", "yahoo", "2026-10-16T16:05:00+07:00", root=d)
        snap2 = D.latest_snapshot_for("KBANK.BK", "2026-10-16", root=d)
        _raises(D.DailyCloseError, D.daily_close_for_prediction, "KBANK.BK", "2026-10-16", snap2)


def test_17_modules_do_not_import_data_loader():
    code = "import sys, daily_close, intraday_1600, tune_1600; print('data_loader' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "False", r.stderr[-500:]


def test_18_smoke_runs_round2():
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, "tune_1600.py", "--smoke", "--round", "r2", "--out", d],
                           cwd=ROOT, capture_output=True, text=True,
                           env={**__import__("os").environ, "PYTHONUTF8": "1"})
        assert r.returncode == 0, r.stderr[-2000:]
        for f in ("phase1d_scores.csv", "phase1d_evals.csv", "phase1d_folds.csv",
                  "phase1d_baselines.csv", "phase1d_econ.csv", "phase1d_summary.md"):
            assert (Path(d) / f).exists(), f
        assert len(pd.read_csv(Path(d) / "phase1d_scores.csv")) == 24
        assert "รอบ 2" in (Path(d) / "phase1d_summary.md").read_text(encoding="utf-8")


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
