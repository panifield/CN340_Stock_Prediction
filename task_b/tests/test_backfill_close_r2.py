"""
tests/test_backfill_close_r2.py — tools/backfill_close_r2.py (ฟังก์ชันล้วน · ข้อมูลสังเคราะห์ · ไม่เขียนไฟล์ลง repo)

    python tests/test_backfill_close_r2.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import backfill_close_r2 as B      # noqa: E402
import intraday_1600 as I          # noqa: E402
import tune_1600 as T              # noqa: E402

BKK = I.BANGKOK


def _meta(when):
    return {"downloaded_at": pd.Timestamp(when), "sha256": "x", "path": Path("s.csv"), "source": "yahoo"}


def test_01_backfill_days_use_holiday_calendar():
    days = [str(d.date()) for d in B.backfill_days()]
    assert days[0] == "2026-09-28" and days[-1] == "2026-10-15" and len(days) == 13
    assert "2026-10-13" not in days                      # วันหยุดนวมินทรมหาราช
    assert all(pd.Timestamp(d).dayofweek < 5 for d in days)


def test_02_pick_bar_source_rules():
    s = lambda when: (None, _meta(when))                  # noqa: E731
    day = "2026-10-05"
    assert B.pick_bar_source([], day) == (None, "yahoo_final")
    one = s("2026-10-05T16:02:23+07:00")
    assert B.pick_bar_source([one], day) == (one, "snapshot_1602")
    late_ok = s("2026-10-05T16:22:00+07:00")
    assert B.pick_bar_source([one, late_ok], day)[1:] == ("snapshot_before_1630",)       # ล่าสุดก่อน 16:30
    assert B.pick_bar_source([one, late_ok], day)[0] is late_ok
    after = s("2026-10-05T18:27:00+07:00")
    assert B.pick_bar_source([after], day) == (after, "snapshot_after_close")
    other_day = s("2026-10-06T16:02:00+07:00")            # snapshot ของวันอื่นไม่นับ
    assert B.pick_bar_source([other_day], day) == (None, "yahoo_final")
    both = [one, after]
    assert B.pick_bar_source(both, day)[1] == "snapshot_1602"


def test_03_outcome_fields():
    row = {"target_date": "2026-10-05", "close_bar15": 100.0, "predicted_return": 0.001}
    closes = {pd.Timestamp("2026-10-05"): 101.0}
    c15 = pd.Series([100.0], index=pd.to_datetime(["2026-10-05"]))
    o = B.outcome_fields(row, closes, c15)
    assert np.isclose(o["actual_return"], 0.01) and np.isclose(o["error_return"], 0.001 - 0.01)
    assert o["beat_naive"] == "win" and o["prev_close_match"] is True and o["c15_final"] == 100.0
    o2 = B.outcome_fields({**row, "close_bar15": 99.5}, closes, c15)          # C15 ตอนทำนายต่างจากค่าสุดท้าย
    assert o2["prev_close_match"] is False and np.isclose(o2["actual_return"], 101.0 / 99.5 - 1)
    o3 = B.outcome_fields(row, {}, pd.Series(dtype=float))                    # ยังไม่มีราคาปิด / c15
    assert np.isnan(o3["actual_close"]) and o3["beat_naive"] == "" and np.isnan(o3["prev_close_match"])


def _bars(n_days=60, seed=2):
    return I.prepare_bars(T.synthetic_bars(n_days=n_days, seed=seed))


def test_04_poison_future_touches_only_forbidden_cells():
    bars = _bars()
    daily = T.synthetic_daily(bars)
    day = sorted(bars["date"].unique())[40]
    rng = np.random.default_rng(1)
    pb, pdly, n = B.poison_future(bars, daily, day, rng)
    forbidden = ((bars["date"] > day) | ((bars["date"] == day) & (bars["hour"] >= 16))).to_numpy()
    assert n == int(forbidden.sum()) + int((daily.index >= day).sum())
    cols = ["Open", "High", "Low", "Close", "Volume"]
    pd.testing.assert_frame_equal(pb.loc[~forbidden, cols], bars.loc[~forbidden, cols],
                                  check_dtype=False)                           # ส่วนที่ควรเห็นไม่ถูกแตะ
    assert not np.allclose(pb.loc[forbidden, "Close"], bars.loc[forbidden, "Close"])
    assert pdly[daily.index < day].equals(daily[daily.index < day])
    assert not np.allclose(pdly[daily.index >= day], daily[daily.index >= day])


def test_05_poison_control_touches_visible_cells():
    bars = _bars()
    daily = T.synthetic_daily(bars)
    day = sorted(bars["date"].unique())[40]
    pb, pdly, n = B.poison_control(bars, daily, day, np.random.default_rng(1))
    m = ((bars["date"] == day) & (bars["hour"] == 15)).to_numpy()
    assert np.allclose(pb.loc[m, "Close"], bars.loc[m, "Close"] * 1.05) and n == int(m.sum()) + 1
    prev = daily.index[daily.index < day].max()
    assert np.isclose(pdly.loc[prev], daily.loc[prev] * 1.10)
    assert pdly[daily.index >= day].equals(daily[daily.index >= day])


def test_06_poison_future_leaves_prediction_inputs_unchanged():
    """ยืนยันกลไกเดียวกับ leak test จริง: ข้อมูลที่วางยา (หลังเวลาตัด) ต้องไม่เปลี่ยน X_live / ชุดเทรน"""
    bars = _bars()
    daily = T.synthetic_daily(bars)
    day = sorted(bars["date"].unique())[40]
    base = I.build_1600_live_feature(bars, day, daily)
    pb, pdly, _ = B.poison_future(bars, daily, day, np.random.default_rng(7))
    out = I.build_1600_live_feature(pb, day, pdly)
    pd.testing.assert_frame_equal(base["X_live"], out["X_live"])
    pd.testing.assert_frame_equal(base["X_train"], out["X_train"])
    pd.testing.assert_series_equal(base["y_train"], out["y_train"])
    cb, cd, _ = B.poison_control(bars, daily, day, np.random.default_rng(7))
    ctl = I.build_1600_live_feature(cb, day, cd)
    assert not np.allclose(base["X_live"].to_numpy(), ctl["X_live"].to_numpy())


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
