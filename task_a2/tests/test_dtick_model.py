"""
tests/test_dtick_model.py — tick grid + การแปลง P(คลาส Δtick) เป็น P(คี่) (2026-10-24)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dtick_model as DM  # noqa: E402


def test_tick_index_steps_match_set_tick_table():
    ti = DM.tick_index
    assert ti(162.5) - ti(162.0) == 1          # tick 0.50
    assert ti(201.0) - ti(200.0) == 1          # tick 1.00
    assert ti(87.25) - ti(87.0) == 1           # tick 0.25
    assert ti(402.0) - ti(400.0) == 1          # tick 2.00


def test_tick_index_counts_across_200_boundary():
    ti = DM.tick_index
    assert ti(200.0) - ti(199.5) == 1
    assert ti(201.0) - ti(199.5) == 2
    assert ti(199.0) - ti(201.0) == -3


def test_price_from_index_roundtrip():
    prices = np.array([87.25, 99.75, 100.0, 162.5, 199.5, 200.0, 241.0, 399.0, 400.0, 402.0])
    assert np.allclose(DM.price_from_index(DM.tick_index(prices)), prices)


def test_off_grid_price_raises():
    try:
        DM.tick_index(162.3)
    except ValueError:
        return
    raise AssertionError("ราคานอก grid ต้อง raise")


def test_dtick_class_clips_to_five_classes():
    assert DM.dtick_class([-5, -2, -1, 0, 1, 2, 7]).tolist() == [-2, -2, -1, 0, 1, 2, 2]


def test_p_odd_half_baht_regime():
    # 162.5 (คู่): -1 -> 162.0 คู่ · 0 -> คู่ · +1 -> 163.0 คี่
    i = DM.tick_index([162.5])
    w = {-2: {-2: 1.0}, 2: {2: 1.0}}
    eye = np.eye(5)
    got = [DM.p_odd_from_classes(eye[[j]], DM.CLASSES, i, w)[0] for j in range(5)]
    # -2 -> 161.5 คี่ · -1 -> 162 คู่ · 0 -> 162.5 คู่ · +1 -> 163 คี่ · +2 -> 163.5 คี่
    assert got == [1.0, 0.0, 0.0, 1.0, 1.0]


def test_p_odd_across_200_uses_correct_ticks():
    # 199.5 (คี่): +1 -> 200 คู่ · +2 -> 201 คี่
    i = DM.tick_index([199.5])
    w = {-2: {-2: 1.0}, 2: {2: 1.0}}
    eye = np.eye(5)
    assert DM.p_odd_from_classes(eye[[3]], DM.CLASSES, i, w)[0] == 0.0
    assert DM.p_odd_from_classes(eye[[4]], DM.CLASSES, i, w)[0] == 1.0


def test_open_class_weights_from_train_only_and_mix():
    w = DM.open_class_weights([3, 2, 2, -4, 0, 1])
    assert w[2] == {2: 2 / 3, 3: 1 / 3} and w[-2] == {-4: 1.0}
    i = DM.tick_index([241.0])                 # tick 1: +2 -> 243 คี่ · +3 -> 244 คู่
    p = DM.p_odd_from_classes(np.array([[0, 0, 0, 0, 1.0]]), DM.CLASSES, i, w)[0]
    assert np.isclose(p, 2 / 3)
    assert DM.open_class_weights([0, 1])[2] == {2: 1.0}


def test_intraday_features_ignore_bars_from_1600():
    idx = pd.DatetimeIndex([pd.Timestamp(f"2026-10-01 {h}:00", tz="Asia/Bangkok")
                            for h in (10, 11, 12, 13, 14, 15, 16)])
    bars = pd.DataFrame({"Open": [160.0] * 7, "High": [161.0] * 6 + [190.0],
                         "Low": [159.0] * 6 + [130.0], "Close": [160.0] * 6 + [170.0]}, index=idx)
    day = {pd.Timestamp("2026-10-01")}
    a = DM.intraday_tick_features(bars, day)
    poisoned = bars.copy()
    poisoned.iloc[6] = [999.0, 999.0, 1.0, 500.0]
    b = DM.intraday_tick_features(poisoned, day)
    assert a.equals(b)
    assert a["range_ticks_so_far"].iloc[0] == 4    # 159 -> 161 = 4 tick (0.50) ไม่รวมแท่ง 16:00
