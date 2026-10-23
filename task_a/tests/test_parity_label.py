"""
tests/test_parity_label.py — rounding.parity_label (label floor ตั้งแต่ 2026-10-24)

    python -m pytest -q tests/test_parity_label.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rounding import parity_label, floor_baht  # noqa: E402
from targets import build_targets  # noqa: E402

EVEN, ODD = 0.0, 1.0


def test_spec_cases():
    assert parity_label(162.0) == EVEN
    assert parity_label(162.5) == EVEN
    assert parity_label(163.0) == ODD
    assert parity_label(199.5) == ODD
    # เศษ float ของ 163.00 -- แปลงเป็นสตางค์ก่อน จึงเป็น 163 = คี่ (ตัดสินใจ 2026-10-24)
    assert parity_label(162.99999999) == ODD


def test_float_noise_below_half_stays_in_same_baht():
    assert parity_label(162.49999999) == EVEN
    assert parity_label(162.50000001) == EVEN


def test_quarter_ticks():
    assert parity_label(87.25) == ODD
    assert parity_label(87.75) == ODD
    assert parity_label(88.00) == EVEN


def test_vectorized_matches_scalar_and_keeps_index_and_nan():
    prices = pd.Series([162.0, 162.5, 163.0, 199.5, np.nan], index=list("abcde"))
    out = parity_label(prices)
    assert list(out.index) == list("abcde")
    assert out.iloc[:4].tolist() == [EVEN, EVEN, ODD, ODD]
    assert np.isnan(out.iloc[4])
    assert parity_label(np.array([162.5, 163.0])).tolist() == [EVEN, ODD]


def test_floor_baht():
    assert floor_baht(162.5) == 162
    assert floor_baht(pd.Series([199.5, 200.0])).tolist() == [199, 200]


def test_targets_use_floor_label():
    idx = pd.date_range("2026-01-05", periods=3, freq="B")
    df = pd.DataFrame({"Close": [162.0, 162.5, 163.0]}, index=idx)
    t = build_targets(df, verbose=False)
    assert t["y_parity"].tolist() == [EVEN, EVEN, ODD]
    assert t["y_flip"].iloc[1:].tolist() == [0.0, 1.0]   # 162 -> 162 ไม่พลิก · 162 -> 163 พลิก
