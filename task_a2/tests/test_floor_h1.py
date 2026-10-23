"""
tests/test_floor_h1.py — label floor + กฎ H1 ของ task_a2 (2026-10-24)

    python -m pytest -q tests/test_floor_h1.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import predict_live as PL   # noqa: E402
import features_a2  # noqa: E402
from log_schema import migrate_log_columns  # noqa: E402


def test_h1_predicts_only_in_half_baht_tick_range():
    assert PL.h1_rule(162.5) == (0.5, "predict", 0.0)     # 162 คู่
    assert PL.h1_rule(163.0) == (0.5, "predict", 1.0)
    assert PL.h1_rule(199.5) == (0.5, "predict", 1.0)
    tick, status, parity = PL.h1_rule(200.0)               # tick 1 บาท -> abstain
    assert (tick, status) == (1.0, "abstain") and np.isnan(parity)
    tick, status, parity = PL.h1_rule(99.75)               # tick 0.25 -> abstain
    assert (tick, status) == (0.25, "abstain") and np.isnan(parity)


def test_dist_to_boundary_is_distance_to_whole_baht_in_ticks():
    d = features_a2._dist_to_round_boundary_ticks
    assert d(np.array([163.0]), np.array([0.5]))[0] == 0.0
    assert d(np.array([162.5]), np.array([0.5]))[0] == 1.0     # ห่าง 163.00 หนึ่ง tick
    assert d(np.array([87.25]), np.array([0.25]))[0] == 1.0
    assert d(np.array([87.50]), np.array([0.25]))[0] == 2.0
    assert d(np.array([231.0]), np.array([1.0]))[0] == 0.0
    assert d(np.array([162.99999999]), np.array([0.5]))[0] == 0.0


def test_real_a2_log_migrates_without_touching_old_rows():
    path = ROOT / "results" / "prediction_log.csv"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "log.csv"
        tmp.write_bytes(path.read_bytes())
        migrate_log_columns(tmp, PL.OLD_LOG_COLUMNS, PL.NEW_LOG_COLUMNS)
        before = pd.read_csv(path, dtype=str)
        after = pd.read_csv(tmp, dtype=str)
        assert after[before.columns].fillna("").equals(before.fillna(""))
        assert after[PL.NEW_LOG_COLUMNS].isna().all().all()
        assert after.columns.tolist() == PL.LOG_COLUMNS
