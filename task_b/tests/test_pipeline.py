"""
tests/test_pipeline.py — ป้องกันบั๊กที่เคยเกิดจริง (E6)

รันได้ 2 แบบ:
    python tests/test_pipeline.py
    python -m pytest tests/          (ถ้าลง pytest ไว้)

ทุก test ใช้ข้อมูลถึง VAL_END เท่านั้น -- ไม่มี test rows เข้ามาเลย
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor

from config import SPLIT_BY_DATE, REQUIRED_LOCK_FIELDS, TICKERS
from data_loader import load_stock
from features import build_features, build_market_features, build_live_feature
from models import SeedAveragedRegressor
from main import verify_lock


def _dev_df(ticker=TICKERS[0]):
    """ข้อมูลของ ticker ตัดที่ val_end (ไม่มี test rows)"""
    df = load_stock(ticker, verbose=False)
    return df.loc[:SPLIT_BY_DATE["val_end"]].copy()


def test_market_feature_alignment():
    """market feature แถว t ต้องเท่ากับ raw indicator แถว t-1"""
    for ticker in TICKERS:
        df = _dev_df(ticker)
        feats = build_features(df, verbose=False)
        raw = build_market_features(df)

        # C1: ต้องไม่มีราคาดิบ
        for col in ["close", "open", "high", "low", "volume"]:
            assert f"{col}_prev" not in feats.columns, f"ยังมี {col}_prev"
        assert raw.shape[1] == 24, f"market ต้องมี 24 ตัว ได้ {raw.shape[1]}"
        assert feats.shape[1] == 29, f"รวมต้องมี 29 ตัว ได้ {feats.shape[1]}"

        # ทุกแถว ไม่ใช่แค่แถวตัวอย่าง
        for col in raw.columns:
            got = feats[f"{col}_prev"].iloc[1:].to_numpy()
            want = raw[col].iloc[:-1].to_numpy()
            assert np.allclose(got, want, equal_nan=True), f"{ticker} {col} ไม่ตรง t-1"
        assert feats.filter(like="_prev").iloc[0].isna().all(), "แถวแรกต้องเป็น NaN ทั้งแถว"


def test_dow_matches_target_day():
    """dow ของแถว t ต้องตรงกับวันของ t จริง -- รวมเคสวันจันทร์หลังวันหยุด"""
    df = _dev_df()
    feats = build_features(df, verbose=False)

    dow_cols = [f"dow_{d}" for d in range(5)]
    expected = pd.get_dummies(df.index.dayofweek).reindex(
        columns=range(5), fill_value=0).to_numpy().astype(int)
    assert (feats[dow_cols].to_numpy() == expected).all(), "dow ไม่ตรงกับวันของแถว"

    # เคสที่ v1 ผิด: มีวันทำการหายไประหว่างแถว (วันหยุดคั่น)
    idx = df.index
    gaps = [i for i in range(1, len(idx))
            if len(pd.bdate_range(idx[i - 1], idx[i])) > 2]
    assert len(gaps) > 0, "ข้อมูลต้องมีวันหยุดคั่นอย่างน้อย 1 ครั้ง"
    for i in gaps:
        day, prev_day = idx[i].dayofweek, idx[i - 1].dayofweek
        assert feats[f"dow_{day}"].iloc[i] == 1
        assert feats[dow_cols].iloc[i].sum() == 1
        if prev_day != day:          # ต้องไม่ใช่ dow ของวันซื้อขายก่อนหน้า
            assert feats[f"dow_{prev_day}"].iloc[i] == 0


def test_nan_only_in_warmup():
    """NaN ต้องอยู่ต้นชุดต่อเนื่องเท่านั้น ไม่มีกลางชุด"""
    for ticker in TICKERS:
        feats = build_features(_dev_df(ticker), verbose=False)
        pos = np.where((~feats.notna().all(axis=1)).to_numpy())[0]
        assert len(pos) > 0
        assert (pos == np.arange(len(pos))).all(), (
            f"{ticker}: มี NaN กลางชุดที่ตำแหน่ง {pos[pos != np.arange(len(pos))][:5]}")
        assert np.isfinite(feats.iloc[len(pos):].to_numpy(dtype=float)).all(), (
            f"{ticker}: มี inf หลังช่วง warm-up")


def test_lock_gate_blocks_test_mode():
    """โหมด test: ไม่มี lock / lock ว่าง / lock ขาด field -> ต้อง fail ทั้งสามกรณี"""
    def must_fail(path, why):
        try:
            verify_lock(path)
        except RuntimeError:
            return
        raise AssertionError(f"verify_lock ต้อง fail กรณี: {why}")

    with tempfile.TemporaryDirectory() as d:
        lock = Path(d) / "PRE_TEST_LOCK.md"
        must_fail(lock, "ไม่มีไฟล์")

        lock.write_text("", encoding="utf-8")
        must_fail(lock, "ไฟล์ว่าง (touch)")

        lock.write_text("\n".join(f"{f} x" for f in REQUIRED_LOCK_FIELDS[:-1]),
                        encoding="utf-8")
        must_fail(lock, "ขาด field สุดท้าย")

        lines = [f"{f} x" for f in REQUIRED_LOCK_FIELDS]
        lines[0] = REQUIRED_LOCK_FIELDS[0]            # หัวข้อมีแต่ไม่มีค่า
        lock.write_text("\n".join(lines), encoding="utf-8")
        must_fail(lock, "field แรกว่าง")

        lock.write_text("\n".join(f"{f} x" for f in REQUIRED_LOCK_FIELDS),
                        encoding="utf-8")
        verify_lock(lock)                             # ครบ -> ต้องผ่าน


def test_seed_averaging_is_mean():
    """SeedAveragedRegressor.predict = ค่าเฉลี่ยของแต่ละ seed จริง"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = X[:, 0] + rng.normal(scale=0.5, size=200)
    base = DecisionTreeRegressor(max_depth=4, max_features=0.5)
    seeds = (0, 1, 2, 3)

    sar = SeedAveragedRegressor(base, seeds=seeds).fit(X, y)
    assert [e.random_state for e in sar.estimators_] == list(seeds)

    singles = [clone(base).set_params(random_state=s).fit(X, y).predict(X)
               for s in seeds]
    assert np.allclose(sar.predict(X), np.mean(singles, axis=0))
    # seed ต่างกันต้องให้โมเดลต่างกันจริง ไม่งั้นการเฉลี่ยไม่มีความหมาย
    assert not np.allclose(singles[0], singles[1])

    # ต้อง clone ได้ (Pipeline / TransformedTargetRegressor clone มันทุกครั้ง)
    c = clone(SeedAveragedRegressor(base, seeds=seeds))
    assert c.seeds == seeds and not hasattr(c, "estimators_")


def test_live_feature_matches_history():
    """
    build_live_feature() ต้องให้ค่าเท่ากับที่ build_features() ให้
    ถ้าข้อมูลของวันนั้นมีอยู่จริง -- ถ้าไม่เท่า แปลว่าเส้นทาง live
    กับ historical แยกจากกันแล้ว ซึ่งเป็นสิ่งที่ห้ามเกิด

    ใช้วันใน val เท่านั้น (2025-02-25) ไม่แตะ test
    """
    cutoff, target = pd.Timestamp("2025-02-24"), pd.Timestamp("2025-02-25")
    for ticker in TICKERS:
        df = _dev_df(ticker)                       # ถึง val_end = 2025-02-25
        hist = df.loc[:cutoff]                     # ตัดก่อน target 1 วัน

        X_live, prev_close, dc = build_live_feature(hist, target, verbose=False)
        assert dc == cutoff, f"data_cutoff ต้องเป็น {cutoff.date()}"
        assert prev_close == float(df.loc[cutoff, "Close"])
        assert X_live.shape == (1, 29)
        assert not X_live.isna().any().any(), "X_live ต้องไม่มี NaN"

        # market feature ต้องตรงกับที่คำนวณจากข้อมูลจริงทั้งชุด
        X_full = build_features(df, verbose=False)
        prev_cols = [c for c in X_live.columns if c.endswith("_prev")]
        assert np.allclose(X_live[prev_cols].to_numpy(float),
                           X_full.loc[[target], prev_cols].to_numpy(float)), (
            f"{ticker}: live feature ไม่ตรงกับ historical feature ของวันเดียวกัน")

        # calendar ต้องเป็นของ target ไม่ใช่ของ cutoff (2025-02-25 = อังคาร)
        assert X_live["dow_1"].iloc[0] == 1
        assert X_live[[f"dow_{d}" for d in range(5)]].iloc[0].sum() == 1

        # guard: วันที่มีข้อมูลแล้ว / อยู่ในอดีต / เสาร์-อาทิตย์ ต้อง error
        for bad_date, why in [(cutoff, "มีข้อมูลแล้ว"),
                              (pd.Timestamp("2024-01-02"), "อยู่ในอดีต"),
                              (pd.Timestamp("2025-03-01"), "วันเสาร์")]:
            try:
                build_live_feature(hist, bad_date, verbose=False)
            except ValueError:
                continue
            raise AssertionError(f"build_live_feature ต้อง fail: {why}")


def test_shape_constant_prediction():
    """ทำนายค่าคงที่ -> StdRatio == 0.0 เป๊ะ และ Rho เป็น NaN (§2.3)"""
    from evaluate import prediction_shape
    y_true = np.random.default_rng(0).normal(0, 0.01, 362)
    s = prediction_shape(y_true, np.full(362, 0.001))
    assert s["StdRatio"] == 0.0, s
    assert np.isnan(s["Rho"]), s


def test_shape_constant_truth():
    """ค่าจริงคงที่ -> StdRatio และ Rho เป็น NaN (§2.3)"""
    from evaluate import prediction_shape
    y_pred = np.random.default_rng(1).normal(0, 0.01, 362)
    s = prediction_shape(np.full(362, 0.001), y_pred)
    assert np.isnan(s["StdRatio"]) and np.isnan(s["Rho"]), s


def test_shape_real_baselines():
    """
    baseline ตัวจริงทั้งสองต้องได้ StdRatio == 0.0 และ Rho NaN (§2.3)
    ใช้ y_train จริงของ KBANK (ถึง train_end) เพื่อให้ Mean Return
    เป็นค่าเฉลี่ยจริงที่เคยทำให้ std ได้ ~1e-19
    """
    from evaluate import prediction_shape
    from baselines import get_regression_baselines
    from targets import build_targets
    df = _dev_df()
    y = build_targets(df, verbose=False)["y_return"].dropna()
    y_train = y.loc[:SPLIT_BY_DATE["train_end"]]
    y_eval = y.loc[y.index > pd.Timestamp(SPLIT_BY_DATE["train_end"])]
    base = get_regression_baselines(y_train, y_eval)
    assert set(base) == {"Baseline: Naive (RW)", "Baseline: Mean Return"}
    for name, p in base.items():
        s = prediction_shape(y_eval, p)
        assert s["StdRatio"] == 0.0, (name, s)
        assert np.isnan(s["Rho"]), (name, s)


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
