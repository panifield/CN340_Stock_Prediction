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


def _wrap(model):
    """เส้นทางเดียวกับ main.py: TransformedTargetRegressor -> Pipeline -> model"""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler
    return TransformedTargetRegressor(regressor=model, transformer=StandardScaler())


def test_no_weight_is_identical():
    """
    RECENCY_HALF_LIFE = None -> ทั้ง 3 โมเดลต้องทำนายเหมือนไม่มี §3 เป๊ะ (§3.6)
    เทียบ fit(X, y, **fit_params) ที่สร้างจาก recency_weights(n, None)
    กับ fit(X, y) ตรง ๆ บนข้อมูลจริง (ใช้ 400 แถวแรกของ train เพื่อความเร็ว)
    """
    from models import get_regressors
    from weighting import recency_weights
    from targets import build_targets
    from splits import prepare_xy
    df = _dev_df()
    X, y = prepare_xy(build_features(df, verbose=False),
                      build_targets(df, verbose=False)["y_return"], verbose=False)
    X, y = X.iloc[:400], y.iloc[:400]
    X_eval = build_features(df, verbose=False).loc[X.index[-50:]]
    def fresh(name):
        # n_jobs=1 เฉพาะใน test: RF n_jobs=-1 รวมผลต้นไม้แบบขนานลำดับไม่แน่นอน
        # -> fit(X, y) สองครั้งเหมือนกันทุกอย่างยังต่างกันระดับ 1e-18
        # ถ้าไม่ปิด การเทียบ "เป๊ะ" จะไม่มีความหมาย (ค่า production ไม่ถูกแตะ)
        m = get_regressors()[name]
        if "n_jobs" in m.named_steps["model"].get_params():
            m.set_params(model__n_jobs=1)
        return _wrap(m)
    for name in get_regressors():
        w = recency_weights(len(X), None)
        assert w is None
        fit_params = {} if w is None else {"model__sample_weight": w}
        a = fresh(name).fit(X, y, **fit_params).predict(X_eval)
        b = fresh(name).fit(X, y).predict(X_eval)
        np.testing.assert_array_equal(a, b, err_msg=name)


def test_recency_weights_shape():
    """นิยามน้ำหนัก: normalize, ใหม่กว่าหนักกว่า, ห่าง half_life = ครึ่งหนึ่ง (§3.6)"""
    from weighting import recency_weights
    w = recency_weights(1000, half_life=100)
    assert len(w) == 1000
    assert np.isclose(w.mean(), 1.0)
    assert w[-1] == w.max() and np.all(np.diff(w) > 0)
    assert np.isclose(w[-1 - 100] / w[-1], 0.5)
    assert recency_weights(1000, None) is None
    for bad in (0, -5):
        try:
            recency_weights(1000, bad)
        except ValueError:
            continue
        raise AssertionError(f"half_life={bad} ต้อง raise")


def test_half_life_keys_match_models():
    """key ของ RECENCY_HALF_LIFE ต้องตรงชื่อโมเดลทุกตัวอักษร (§3.6)"""
    from config import RECENCY_HALF_LIFE
    from models import get_regressors
    assert set(RECENCY_HALF_LIFE) == set(get_regressors())


def test_recency_weight_takes_effect():
    """
    น้ำหนักต้องมีผลจริงผ่านเส้นทางจริงทั้ง 3 โมเดล (§3.6)
    TransformedTargetRegressor -> Pipeline -> model (ANN ผ่าน SeedAveragedRegressor อีกชั้น)
    y = +1 ครึ่งแรก, -1 ครึ่งหลัง · X เป็น noise (ไม่มีข้อมูลเวลา)
      half_life 5 แถว -> ทำนายต้องเอียงไป -1 ชัดเจน
      ไม่ถ่วงน้ำหนัก   -> ทำนายต้องอยู่ราว 0
    """
    from models import get_regressors
    from weighting import recency_weights
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.r_[np.ones(n // 2), -np.ones(n // 2)])
    X_eval = pd.DataFrame(rng.normal(size=(100, 5)), columns=X.columns)
    w = recency_weights(n, 5)
    for name in get_regressors():
        weighted = _wrap(get_regressors()[name]).fit(
            X, y, model__sample_weight=w).predict(X_eval).mean()
        plain = _wrap(get_regressors()[name]).fit(X, y).predict(X_eval).mean()
        assert weighted < -0.7, f"{name}: ถ่วงน้ำหนักแล้วยังไม่เอียงไป -1 ({weighted:.3f})"
        assert abs(plain) < 0.3, f"{name}: ไม่ถ่วงน้ำหนักควรอยู่ราว 0 ({plain:.3f})"


def test_signal_closes_position():
    """
    position = [1, 1] -> ซื้อ 1 + ขายปิดวันสุดท้าย 1 = turnover 2, n_trades 1 (§5.4)
    ต้นทุนรวมต้องเท่า 1 round-trip พอดี (turnover 2 x cost_rt/2)
    """
    from trading_costs import signal_economics
    cost_rt = 0.003
    r = signal_economics(np.array([0.01, -0.02]), np.array([0.05, 0.05]), cost_rt)
    assert r["n_days_long"] == 2
    assert r["n_trades"] == 1
    assert np.isclose(r["total_cost"], 2 * cost_rt / 2)        # turnover.sum() == 2
    assert np.isclose(r["gross_return"], -0.01)
    assert np.isclose(r["net_return"], -0.01 - cost_rt)
    # ไม่เคยเกิน threshold -> ไม่มี trade และต้นทุน 0 (ผลลัพธ์ ไม่ใช่บั๊ก)
    r0 = signal_economics(np.array([0.01, 0.02]), np.array([0.0, 0.001]), cost_rt)
    assert r0["n_trades"] == 0 and r0["total_cost"] == 0.0


def test_cost_and_tick():
    """cost_round_trip ราว 0.0034 (±0.0002) และ tick ตาม band รวมขอบพอดี (§5.4)"""
    from trading_costs import cost_round_trip, tick_size
    assert abs(cost_round_trip() - 0.0034) <= 0.0002
    assert tick_size(150) == 0.50
    assert tick_size(250) == 1.00
    assert tick_size(450) == 2.00
    assert tick_size(400) == 2.00


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
