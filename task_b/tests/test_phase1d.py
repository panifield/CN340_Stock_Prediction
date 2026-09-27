"""
tests/test_phase1d.py — Phase 1D (Mode A 16:00) builder / metric / policy tests

    python tests/test_phase1d.py

ใช้ข้อมูลสังเคราะห์เท่านั้น -- ไม่ fit โมเดลใด ๆ บน target จริงของ Phase 1D
(test_smoke_runs รัน tune_1600.py --smoke ราว 3 นาที)
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import intraday_1600 as I      # noqa: E402
import metrics_1600 as M       # noqa: E402
import tune_1600 as T          # noqa: E402


def _bars(n_days=80, seed=1):
    return I.prepare_bars(T.synthetic_bars(n_days=n_days, seed=seed))


def _day(bars, k=40):
    """วันที่ลำดับ k (อยู่หลัง warm-up และมีวันถัดไป)"""
    return sorted(bars["date"].unique())[k]


def _mod(bars, date, hour, **vals):
    b = bars.copy()
    m = (b["date"] == date) & (b["hour"] == hour)
    assert m.sum() == 1
    for c, v in vals.items():
        b.loc[m, c] = v
    return b


def _drop(bars, date, hour):
    return bars[~((bars["date"] == date) & (bars["hour"] == hour))].reset_index(drop=True)


def test_01_target():
    bars = _bars()
    X, y, info, _ = I.build_dataset(bars)
    t = X.index[10]
    c = bars[bars["date"] == t].set_index("hour")["Close"]
    assert np.isclose(y.loc[t], c[16] / c[15] - 1)
    assert np.isclose(info.loc[t, "C15"], c[15])
    assert X.shape[1] == 21 and list(X.columns) == I.FEATURE_COLS


def test_02_bar16_does_not_touch_features_of_same_day():
    bars = _bars()
    t = _day(bars)
    X0, y0, _, _ = I.build_dataset(bars)
    b = _mod(bars, t, 16, Open=999.0, High=999.0, Low=1.0, Close=777.0,
             **{"Adj Close": 777.0, "Volume": 123})
    X1, y1, _, _ = I.build_dataset(b)
    pd.testing.assert_series_equal(X0.loc[t], X1.loc[t])
    assert y0.loc[t] != y1.loc[t]                      # target เปลี่ยนตามนิยาม


def test_03_live_without_bar16_equals_historical():
    bars = _bars()
    t = _day(bars)
    X, _, _, _ = I.build_dataset(bars)
    live = I.build_1600_live_feature(_drop(bars, t, 16), t)
    pd.testing.assert_series_equal(live["X_live"].iloc[0], X.loc[t], check_names=False)


def test_04_historical_requires_bar16():
    bars = _bars()
    t = _day(bars)
    X, _, _, rep = I.build_dataset(_drop(bars, t, 16))
    assert t not in X.index
    assert rep["dropped_days"][str(t.date())] == [16]


def test_05_missing_required_bar():
    bars = _bars()
    t = _day(bars)
    for h in (14, 15):
        b = _drop(bars, t, h)
        X, _, _, rep = I.build_dataset(b)
        assert t not in X.index and h in rep["dropped_days"][str(t.date())]
        try:
            I.build_1600_live_feature(b, t)
        except ValueError:
            continue
        raise AssertionError(f"live ต้อง raise เมื่อขาดแท่ง {h}")


def test_06_bar13_ignored():
    bars = _bars()
    X0, _, _, _ = I.build_dataset(bars)
    b13 = bars.copy()
    m = b13["hour"] == 13
    b13.loc[m, ["Open", "High", "Low", "Close", "Adj Close"]] = 5000.0
    b13.loc[m, "Volume"] = 10 ** 9
    pd.testing.assert_frame_equal(X0, I.build_dataset(b13)[0])
    pd.testing.assert_frame_equal(X0, I.build_dataset(bars[bars["hour"] != 13])[0])


def test_07_volume_bar10_ignored():
    bars = _bars()
    X0, _, _, _ = I.build_dataset(bars)
    b = bars.copy()
    b.loc[b["hour"] == 10, "Volume"] = 10 ** 9
    pd.testing.assert_frame_equal(X0, I.build_dataset(b)[0])


def test_08_target_change_keeps_own_lags():
    bars = _bars()
    t = _day(bars)
    X0, _, _, _ = I.build_dataset(bars)
    X1, _, _, _ = I.build_dataset(_mod(bars, t, 16, Close=1.0))
    lags = [c for c in I.FEATURE_COLS if c.startswith("y_lag")]
    pd.testing.assert_series_equal(X0.loc[t, lags], X1.loc[t, lags])
    nxt = X0.index[X0.index.get_loc(t) + 1]
    assert X0.loc[nxt, "y_lag1"] != X1.loc[nxt, "y_lag1"]   # วันถัดไปเปลี่ยนตามนิยาม


def test_09_future_data_irrelevant():
    bars = _bars()
    t = _day(bars)
    X0, _, _, _ = I.build_dataset(bars)
    cut = bars[bars["date"] <= t]
    X1, _, _, _ = I.build_dataset(cut)
    pd.testing.assert_frame_equal(X0.loc[:t], X1)
    b = bars.copy()
    b.loc[b["date"] > t, ["Open", "High", "Low", "Close"]] *= 3.0
    pd.testing.assert_frame_equal(X0.loc[:t], I.build_dataset(b)[0].loc[:t])


def test_10_live_full_snapshot_equals_truncated():
    bars = _bars()
    t = _day(bars)
    full = I.build_1600_live_feature(bars, t)
    tf = I.time_fields(t)
    trunc = bars[(bars["date"] < t) | ((bars["date"] == t) & (bars["ts"] < tf["feature_window_end"]))]
    part = I.build_1600_live_feature(trunc.reset_index(drop=True), t)
    pd.testing.assert_frame_equal(full["X_live"], part["X_live"])
    pd.testing.assert_frame_equal(full["X_train"], part["X_train"])
    assert full["close_bar15"] == part["close_bar15"]


def test_11_walk_forward_train_only():
    bars = _bars(n_days=420)
    ds = {"SYN": I.build_dataset(bars)}
    data = T.prepare(ds)
    d = data["SYN"]
    for tr, ev in d["folds"]:
        assert tr.max() < ev.min() and len(set(tr) & set(ev)) == 0
    last = d["folds"][-1][1][-1]
    assert d["omitted"] == [str(x.date()) for x in d["X"].index[last + 1:]]
    tr, ev = d["folds"][0]
    X_tr, y_tr = d["X"].iloc[tr], d["y"].iloc[tr]
    _, fitted = T.fit_predict(T.make_model("ANN (MLP)", {"alpha": 7, "hidden_layer_sizes": (8,)},
                                           ann_seeds=[0]), X_tr, y_tr, d["X"].iloc[ev])
    np.testing.assert_allclose(fitted.regressor_.named_steps["scale"].mean_, X_tr.mean().values)
    np.testing.assert_allclose(fitted.transformer_.mean_[0], y_tr.mean())


def test_12_live_train_days():
    bars = _bars()
    t = _day(bars)
    live = I.build_1600_live_feature(bars, t)
    X, _, _, _ = I.build_dataset(bars)
    expected = X[X.index < t]
    assert live["train_days"] == len(expected) == len(live["X_train"])
    pd.testing.assert_frame_equal(live["X_train"], expected)
    assert (live["X_train"].index < t).all()


def test_13_causal_time():
    bars = _bars()
    t = _day(bars)
    tf = I.time_fields(t)
    assert str(tf["feature_window_end"].tzinfo) in ("UTC+07:00", "+07:00")
    assert tf["last_feature_bar_start"] == tf["feature_window_end"] - pd.Timedelta(hours=1)
    assert tf["feature_window_end"] <= tf["target_bar_start"]
    day = bars[bars["date"] == t]
    feat = day.loc[day["hour"].isin(I.FEATURE_HOURS), "ts"]
    assert (feat < tf["target_bar_start"]).all()
    I.check_causal(feat, tf)
    try:
        I.check_causal(day.loc[day["hour"] == 16, "ts"], tf)
    except AssertionError:
        return
    raise AssertionError("check_causal ต้องปฏิเสธแท่ง 16:00")


def test_14_seed_average_and_live_seeds():
    bars = _bars()
    X, y, _, _ = I.build_dataset(bars)
    cfg = {"alpha": 7, "hidden_layer_sizes": (8,)}
    pipe = T.make_model("ANN (MLP)", cfg, ann_seeds=[0, 1, 2]).fit(X, y)
    sar = pipe.named_steps["model"]
    Xs = pipe[:-1].transform(X)
    each = np.mean([e.predict(Xs) for e in sar.estimators_], axis=0)
    np.testing.assert_allclose(pipe.predict(X), each)
    assert tuple(T.make_live_model("ANN (MLP)", cfg).named_steps["model"].seeds) == (0, 1, 2)
    assert T.LIVE_ANN_SEEDS == T.SEARCH_ANN_SEEDS == [0, 1, 2]


def test_15_no_test_label():
    bars = _bars(n_days=420)
    data = T.prepare({"SYN": I.build_dataset(bars)})
    labels = list(T.fold_table(data).columns) + list(data["SYN"]["report"].keys())
    assert not any("test" in s.lower() for s in labels)
    src = (ROOT / "intraday_1600.py").read_text(encoding="utf-8")
    assert "historical test" not in src.replace("ไม่มี historical test", "")


def test_16_no_investing_source():
    code = ("import sys; import intraday_1600, tune_1600, metrics_1600; "
            "print('data_loader' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "False", r.stderr
    import ast
    for f in ("intraday_1600.py", "tune_1600.py", "metrics_1600.py"):
        tree = ast.parse((ROOT / f).read_text(encoding="utf-8"))
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in n.names] + [getattr(n, "module", None) or ""]
                assert not any("data_loader" in x or x == "RAW_DATA_DIR" for x in names), f
            if isinstance(n, ast.Name):
                assert n.id != "RAW_DATA_DIR", f
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
                s = n.value.replace("raw_data_intraday", "")
                assert "raw_data" not in s, f"{f}: {n.value!r}"


def test_17_relmae_and_tie_rule():
    y = np.array([0.01, -0.02, 0.0, 0.03])
    p = np.array([0.0, -0.01, 0.01, 0.01])
    mae, nm = np.mean(np.abs(p - y)), np.mean(np.abs(y))        # 0.0125, 0.015
    assert np.isclose(M.rel_mae(mae, nm), 0.0125 / 0.015)
    assert np.isclose(M.primary_score([0.9, 1.0, 1.1, 0.95, 1.05, 1.0]), 1.0)
    s = pd.DataFrame({
        "model": ["ANN (MLP)"] * 3, "config_id": [1, 0, 7], "valid": True, "passed_guard": True,
        "primary_score": [0.9800, 0.9805, 0.9700]})
    assert T.select_winner(s, "ANN (MLP)") == 7                  # ห่างเกิน 1e-3 -> score ชนะ
    s["primary_score"] = [0.9800, 0.9801, 0.9795]                # ทั้งสามเสมอ (< 1e-3)
    # (8,) alpha 1 = id 0 · (8,4) alpha 1 = id 1 · (8,4) alpha 20 = id 7 -> neuron น้อยสุดคือ id 0
    assert T.select_winner(s, "ANN (MLP)") == 0
    s["primary_score"] = [0.9795, 0.9801, 0.9790]                # id1 (8,4) a1 · id0 (8,) a1 · id7 (8,4) a20
    # config_id ในตาราง = [1, 0, 7] -> id1 = .9795 (ห่าง .0005 เสมอ) · id0 = .9801 (ห่าง .0011 ไม่เสมอ)
    assert T.select_winner(s, "ANN (MLP)") == 7                  # id1 กับ id7 neuron เท่ากัน -> alpha สูงกว่า
    r = pd.DataFrame({"model": ["Random Forest"] * 2, "config_id": [4, 0], "valid": True,
                      "passed_guard": True, "primary_score": [0.9700, 0.9705]})
    assert T.select_winner(r, "Random Forest") == 0              # depth 3 < 5
    r.loc[1, "passed_guard"] = False
    assert T.select_winner(r, "Random Forest") == 4
    r["passed_guard"] = False
    assert T.select_winner(r, "Random Forest") is None           # family ไม่มี valid config


def _raises(exc, fn, *a):
    try:
        fn(*a)
    except exc:
        return
    raise AssertionError(f"ต้อง raise {exc.__name__}")


def test_18_nan_policy():
    _raises(M.RankingUndefined, M.rel_mae, 0.01, 0.0)
    _raises(M.IncompleteEvaluations, M.primary_score, [1.0] * 5)
    _raises(M.IncompleteEvaluations, M.primary_score, [1.0] * 5 + [np.nan])
    ev = pd.DataFrame({"model": "XGBoost", "config_id": 0, "config": "{}",
                       "pred_finite": [True] * 5 + [False],
                       "relMAE": [1.0] * 5 + [np.nan], "MAE_return": 0.01, "StdRatio": 0.2})
    sc = T.score(ev)
    assert not sc["valid"].iloc[0] and not sc["passed_guard"].iloc[0]
    _raises(M.IncompleteEvaluations, T.score, ev.iloc[:5])
    sh = M.shape(np.zeros(4), np.array([0.1, 0.2, 0.0, 0.1]))
    assert np.isnan(sh["StdRatio"]) and np.isnan(sh["Rho"])
    assert not M.passes_guard([sh["StdRatio"]] + [0.2] * 5)
    assert np.isnan(M.r2_oos(np.zeros(3), np.ones(3)))
    assert np.isnan(M.main_metrics(np.zeros(3), np.ones(3), np.ones(3) * 100)["R2_return"])
    e = M.economic_diagnostics(np.array([0.01, -0.01]), np.zeros(2), np.array([250.0, 250.0]))
    assert e["signal_days"] == 0 and np.isnan(e["sign_hit_rate_on_signal"])


def test_19_economic_units():
    cost = 0.0034
    y = np.array([0.004, -0.001, 0.0, -0.005])
    p = np.array([0.005, -0.004, 0.004, 0.001])
    c15 = np.array([250.0, 250.0, 150.0, 450.0])      # tick 1.00, 1.00, 0.50, 2.00
    e = M.economic_diagnostics(y, p, c15, cost=cost)
    # |y|: .004 .001 0 .005 -> >= cost: 2/4
    assert np.isclose(e["frac_actual_ge_cost"], 0.5)
    # บาท: 1.00 .25 0 2.25 vs tick 1 1 .5 2 -> 2/4
    assert np.isclose(e["frac_actual_ge_1tick"], 0.5)
    # |p| > cost: .005 .004 .004 .001 -> 3 วัน
    assert e["signal_days"] == 3 and np.isclose(e["frac_pred_gt_cost"], 0.75)
    # sign บน signal days: (+,+) hit · (-,-) hit · (+,0) y=0 ไม่นับ -> 2/3
    assert np.isclose(e["sign_hit_rate_on_signal"], 2 / 3)
    assert e["signal_days_y_zero"] == 1
    assert np.isclose(e["mean_actual_move_baht"], (1.0 + 0.25 + 0.0 + 2.25) / 4)


def test_20_bom_header():
    df = T.synthetic_bars(n_days=30)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.csv"
        p.write_bytes(b"\xef\xbb\xbf" + df.to_csv(index=False).encode("utf-8"))
        b = I.read_bars(p, expected_sha256=I.sha256_file(p))
        assert "Datetime" in b.columns and len(b) == len(df)
        _raises(RuntimeError, I.read_bars, p, "0" * 64)


def test_21_smoke_runs():
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, "tune_1600.py", "--smoke", "--out", d],
                           cwd=ROOT, capture_output=True, text=True,
                           env={**__import__("os").environ, "PYTHONUTF8": "1"})
        assert r.returncode == 0, r.stderr[-2000:]
        for f in ("phase1d_scores.csv", "phase1d_evals.csv", "phase1d_folds.csv",
                  "phase1d_baselines.csv", "phase1d_econ.csv", "phase1d_summary.md"):
            assert (Path(d) / f).exists(), f
        assert len(pd.read_csv(Path(d) / "phase1d_scores.csv")) == 24


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
