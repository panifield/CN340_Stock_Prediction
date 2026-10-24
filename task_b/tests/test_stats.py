"""
tests/test_stats.py — ตรวจ stats_tests.py ด้วยข้อมูลสังเคราะห์ (feedback ข้อ 2)

รันได้ 2 แบบ:
    python tests/test_stats.py
    python -m pytest tests/

ไม่โหลดข้อมูลจริงเลย -> เร็ว และไม่มีทางแตะ test set
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy import stats as sps

from stats_tests import (
    diebold_mariano, direction_metrics, mae_decomposition, full_report,
)


# ---------------------------------------------------------------
# diebold_mariano
# ---------------------------------------------------------------
def test_dm_zero_when_model_equals_naive():
    """โมเดล = Naive เป๊ะ -> d = 0 ทุกตัว -> DM = 0 และ p = 1 (ไม่ใช่ nan)"""
    rng = np.random.default_rng(0)
    e = rng.normal(0, 0.01, 200)
    for loss in ("abs", "sq"):
        r = diebold_mariano(e, e, loss=loss)
        assert r["mean_loss_diff"] == 0.0
        assert r["dm_stat"] == 0.0 and r["dm_stat_hln"] == 0.0
        assert r["p_two_sided"] == 1.0
        assert r["p_one_sided_model_better"] == 1.0
        assert r["model_better"] is False


def test_dm_small_p_when_model_clearly_better():
    """โมเดล error เล็กกว่าชัดเจนทุกวัน -> p one-sided ต้องเล็กมาก"""
    rng = np.random.default_rng(1)
    e_naive = rng.normal(0, 0.01, 300)
    e_model = e_naive * 0.5            # error ครึ่งเดียวทุกวัน
    r = diebold_mariano(e_model, e_naive, loss="abs")
    assert r["mean_loss_diff"] < 0
    assert r["model_better"] is True
    assert r["dm_stat_hln"] < 0
    assert r["p_one_sided_model_better"] < 1e-10, r["p_one_sided_model_better"]
    assert r["p_two_sided"] < 1e-10
    # ทิศกลับกันต้องได้ p one-sided เข้าใกล้ 1
    rev = diebold_mariano(e_naive, e_model, loss="abs")
    assert rev["p_one_sided_model_better"] > 0.99
    # two-sided เท่ากันทั้งสองทิศ (สมมาตร)
    assert np.isclose(rev["p_two_sided"], r["p_two_sided"])


def test_dm_one_sided_is_half_of_two_sided():
    """เมื่อ dbar < 0: p_one = p_two / 2 เป๊ะ (t-distribution สมมาตร)"""
    rng = np.random.default_rng(2)
    e_naive = rng.normal(0, 0.01, 120)
    e_model = e_naive * 0.9
    r = diebold_mariano(e_model, e_naive, loss="sq")
    assert np.isclose(r["p_one_sided_model_better"], r["p_two_sided"] / 2, rtol=1e-12)


def test_hln_correction_matches_hand_calculation():
    """
    ตรวจ HLN correction กับค่าที่คำนวณมือจากสูตร DM ตรง ๆ

    h = 1  -> V = gamma_0 / n · DM = dbar / sqrt(V) · factor = sqrt((n-1)/n)
    เทียบ p กับ scipy.stats.t ที่ df = n - 1
    """
    # ข้อมูลเล็กที่คำนวณมือได้: d = l_model - l_naive
    e_model = np.array([0.1, -0.2, 0.3, -0.1, 0.05, -0.15, 0.2, -0.05])
    e_naive = np.array([0.2, -0.3, 0.1, -0.4, 0.15, -0.05, 0.3, -0.25])
    n = len(e_model)

    d = np.abs(e_model) - np.abs(e_naive)
    dbar = d.mean()
    gamma0 = ((d - dbar) ** 2).mean()           # biased: หารด้วย n
    dm_manual = dbar / np.sqrt(gamma0 / n)
    factor_manual = np.sqrt((n + 1 - 2 * 1 + 0) / n)
    assert np.isclose(factor_manual, np.sqrt((n - 1) / n))
    hln_manual = dm_manual * factor_manual
    p_two_manual = 2 * sps.t.sf(abs(hln_manual), n - 1)
    p_one_manual = sps.t.cdf(hln_manual, n - 1)

    r = diebold_mariano(e_model, e_naive, h=1, loss="abs")
    assert r["n"] == n and r["df"] == n - 1
    assert np.isclose(r["mean_loss_diff"], dbar, rtol=1e-15)
    assert np.isclose(r["dm_stat"], dm_manual, rtol=1e-12)
    assert np.isclose(r["hln_factor"], factor_manual, rtol=1e-15)
    assert np.isclose(r["dm_stat_hln"], hln_manual, rtol=1e-12)
    assert np.isclose(r["p_two_sided"], p_two_manual, rtol=1e-12)
    assert np.isclose(r["p_one_sided_model_better"], p_one_manual, rtol=1e-12)
    # HLN ทำให้สถิติ "เบาลง" และ p ใหญ่ขึ้นเสมอ (conservative) เมื่อ h = 1
    assert abs(r["dm_stat_hln"]) < abs(r["dm_stat"])
    p_two_uncorrected = 2 * sps.norm.sf(abs(dm_manual))
    assert r["p_two_sided"] > p_two_uncorrected


def test_hln_factor_with_longer_horizon():
    """h > 1 ต้องใช้สูตรเต็ม (n + 1 - 2h + h(h-1)/n)/n และนับ autocovariance"""
    rng = np.random.default_rng(3)
    e_model = rng.normal(0, 1, 50)
    e_naive = rng.normal(0, 1, 50)
    n, h = 50, 4
    r = diebold_mariano(e_model, e_naive, h=h, loss="sq")
    want = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    assert np.isclose(r["hln_factor"], want, rtol=1e-15)
    assert r["h"] == h
    # h = 1 กับ h = 4 ต้องได้ dm_stat ต่างกัน (เพราะ long-run variance ต่างกัน)
    assert not np.isclose(r["dm_stat"], diebold_mariano(e_model, e_naive, h=1,
                                                        loss="sq")["dm_stat"])


def test_dm_rejects_bad_input():
    rng = np.random.default_rng(4)
    e = rng.normal(0, 1, 20)
    for bad in (lambda: diebold_mariano(e, e[:10]),
                lambda: diebold_mariano(e, e, loss="mape"),
                lambda: diebold_mariano(e, e, h=0),
                lambda: diebold_mariano(e[:2], e[:2]),
                lambda: diebold_mariano(e, e, h=99)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("ควร raise ValueError แต่ไม่ raise")


# ---------------------------------------------------------------
# direction_metrics
# ---------------------------------------------------------------
def test_direction_metrics_counts_only_moved_days():
    """วันที่ y = 0 ต้องไม่ถูกนับใน hit rate"""
    y_true = np.array([0.0, 0.0, 0.01, -0.01, 0.02, -0.02])
    y_pred = np.array([0.5, -0.5, 0.01, -0.01, -0.01, 0.01])   # 2 ถูก 2 ผิด
    m = direction_metrics(y_true, y_pred)
    assert np.isclose(m["frac_zero_move"], 2 / 6)
    assert m["n_moved_days"] == 4
    assert np.isclose(m["dir_hit_rate_moved_days"], 0.5)
    assert m["frac_pred_zero"] == 0.0


def test_direction_metrics_pred_zero_counts_as_miss():
    """y_pred = 0 เป๊ะ ไม่บอกทิศ -> นับว่าไม่ถูก (sign(0) = 0 != +-1)"""
    y_true = np.array([0.01, -0.01, 0.01, -0.01])
    m = direction_metrics(y_true, np.zeros(4))
    assert m["dir_hit_rate_moved_days"] == 0.0
    assert m["frac_pred_zero"] == 1.0
    # Naive เป็นค่าคงที่ -> IC ไม่นิยาม
    assert np.isnan(m["ic_pearson"]) and np.isnan(m["ic_spearman"])


def test_direction_metrics_perfect_ic():
    y_true = np.array([0.01, 0.02, -0.01, -0.03, 0.005])
    m = direction_metrics(y_true, y_true * 2.0)       # สัดส่วนเดียวกัน -> corr = 1
    assert np.isclose(m["ic_pearson"], 1.0)
    assert np.isclose(m["ic_spearman"], 1.0)
    assert np.isclose(m["dir_hit_rate_moved_days"], 1.0)


# ---------------------------------------------------------------
# mae_decomposition
# ---------------------------------------------------------------
def test_mae_decomposition_splits_exactly():
    """contrib ของสองกลุ่มรวมกันต้องเท่ากับส่วนต่าง MAE รวมเป๊ะ"""
    rng = np.random.default_rng(5)
    y_true = rng.normal(0, 0.01, 200)
    y_true[::5] = 0.0                               # 20% วันนิ่ง
    y_pred = rng.normal(0, 0.003, 200)
    d = mae_decomposition(y_true, y_pred)
    assert np.isclose(d["frac_zero_move"], 0.2)
    assert d["n_zero_days"] + d["n_moved_days"] == d["n"]
    assert np.isclose(d["contrib_zero_days"] + d["contrib_moved_days"], d["mae_diff"])
    assert np.isclose(d["mae_diff"], d["mae_model"] - d["mae_naive"])
    # Naive มี error = 0 เป๊ะทุกวันนิ่ง -> โมเดลเสียเปรียบบนกลุ่มนั้นแน่นอน
    assert d["mae_naive_zero_days"] == 0.0
    assert d["contrib_zero_days"] > 0


def test_mae_decomposition_matches_naive_baseline():
    """ต้องให้ผลเดียวกับ baselines.baseline_naive_return ตัวจริง"""
    from baselines import baseline_naive_return
    rng = np.random.default_rng(6)
    y_true = rng.normal(0, 0.01, 50)
    y_true[::4] = 0.0                  # ต้องมีวันนิ่ง ไม่งั้นค่ากลุ่ม zero เป็น nan ทั้งแถว
    y_pred = rng.normal(0, 0.01, 50)
    a = mae_decomposition(y_true, y_pred)
    b = mae_decomposition(y_true, y_pred, baseline_naive_return(y_true))
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k], k       # ไม่มี nan เพราะใส่วันนิ่งไว้แล้ว
    assert not any(isinstance(v, float) and np.isnan(v) for v in a.values())


def test_mae_decomposition_model_equals_naive():
    rng = np.random.default_rng(7)
    y_true = rng.normal(0, 0.01, 40)
    d = mae_decomposition(y_true, np.zeros(40))
    assert d["mae_diff"] == 0.0 and np.isclose(d["relMAE"], 1.0)
    assert d["contrib_zero_days"] == 0.0 and d["contrib_moved_days"] == 0.0


# ---------------------------------------------------------------
# full_report
# ---------------------------------------------------------------
def test_full_report_has_all_keys_and_agrees_with_parts():
    rng = np.random.default_rng(8)
    y_true = rng.normal(0, 0.01, 120)
    y_true[::7] = 0.0
    y_pred = rng.normal(0, 0.002, 120)
    r = full_report(y_true, y_pred)
    for k in ("relMAE", "ic_spearman", "contrib_zero_days",
              "dm_abs_dm_stat_hln", "dm_abs_p_one_sided_model_better",
              "dm_sq_dm_stat_hln", "dm_sq_p_two_sided"):
        assert k in r, k
    assert np.isclose(r["relMAE"], mae_decomposition(y_true, y_pred)["relMAE"])
    dm = diebold_mariano(y_true - y_pred, y_true, loss="abs")   # e_naive = y_true - 0
    assert np.isclose(r["dm_abs_dm_stat_hln"], dm["dm_stat_hln"])


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
