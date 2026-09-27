"""
experiment_matrix.py — งาน A (คู่/คี่)
=====================================
Stage 1 screening ตามแผน "13 representation x 25 model" — ไม่แตะ test
เลย (ใช้แค่ train+val pool + walk-forward CV เหมือน main.py) รันแยก
จาก main.py โดยสิ้นเชิง ไม่กระทบ production pipeline

*** ตัดออกจาก 13 representation เดิม (ยังไม่มี data/engineering พอ) ***
  R3  Δticks multiclass      - ต้อง pipeline multiclass + aggregate แยก
  R4  Regression -> round    - ต้อง regressor pipeline แยกทั้งชุด
  R10 Cross-sectional        - ต้องข้อมูล SET index / หุ้นอื่น ที่ไม่มี
                                cache ไว้ (มีแค่ KBANK, ADVANC เอง)
  R9  Calendar-only          - ทำแค่บางส่วน (dow/month-end/quarter-end)
                                ไม่มีข้อมูลวัน XD / rebalance SET50

เหลือ 11 representation x โมเดลตัวแทน 4 ตัว (LogReg-L2, LightGBM,
kNN-25, Markov-1) x 2 หุ้น = 88 การทดลอง แก้ multiple comparison ด้วย
Bonferroni + Benjamini-Hochberg FDR ตอนจบ

รัน: python experiment_matrix.py
ผลลัพธ์: stage1_screening_results.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

from config import RANDOM_STATE, TICKERS, LOGREG_PARAMS, LGBM_PARAMS
from data_loader import load_stock
from targets import build_targets
from rounding import to_int_baht, parity as parity_fn, tick_size
from features import build_raw_features as _R0_raw, atr as _atr
from splits import chronological_split, walk_forward_splits
from baselines import get_classification_baselines
from eval_stats import binom_ci, binom_test_vs
from main import _prepare, _reconstruct_parity


# ============================================================
# โมเดลตัวแทน 4 ตัว (LogReg-L2 / LightGBM / kNN-25 / Markov-1)
# ============================================================

def _scaled(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("model", est)])


def _unscaled(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("model", est)])


def _model_ctors():
    """สร้างใหม่ทุกครั้งที่เรียก กัน state ค้างข้าม fold"""
    return {
        "LogReg-L2": lambda: _scaled(LogisticRegression(**LOGREG_PARAMS)),
        "LightGBM": lambda: _unscaled(LGBMClassifier(**LGBM_PARAMS)),
        "kNN-25": lambda: _scaled(KNeighborsClassifier(n_neighbors=25)),
    }


# ============================================================
# Representation builders — คืนค่า "raw" (ยังไม่ shift, ข้อมูลวัน t)
# ยกเว้น R12/R13 ที่ special-case เรื่อง shift เอง
# ============================================================

def _shift1(raw):
    s = raw.shift(1)
    s.columns = [f"{c}_prev" for c in s.columns]
    return s


def _base_price_bits(df):
    close = df["Close"]
    tick = tick_size(close)
    n = np.round(close / tick)
    satang = np.rint(close * 100).astype("int64")
    return close, tick, n, satang


def rep_R0(df):
    """Current — 16 feature หน่วย tick ที่ใช้อยู่ใน pipeline หลัก"""
    return _R0_raw(df)


def rep_R1(df):
    """Minimal state — n_mod2, n_mod4, dec_flag (3 ตัว)"""
    close, tick, n, satang = _base_price_bits(df)
    dec = satang % 100
    f = pd.DataFrame(index=df.index)
    f["n_mod2"] = n % 2
    f["n_mod4"] = n % 4
    f["dec_flag"] = np.select(
        [dec == 0, dec == 25, dec == 50, dec == 75], [0, 1, 2, 3], default=4)
    return f


def rep_R2(df):
    """Parity n-gram — pattern ของ parity 5 วันหลัง one-hot 32 แบบ"""
    close = df["Close"]
    par = parity_fn(to_int_baht(close))
    pattern = pd.Series(0.0, index=df.index)
    for k in range(1, 6):
        pattern = pattern + par.shift(k - 1).fillna(0) * (2 ** (k - 1))
    pattern = pattern.astype(int)
    dummies = pd.get_dummies(
        pd.Categorical(pattern.values, categories=range(32)), prefix="pat")
    dummies.index = df.index
    return dummies.astype(float)


def rep_R5(df):
    """Digit-level — หลักสุดท้าย (0-9) และสตางค์ (0-99) one-hot"""
    close, tick, n, satang = _base_price_bits(df)
    last_digit = satang % 10
    dec = satang % 100
    d1 = pd.get_dummies(pd.Categorical(last_digit.values, categories=range(10)),
                        prefix="digit1")
    d2 = pd.get_dummies(pd.Categorical(dec.values, categories=range(100)),
                        prefix="satang")
    d1.index = df.index
    d2.index = df.index
    return pd.concat([d1, d2], axis=1).astype(float)


def rep_R6(df):
    """Cyclical encoding — sin/cos ของ n mod 2, 4, 8"""
    close, tick, n, satang = _base_price_bits(df)
    f = pd.DataFrame(index=df.index)
    for period, label in [(2, "2"), (4, "4"), (8, "8")]:
        theta = 2 * np.pi * (n % period) / period
        f[f"sin{label}"] = np.sin(theta)
        f[f"cos{label}"] = np.cos(theta)
    return f


def rep_R7_bucket(df):
    """Volatility-regime — คืน (raw R0, atr14_ticks ดิบ) ใช้แบ่ง bucket"""
    raw = _R0_raw(df)
    return raw, raw["atr14_ticks"]


def rep_R8(df):
    """Market-only — return/volume/ATR/range ไม่มี parity feature เลย"""
    close, high, low = df["Close"], df["High"], df["Low"]
    volume = df["Volume"]
    tick = tick_size(close)
    f = pd.DataFrame(index=df.index)
    f["ret1"] = close.pct_change()
    vol_ma = volume.rolling(20).mean()
    f["vol_ratio"] = np.log(volume / vol_ma.replace(0, np.nan))
    f["atr14_ticks"] = _atr(high, low, close) / tick
    f["range_ticks"] = (high - low) / tick
    return f


def rep_R9(df):
    """Calendar-only (partial) — dow/month-end/quarter-end เท่านั้น
    (ไม่มีข้อมูลวัน XD / SET50 rebalance ให้ใช้)"""
    f = pd.DataFrame(index=df.index)
    f["dow"] = df.index.dayofweek
    month, quarter = df.index.to_period("M"), df.index.to_period("Q")
    idx = df.index.to_series()
    f["is_month_end"] = (idx.groupby(month).transform("max") == idx).astype(int)
    f["is_quarter_end"] = (idx.groupby(quarter).transform("max") == idx).astype(int)
    return f


def rep_R11(df):
    """Kitchen sink — รวม R0+R2+R5+R6+R9 (ตัด dow/is_month_end ของ R9
    ออกเพราะซ้ำกับ R0 อยู่แล้ว เหลือแค่ is_quarter_end ที่ R0 ไม่มี)"""
    r9_extra = rep_R9(df)[["is_quarter_end"]]
    return pd.concat([rep_R0(df), rep_R2(df), rep_R5(df), rep_R6(df),
                      r9_extra], axis=1)


def rep_R12(df, seed=RANDOM_STATE):
    """Negative control — feature สุ่มล้วน จำนวนเท่า R0 (16 ตัว)
    ไม่ต้อง shift เพราะเป็น noise ล้วนๆ ไม่มีความหมายเชิงเวลา"""
    rng = np.random.default_rng(seed)
    n = len(df)
    cols = {f"noise{i}_prev": rng.normal(size=n) for i in range(16)}
    return pd.DataFrame(cols, index=df.index)


def rep_R13(df):
    """Oracle control — R0 ปกติ (shift ถูกต้อง) + leak Δticks ของวันนี้
    ตรงๆ 1 คอลัมน์ (ตั้งใจ leak เพื่อ sanity check ขีดบน)"""
    safe = _shift1(_R0_raw(df))
    _, _, n, _ = _base_price_bits(df)
    dtick_today = n.diff()  # ของ "วันนี้" ไม่ shift เลย <- ตั้งใจ leak
    safe["oracle_leak_dtick_today"] = dtick_today.reindex(safe.index)
    return safe


def rep_R14(df):
    """Intraday-horizon variant ("Task A2") — เพิ่ม feature จากราคาเปิด
    ของ "วันนี้" เอง ไม่ใช่ leakage เพราะ Open_t รู้ตอน 10:00 น. ก่อน
    Close_t ที่ 16:30 น. — ถ้าตั้งกรอบว่าทำนายตอน 10:00 น. (ระยะห่าง
    ~6.5 ชม. แทนที่จะเป็น ~24 ชม. จาก close เมื่อวาน) นี่คือข้อมูลจริง
    รวมกับ R0 (shift ถูกต้อง, ข้อมูลถึง t-1) เป็นฐาน"""
    safe = _shift1(_R0_raw(df))
    close, open_ = df["Close"], df["Open"]
    tick_open = tick_size(open_)
    open_n = np.round(open_ / tick_open)
    safe["open_mod2"] = (open_n % 2).reindex(safe.index)
    safe["gap_ticks"] = ((open_ - close.shift(1)) / tick_open).reindex(safe.index)
    return safe


RAW_REPS = {
    "R0_current": rep_R0,
    "R1_minimal_state": rep_R1,
    "R2_parity_ngram": rep_R2,
    "R5_digit_level": rep_R5,
    "R6_cyclical": rep_R6,
    "R8_market_only": rep_R8,
    "R9_calendar_only": rep_R9,
    "R11_kitchen_sink": rep_R11,
}

REPRESENTATIONS = [
    "R0_current", "R1_minimal_state", "R2_parity_ngram", "R5_digit_level",
    "R6_cyclical", "R7_vol_regime_split", "R8_market_only",
    "R9_calendar_only", "R11_kitchen_sink", "R12_negative_control",
    "R13_oracle_control", "R14_intraday_open",
]


def build_X(rep_name, df):
    if rep_name == "R12_negative_control":
        return rep_R12(df)
    if rep_name == "R13_oracle_control":
        return rep_R13(df)
    if rep_name == "R14_intraday_open":
        return rep_R14(df)
    if rep_name == "R7_vol_regime_split":
        raw, atr_series = rep_R7_bucket(df)
        return _shift1(raw), _shift1(atr_series.to_frame("atr"))["atr_prev"]
    return _shift1(RAW_REPS[rep_name](df))


# ============================================================
# Walk-forward pooled OOF evaluator (ทั่วไป + เวอร์ชันแบ่ง bucket ของ R7)
# ============================================================

def _wf_eval(X_pool, yflip_pool, extra_, model_ctors, n_splits=5, min_train=250):
    folds = list(walk_forward_splits(X_pool, n_splits=n_splits, min_train=min_train))
    oof_true = []
    oof_pred = {name: [] for name in model_ctors}
    for train_idx, test_idx in folds:
        X_tr, X_te = X_pool.iloc[train_idx], X_pool.iloc[test_idx]
        yflip_tr = yflip_pool.iloc[train_idx]
        prev_parity_te = extra_.loc[X_te.index, "prev_parity"].fillna(0)
        parity_te = extra_.loc[X_te.index, "y_parity"]
        oof_true.append(parity_te)
        for name, ctor in model_ctors.items():
            m = ctor()
            m.fit(X_tr, yflip_tr)
            flip_pred = m.predict(X_te)
            parity_pred, _ = _reconstruct_parity(prev_parity_te, flip_pred)
            oof_pred[name].append(pd.Series(parity_pred, index=X_te.index))
    y_oof = pd.concat(oof_true)
    preds_oof = {n: pd.concat(v) for n, v in oof_pred.items()}
    return y_oof, preds_oof


def _wf_eval_with_proba(X_pool, yflip_pool, extra_, model_ctors,
                        n_splits=5, min_train=250):
    """เหมือน _wf_eval แต่เก็บ probability ของ parity=1 ไว้ด้วย
    (สำหรับ selective prediction / risk-coverage curve — ข้อ 1.3)"""
    folds = list(walk_forward_splits(X_pool, n_splits=n_splits, min_train=min_train))
    oof_true = []
    oof_pred = {name: [] for name in model_ctors}
    oof_proba = {name: [] for name in model_ctors}
    for train_idx, test_idx in folds:
        X_tr, X_te = X_pool.iloc[train_idx], X_pool.iloc[test_idx]
        yflip_tr = yflip_pool.iloc[train_idx]
        prev_parity_te = extra_.loc[X_te.index, "prev_parity"].fillna(0)
        parity_te = extra_.loc[X_te.index, "y_parity"]
        oof_true.append(parity_te)
        for name, ctor in model_ctors.items():
            m = ctor()
            m.fit(X_tr, yflip_tr)
            flip_pred = m.predict(X_te)
            try:
                flip_proba = m.predict_proba(X_te)[:, 1]
            except Exception:
                flip_proba = None
            parity_pred, parity_proba = _reconstruct_parity(
                prev_parity_te, flip_pred, flip_proba)
            oof_pred[name].append(pd.Series(parity_pred, index=X_te.index))
            if parity_proba is not None:
                oof_proba[name].append(pd.Series(parity_proba, index=X_te.index))
    y_oof = pd.concat(oof_true)
    preds_oof = {n: pd.concat(v) for n, v in oof_pred.items()}
    probas_oof = {n: pd.concat(v) for n, v in oof_proba.items() if v}
    return y_oof, preds_oof, probas_oof


def summarize_significant(df, p_col="p_FDR", acc_col="accuracy", alpha=0.05):
    """
    แยกผล "นัยสำคัญ" ตามทิศทาง — ห้ามใช้ idxmax(accuracy) เฉยๆ เพราะจะ
    หยิบค่าที่สูงสุด "ไม่ว่าจะนัยสำคัญหรือไม่" มาให้ และไม่บอกว่าต่ำกว่า
    0.5 อย่างมีนัยสำคัญก็เป็นเรื่องที่ต้องรายงานเหมือนกัน (แปลว่าโมเดล/
    ฟีเจอร์ตัวนั้นแย่กว่าสุ่มอย่างเป็นระบบ ไม่ใช่แค่ "ไม่ชนะ")
    คืน (below, above) — แถวที่ p ต่ำกว่า alpha แยกเป็นสองกลุ่มตาม
    ทิศทางเทียบกับ 0.50
    """
    sig = df[df[p_col] < alpha]
    below = sig[sig[acc_col] < 0.5]
    above = sig[sig[acc_col] > 0.5]
    return below, above


def risk_coverage(y_true, y_pred, proba, coverages=(1.0, 0.5, 0.3, 0.2, 0.1, 0.05)):
    """
    Selective prediction: ทำนายเฉพาะวันที่มั่นใจ (|proba-0.5| สูงสุด)
    คืนตาราง coverage vs accuracy — ถ้ามี signal จริง เส้นควรไต่ขึ้น
    เมื่อ coverage ลด ถ้าแบนอยู่ที่ ~0.50 ตลอด = probability ไม่มีความหมาย
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    proba = np.asarray(proba)
    conf = np.abs(proba - 0.5)
    rows = []
    for cov in coverages:
        thr = np.quantile(conf, 1 - cov)
        mask = conf >= thr
        n = int(mask.sum())
        acc = float(np.mean(y_pred[mask] == y_true[mask])) if n > 0 else np.nan
        rows.append({"coverage": cov, "n": n, "accuracy": round(acc, 4)})
    return pd.DataFrame(rows)


def _wf_eval_R7(X_pool, yflip_pool, atr_pool, extra_, model_ctors,
               n_splits=5, min_train=250):
    """เหมือน _wf_eval แต่เทรนแยกโมเดลต่อ vol-tercile ต่อ fold
    (tercile fit จาก train ของ fold นั้นเท่านั้น กัน leak ขอบเขต bucket)"""
    folds = list(walk_forward_splits(X_pool, n_splits=n_splits, min_train=min_train))
    oof_true = []
    oof_pred = {name: [] for name in model_ctors}
    for train_idx, test_idx in folds:
        X_tr_all, X_te_all = X_pool.iloc[train_idx], X_pool.iloc[test_idx]
        yflip_tr_all = yflip_pool.iloc[train_idx]
        atr_tr, atr_te = atr_pool.iloc[train_idx], atr_pool.iloc[test_idx]

        edges = np.nanquantile(atr_tr.dropna().values, [1 / 3, 2 / 3])
        fill = atr_tr.median()
        bucket_tr = np.digitize(atr_tr.fillna(fill).values, edges)
        bucket_te = np.digitize(atr_te.fillna(fill).values, edges)

        prev_parity_te = extra_.loc[X_te_all.index, "prev_parity"].fillna(0)
        parity_te = extra_.loc[X_te_all.index, "y_parity"]
        oof_true.append(parity_te)

        for name, ctor in model_ctors.items():
            parity_pred_full = pd.Series(index=X_te_all.index, dtype=float)
            for b in (0, 1, 2):
                te_mask = bucket_te == b
                if te_mask.sum() == 0:
                    continue
                tr_mask = bucket_tr == b
                if tr_mask.sum() < 30:
                    Xtr_b, ytr_b = X_tr_all, yflip_tr_all
                else:
                    Xtr_b, ytr_b = X_tr_all[tr_mask], yflip_tr_all[tr_mask]
                Xte_b = X_te_all[te_mask]
                m = ctor()
                m.fit(Xtr_b, ytr_b)
                flip_pred_b = m.predict(Xte_b)
                prev_parity_b = prev_parity_te[te_mask]
                parity_pred_b, _ = _reconstruct_parity(prev_parity_b, flip_pred_b)
                parity_pred_full.loc[Xte_b.index] = parity_pred_b
            oof_pred[name].append(parity_pred_full)
    y_oof = pd.concat(oof_true)
    preds_oof = {n: pd.concat(v) for n, v in oof_pred.items()}
    return y_oof, preds_oof


def _markov1_oof(X_pool, extra_, n_splits=5, min_train=250):
    """Markov(1) ไม่ใช้ feature เลย ประเมินบน OOF ชุดเดียวกันเพื่อเทียบเป็นธรรม"""
    folds = list(walk_forward_splits(X_pool, n_splits=n_splits, min_train=min_train))
    preds = []
    for train_idx, test_idx in folds:
        idx_tr = X_pool.iloc[train_idx].index
        idx_te = X_pool.iloc[test_idx].index
        parity_tr = extra_.loc[idx_tr, "y_parity"]
        parity_te = extra_.loc[idx_te, "y_parity"]
        prev_te = extra_.loc[idx_te, "prev_parity"].fillna(0)
        base = get_classification_baselines(parity_tr, parity_te, prev_values=prev_te)
        preds.append(pd.Series(base["Baseline: Markov(1)"], index=idx_te))
    return pd.concat(preds)


# ============================================================
# Stage 1 runner
# ============================================================

def run_stage1():
    rows = []
    for ticker in TICKERS:
        df = load_stock(ticker)
        targets = build_targets(df)
        y_flip_full = targets["y_flip"]
        extra_full = targets[["prev_parity", "y_parity"]]

        for rep in REPRESENTATIONS:
            print(f"\n[{ticker}] representation={rep} ...", flush=True)

            if rep == "R7_vol_regime_split":
                X_raw, atr_raw = build_X(rep, df)
                X_, yflip_, extra_ = _prepare(X_raw, y_flip_full, extra_full)
                atr_ = atr_raw.reindex(X_.index)
                parts = chronological_split(X_, yflip_, verbose=False, name=rep)
                dev_X = pd.concat([parts["train"][0], parts["val"][0]])
                dev_yflip = pd.concat([parts["train"][1], parts["val"][1]])
                dev_atr = atr_.loc[dev_X.index]
                y_oof, preds_oof = _wf_eval_R7(dev_X, dev_yflip, dev_atr, extra_,
                                              _model_ctors())
            else:
                X_raw = build_X(rep, df)
                X_, yflip_, extra_ = _prepare(X_raw, y_flip_full, extra_full)
                parts = chronological_split(X_, yflip_, verbose=False, name=rep)
                dev_X = pd.concat([parts["train"][0], parts["val"][0]])
                dev_yflip = pd.concat([parts["train"][1], parts["val"][1]])
                y_oof, preds_oof = _wf_eval(dev_X, dev_yflip, extra_, _model_ctors())

            preds_oof["Markov-1"] = _markov1_oof(dev_X, extra_)

            for model_name, pred in preds_oof.items():
                y_true = y_oof.loc[pred.index]
                k = int(np.sum(pred.values == y_true.values))
                n = len(y_true)
                acc = k / n
                lo, hi = binom_ci(k, n)
                p = binom_test_vs(k, n, 0.5)
                rows.append({
                    "ticker": ticker, "representation": rep, "model": model_name,
                    "n": n, "accuracy": round(acc, 4),
                    "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                    "p_vs_0.50": round(p, 4),
                })
                print(f"    {model_name:12s} acc={acc:.4f}  "
                      f"CI=[{lo:.3f},{hi:.3f}]  p={p:.4f}")

    results_df = pd.DataFrame(rows)

    pvals = results_df["p_vs_0.50"].values
    _, p_bonf, _, _ = multipletests(pvals, alpha=0.05, method="bonferroni")
    rej_fdr, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    results_df["p_bonferroni"] = np.round(p_bonf, 5)
    results_df["p_FDR"] = np.round(p_fdr, 5)
    results_df["significant_FDR"] = rej_fdr

    results_df.to_csv("stage1_screening_results.csv", index=False,
                      encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print(f"  Stage 1 screening: {len(results_df)} การทดลอง "
          f"({len(REPRESENTATIONS)} representation x "
          f"{results_df['model'].nunique()} โมเดล x {len(TICKERS)} หุ้น)")
    print(f"  ผ่าน FDR correction (Benjamini-Hochberg, alpha=0.05): "
          f"{int(results_df['significant_FDR'].sum())} / {len(results_df)} รายการ")
    print("  ไม่แตะ test set เลยตลอด stage นี้")
    print("=" * 90)
    print(results_df.sort_values("accuracy", ascending=False)
         .to_string(index=False))

    return results_df


def run_selective_prediction(reps=("R0_current", "R11_kitchen_sink", "R14_intraday_open"),
                             model_names=("LogReg-L2", "LightGBM")):
    """
    ข้อ 1.3 — selective prediction / risk-coverage curve
    ทำนายเฉพาะวันที่โมเดลมั่นใจสุด ดูว่า accuracy ไต่ขึ้นไหมเมื่อ
    coverage ลด (ถ้ามี signal จริง) หรือแบนอยู่ ~0.50 ตลอด (ไม่มี signal)
    ไม่แตะ test — ใช้ train+val pool + walk-forward OOF เหมือน stage 1
    """
    rows = []
    for ticker in TICKERS:
        df = load_stock(ticker)
        targets = build_targets(df)
        y_flip_full = targets["y_flip"]
        extra_full = targets[["prev_parity", "y_parity"]]

        for rep in reps:
            X_raw = build_X(rep, df)
            X_, yflip_, extra_ = _prepare(X_raw, y_flip_full, extra_full)
            parts = chronological_split(X_, yflip_, verbose=False, name=rep)
            dev_X = pd.concat([parts["train"][0], parts["val"][0]])
            dev_yflip = pd.concat([parts["train"][1], parts["val"][1]])

            ctors = {n: c for n, c in _model_ctors().items() if n in model_names}
            y_oof, preds_oof, probas_oof = _wf_eval_with_proba(
                dev_X, dev_yflip, extra_, ctors)

            for model_name in model_names:
                if model_name not in probas_oof:
                    continue
                y_true = y_oof.loc[preds_oof[model_name].index].values
                y_pred = preds_oof[model_name].values
                proba = probas_oof[model_name].values
                rc = risk_coverage(y_true, y_pred, proba)
                rc["ticker"] = ticker
                rc["representation"] = rep
                rc["model"] = model_name
                rows.append(rc)
                print(f"\n[{ticker}] {rep} / {model_name}")
                print(rc[["coverage", "n", "accuracy"]].to_string(index=False))

    out = pd.concat(rows, ignore_index=True)

    out["p_vs_0.50"] = out.apply(
        lambda r: binom_test_vs(int(round(r["accuracy"] * r["n"])), int(r["n"]), 0.5),
        axis=1)
    _, p_fdr, _, _ = multipletests(out["p_vs_0.50"].values, alpha=0.05, method="fdr_bh")
    out["p_FDR"] = p_fdr

    out.to_csv("selective_prediction_results.csv", index=False,
              encoding="utf-8-sig")

    below, above = summarize_significant(out)
    print("\n" + "=" * 78)
    print(f"  Selective prediction: {len(out)} จุด (coverage x representation "
          f"x โมเดล x หุ้น) หลัง FDR correction")
    print(f"  นัยสำคัญ 'ดีกว่าสุ่ม' (accuracy > 0.50): {len(above)} จุด")
    if len(above):
        print(above[["ticker", "representation", "model", "coverage", "n",
                     "accuracy", "p_FDR"]].to_string(index=False))
    print(f"  นัยสำคัญ 'แย่กว่าสุ่มอย่างเป็นระบบ' (accuracy < 0.50): {len(below)} จุด")
    if len(below):
        print(below[["ticker", "representation", "model", "coverage", "n",
                     "accuracy", "p_FDR"]].to_string(index=False))
    print("=" * 78)

    return out


if __name__ == "__main__":
    run_stage1()
