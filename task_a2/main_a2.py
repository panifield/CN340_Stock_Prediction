"""
main_a2.py — Task A2 (ทำนายคู่/คี่ ณ เวลา 16:00)
==================================================
โครงเดียวกับ task_a/main.py: เทรนบน y_flip (พลิก parity จาก 16:00 ไป
ปิด) reconstruct กลับเป็น parity แล้วเลือกโมเดลด้วย walk-forward CV
บน dev pool เท่านั้น — **ไม่แตะ test (20% ท้ายสุด) จนกว่าจะสั่งยิง**

รัน (แค่ CV ยังไม่แตะ test):
    python main_a2.py --dev

รัน (ยิง test ครั้งเดียว หลังจากรีวิว CV แล้ว):
    python main_a2.py
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_A = os.path.join(HERE, "..", "task_a")
sys.path.insert(0, TASK_A)

from splits import walk_forward_splits  # noqa: E402
from config import ANN_PARAMS, LGBM_PARAMS, LOGREG_PARAMS, RANDOM_STATE  # noqa: E402
from evaluate import (  # noqa: E402
    classification_metrics, results_table, print_table, print_confusion,
)
from eval_stats import verdict, full_report, binom_ci, binom_test_vs  # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import features_a2

TICKERS = {"KBANK.BK": "KBANK_BK", "ADVANC.BK": "ADVANC_BK"}
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TEST_FRACTION = 0.20
N_SPLITS = 5
MIN_TRAIN = 150

BOUNDARY_BUCKETS = [1, 2, 3]  # เศษ: [0,1)->0, [1,2)->1, [2,3)->2, [3,inf)->3


# =============================================================================
def _scaled(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("model", est)])


def _unscaled(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("model", est)])


def get_classifiers():
    return {
        "ANN (MLP)": _scaled(MLPClassifier(**ANN_PARAMS)),
        "LightGBM": _unscaled(LGBMClassifier(**LGBM_PARAMS)),
        "Logistic Regression": _scaled(LogisticRegression(**LOGREG_PARAMS)),
    }


def _reconstruct_parity(parity_1600, flip_pred, flip_proba=None):
    parity_1600 = np.asarray(parity_1600, dtype=float)
    flip_pred = np.asarray(flip_pred, dtype=float)
    parity_pred = (parity_1600 + flip_pred) % 2
    proba1 = None
    if flip_proba is not None:
        flip_proba = np.asarray(flip_proba, dtype=float)
        proba1 = np.where(parity_1600 == 0, flip_proba, 1 - flip_proba)
    return parity_pred, proba1


# =============================================================================
# Baselines: Majority, Persistence-1600, Rule (ไม่ใช่ ML)
# =============================================================================

def baseline_majority(y_train, n_test):
    m = pd.Series(y_train).mode().iloc[0]
    return np.full(n_test, m)


def baseline_persistence_1600(n_test):
    """ทาย parity ปิด = parity ณ 16:00 เสมอ -> flip_hat = 0 เสมอ"""
    return np.zeros(n_test)


def baseline_rule(dist_train, yflip_train, dist_test):
    """
    Rule baseline (ไม่ใช่ ML): bucket ตามระยะห่างจากจุดปัด (tick) แล้วทาย
    ตาม historical flip-rate ของ bucket นั้นจาก train เท่านั้น
    """
    def bucket(d):
        return np.digitize(d, BOUNDARY_BUCKETS)

    b_train = bucket(dist_train)
    b_test = bucket(dist_test)
    rates = {}
    for b in np.unique(b_train):
        rates[b] = float(np.mean(yflip_train[b_train == b]))
    default_rate = float(np.mean(yflip_train))
    pred = np.array([1.0 if rates.get(b, default_rate) > 0.5 else 0.0
                     for b in b_test])
    return pred


# =============================================================================
def make_oracle_features(X, yflip):
    """
    Oracle control: leak คำตอบ (y_flip) ตรงๆ เป็น sanity check ขีดบน
    ต้องได้ ~100% เสมอ (แค่ split เดียวก็พอ) ถ้าไม่ได้ = harness/
    reconstruction มีบั๊ก

    *** เดิมลองลีค daily_close อย่างเดียวแล้วไม่พอ ***
    y_flip = parity(daily_close) XOR parity(price_1600) แต่ X ไม่มี
    price_1600 ตรงๆ (ตั้งใจตัดออกกันโมเดลอาศัยราคาจริงเป็น feature)
    โมเดลเลยขาดครึ่งหนึ่งของสมการ XOR ไปคำนวณเองไม่ได้ -> ได้ผลแย่กว่า
    baseline ด้วยซ้ำ ไม่ใช่ oracle ที่ดี แก้โดย leak คำตอบตรงๆ แทน
    """
    X_oracle = X.copy()
    X_oracle["oracle_leak_yflip_exact"] = yflip.values
    return X_oracle


def make_negative_control(X, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=X.shape), index=X.index,
        columns=[f"noise{i}" for i in range(X.shape[1])])


# =============================================================================
def walk_forward_eval(X_pool, yflip_pool, parity1600_pool, yparity_pool,
                      dist_pool, n_splits=N_SPLITS, min_train=MIN_TRAIN,
                      verbose=True):
    folds = list(walk_forward_splits(X_pool, n_splits=n_splits,
                                     min_train=min_train))
    model_names = list(get_classifiers().keys())
    all_names = model_names + ["Baseline: Majority",
                               "Baseline: Persistence-1600",
                               "Baseline: Rule"]

    oof_true, oof_pred = [], {n: [] for n in all_names}
    oof_proba = {n: [] for n in model_names}

    for fold_i, (tr, te) in enumerate(folds, 1):
        X_tr, X_te = X_pool.iloc[tr], X_pool.iloc[te]
        yflip_tr, yflip_te = yflip_pool.iloc[tr], yflip_pool.iloc[te]
        p1600_te = parity1600_pool.iloc[te].values
        yparity_te = yparity_pool.iloc[te]
        oof_true.append(yparity_te)

        fold_accs = []
        for name, model in get_classifiers().items():
            model.fit(X_tr, yflip_tr)
            flip_pred = model.predict(X_te)
            try:
                flip_proba = model.predict_proba(X_te)[:, 1]
            except Exception:
                flip_proba = None
            parity_pred, parity_proba = _reconstruct_parity(
                p1600_te, flip_pred, flip_proba)
            oof_pred[name].append(pd.Series(parity_pred, index=X_te.index))
            if parity_proba is not None:
                oof_proba[name].append(pd.Series(parity_proba, index=X_te.index))
            fold_accs.append(f"{name}={np.mean(parity_pred == yparity_te.values):.3f}")

        maj_flip = baseline_majority(yflip_tr.values, len(te))
        maj_par, _ = _reconstruct_parity(p1600_te, maj_flip)
        oof_pred["Baseline: Majority"].append(pd.Series(maj_par, index=X_te.index))

        pers_flip = baseline_persistence_1600(len(te))
        pers_par, _ = _reconstruct_parity(p1600_te, pers_flip)
        oof_pred["Baseline: Persistence-1600"].append(
            pd.Series(pers_par, index=X_te.index))

        dist_tr = dist_pool.iloc[tr].values
        dist_te = dist_pool.iloc[te].values
        rule_flip = baseline_rule(dist_tr, yflip_tr.values, dist_te)
        rule_par, _ = _reconstruct_parity(p1600_te, rule_flip)
        oof_pred["Baseline: Rule"].append(pd.Series(rule_par, index=X_te.index))

        if verbose:
            print(f"    fold {fold_i}/{len(folds)} "
                  f"({X_te.index[0].date()}~{X_te.index[-1].date()}, "
                  f"n={len(te)}): " + ", ".join(fold_accs))

    y_oof = pd.concat(oof_true)
    preds_oof = {n: pd.concat(v) for n, v in oof_pred.items()}
    probas_oof = {n: pd.concat(v) for n, v in oof_proba.items() if v}
    return y_oof, preds_oof, probas_oof


# =============================================================================
def run_ticker(ticker, safe, dev=True):
    print("\n" + "#" * 78)
    print(f"#  Task A2 — {ticker}")
    print("#" * 78)

    X, targets = features_a2.build_dataset(ticker, safe)
    n = len(X)
    n_test = int(round(n * TEST_FRACTION))
    n_pool = n - n_test

    X_pool, X_test = X.iloc[:n_pool], X.iloc[n_pool:]
    t_pool, t_test = targets.iloc[:n_pool], targets.iloc[n_pool:]

    print(f"  ทั้งหมด n={n}  dev_pool={n_pool} "
          f"({X_pool.index[0].date()} -> {X_pool.index[-1].date()})  "
          f"test={n_test} ({X_test.index[0].date()} -> {X_test.index[-1].date()})")

    split_info = {
        "ticker": ticker, "n_total": n, "n_pool": n_pool, "n_test": n_test,
        "pool_start": X_pool.index[0].date(), "pool_end": X_pool.index[-1].date(),
        "test_start": X_test.index[0].date(), "test_end": X_test.index[-1].date(),
    }

    yflip_pool = pd.Series(t_pool["y_flip"].values, index=X_pool.index)
    parity1600_pool = pd.Series(t_pool["parity_1600"].values, index=X_pool.index)
    yparity_pool = pd.Series(t_pool["y_parity"].values, index=X_pool.index)
    dist_pool = X_pool["dist_to_boundary_1600"]

    print("\n  --- Walk-forward CV บน dev pool (ไม่แตะ test) ---")
    y_oof, preds_oof, probas_oof = walk_forward_eval(
        X_pool, yflip_pool, parity1600_pool, yparity_pool, dist_pool)

    # ---------- controls: oracle + negative (ใน CV เดียวกัน) ----------
    print("\n  --- Controls (oracle + negative) ---")
    X_oracle = make_oracle_features(X_pool, yflip_pool)
    X_neg = make_negative_control(X_pool)

    ctrl_names = ["Oracle (leak y_flip)", "Negative control (random)"]
    ctrl_pred, ctrl_proba = {n: [] for n in ctrl_names}, {n: [] for n in ctrl_names}
    ctrl_true = []
    folds = list(walk_forward_splits(X_pool, n_splits=N_SPLITS, min_train=MIN_TRAIN))
    for tr, te in folds:
        p1600_te = parity1600_pool.iloc[te].values
        yparity_te = yparity_pool.iloc[te]
        ctrl_true.append(yparity_te)

        for name, Xc in [("Oracle (leak y_flip)", X_oracle),
                         ("Negative control (random)", X_neg)]:
            m = _unscaled(LGBMClassifier(**LGBM_PARAMS))
            m.fit(Xc.iloc[tr], yflip_pool.iloc[tr])
            flip_pred = m.predict(Xc.iloc[te])
            try:
                flip_proba = m.predict_proba(Xc.iloc[te])[:, 1]
            except Exception:
                flip_proba = None
            par_pred, par_proba = _reconstruct_parity(p1600_te, flip_pred, flip_proba)
            ctrl_pred[name].append(pd.Series(par_pred, index=Xc.iloc[te].index))
            if par_proba is not None:
                ctrl_proba[name].append(pd.Series(par_proba, index=Xc.iloc[te].index))

    y_oof_ctrl = pd.concat(ctrl_true)
    ctrl_pred = {n: pd.concat(v) for n, v in ctrl_pred.items()}
    ctrl_proba = {n: pd.concat(v) for n, v in ctrl_proba.items() if v}

    for name, pred in ctrl_pred.items():
        acc = float(np.mean(pred.values == y_oof_ctrl.loc[pred.index].values))
        print(f"    {name}: accuracy={acc:.4f}")

    # ---------- รายงานหลัก ----------
    all_results = {name: classification_metrics(y_oof, pred, probas_oof.get(name))
                  for name, pred in preds_oof.items()}
    df = results_table(all_results, sort_by="Accuracy", ascending=False)
    print_table(df, f"Task A2 — {ticker}: Walk-forward OOF (n={len(y_oof)})")

    model_names = ["ANN (MLP)", "LightGBM", "Logistic Regression"]
    best = max(model_names, key=lambda nm: all_results[nm]["Accuracy"])
    best_baseline = max(
        (nm for nm in df.index if nm.startswith("Baseline")),
        key=lambda nm: df.loc[nm, "Accuracy"])

    print("\n" + verdict(df.loc[best, "Accuracy"], df.loc[best_baseline, "Accuracy"],
                        n=len(y_oof)))
    print(f"  (เกณฑ์หลักที่ต้องชนะคือ Baseline: Persistence-1600 = "
          f"{df.loc['Baseline: Persistence-1600', 'Accuracy']:.4f})")

    full_report(y_oof, preds_oof, probabilities=probas_oof,
               baseline_name="Baseline: Persistence-1600")

    # ---------- ROC ----------
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"ANN (MLP)": "#4c78a8", "LightGBM": "#e45756",
             "Logistic Regression": "#54a24b"}
    for name in model_names:
        if name not in probas_oof:
            continue
        yt = y_oof.loc[probas_oof[name].index].values
        pr = probas_oof[name].values
        fpr, tpr, _ = roc_curve(yt, pr)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=colors.get(name), label=f"{name} (AUC={auc_val:.3f})")
    if "Oracle (leak y_flip)" in ctrl_proba:
        yt = y_oof_ctrl.loc[ctrl_proba["Oracle (leak y_flip)"].index].values
        pr = ctrl_proba["Oracle (leak y_flip)"].values
        fpr, tpr, _ = roc_curve(yt, pr)
        auc_o = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, ls="--", color="grey", label=f"Oracle (AUC={auc_o:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"Task A2 — {ticker} ROC (walk-forward OOF)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"A2_roc_{safe}.png"), dpi=150)
    plt.close(fig)

    # ---------- threshold sweep ----------
    coverages = [1.0, 0.75, 0.5, 0.3, 0.2, 0.1]
    rows = []
    for name in model_names:
        if name not in probas_oof:
            continue
        proba = probas_oof[name].values
        yt = y_oof.loc[probas_oof[name].index].values
        pred = (proba >= 0.5).astype(int)
        conf = np.abs(proba - 0.5)
        for cov in coverages:
            thr = np.quantile(conf, 1 - cov)
            m = conf >= thr
            rows.append({"ticker": ticker, "model": name, "coverage": cov,
                        "n": int(m.sum()),
                        "accuracy": round(float(np.mean(pred[m] == yt[m])), 4)})
    thr_df = pd.DataFrame(rows)
    thr_df.to_csv(os.path.join(RESULTS_DIR, f"A2_threshold_{safe}.csv"),
                 index=False, encoding="utf-8-sig")

    # ---------- breakeven ----------
    close = t_pool["daily_close"]
    ret = pd.Series(close).pct_change().dropna()
    e_abs_r = float(ret.abs().mean())
    COMMISSION, VAT, FEE = 0.00157, 0.07, 0.00001
    c = 2 * (COMMISSION * (1 + VAT) + FEE)
    p_star = 0.5 + c / (2 * e_abs_r)
    breakeven_rows = []
    for name in model_names:
        acc = all_results[name]["Accuracy"]
        breakeven_rows.append({"ticker": ticker, "model": name,
                              "oof_accuracy": round(acc, 4),
                              "breakeven_accuracy": round(p_star, 4),
                              "beats_breakeven": "YES" if acc > p_star else "no"})
    pd.DataFrame(breakeven_rows).to_csv(
        os.path.join(RESULTS_DIR, f"A2_breakeven_{safe}.csv"),
        index=False, encoding="utf-8-sig")
    print(f"\n  Breakeven accuracy (ต้นทุน {c*100:.3f}%): {p_star:.4f}")

    df.to_csv(os.path.join(RESULTS_DIR, f"A2_cv_results_{safe}.csv"),
             encoding="utf-8-sig")

    # ---------- ส่วนเสริม: แยกผลเฉพาะช่วงหลัง 2025-09-07 ----------
    # อยู่ใน dev_pool ทั้งหมด (pool ถึง 2026-01-26, test เริ่ม 2026-01-27)
    # -> ไม่แตะ test เลย ปลอดภัย เป็นแค่การแบ่งย่อยผล CV ที่มีอยู่แล้ว
    print(f"\n  --- ส่วนเสริม: เฉพาะวันที่ >= 2025-09-07 (ยังอยู่ใน dev_pool) ---")
    sub_start = pd.Timestamp("2025-09-07")
    sub_mask = y_oof.index >= sub_start
    n_sub = int(sub_mask.sum())
    if n_sub < 30:
        print(f"    n={n_sub} วัน น้อยเกินจะสรุปอะไรได้ ข้าม")
    else:
        y_sub = y_oof[sub_mask]
        sub_results = {}
        for name, pred in preds_oof.items():
            p = pred.loc[pred.index.intersection(y_sub.index)]
            yt = y_sub.loc[p.index]
            sub_results[name] = classification_metrics(yt, p, None)
        sub_df = results_table(sub_results, sort_by="Accuracy", ascending=False)
        print_table(sub_df, f"Task A2 — {ticker}: เฉพาะ >= 2025-09-07 (n={n_sub})")

        sub_best = max(model_names, key=lambda nm: sub_results[nm]["Accuracy"])
        print("\n    " + verdict(
            sub_df.loc[sub_best, "Accuracy"],
            sub_df.loc["Baseline: Persistence-1600", "Accuracy"],
            n=n_sub).replace("\n", "\n    "))
        sub_df.to_csv(os.path.join(RESULTS_DIR, f"A2_cv_since20250907_{safe}.csv"),
                     encoding="utf-8-sig")

    if dev:
        print("\n  (โหมด dev — จบแค่นี้ ไม่แตะ test)")
        return {"ticker": ticker, "safe": safe, "split_info": split_info,
                "X": X, "targets": targets, "n_pool": n_pool, "n_test": n_test,
                "best": best, "cv_table": df, "stage": "cv"}

    # =========================================================================
    # ยิงนัดเดียวบน test: refit โมเดลที่ชนะ (จาก walk-forward CV) บน
    # dev_pool ทั้งก้อน แล้วประเมิน test ครั้งเดียว (โครงเดียวกับ Task A)
    # =========================================================================
    print(f"\n  === รัน test ครั้งเดียว: refit {best} บน dev_pool "
          f"({n_pool} แถว) ===")

    yflip_test = pd.Series(t_test["y_flip"].values, index=X_test.index)
    parity1600_test = pd.Series(t_test["parity_1600"].values, index=X_test.index)
    yparity_test = pd.Series(t_test["y_parity"].values, index=X_test.index)
    dist_test = X_test["dist_to_boundary_1600"]

    final_model = get_classifiers()[best]
    final_model.fit(X_pool, yflip_pool)
    flip_pred = final_model.predict(X_test)
    try:
        flip_proba = final_model.predict_proba(X_test)[:, 1]
    except Exception:
        flip_proba = None
    parity_pred, parity_proba = _reconstruct_parity(
        parity1600_test.values, flip_pred, flip_proba)

    test_results = {best: classification_metrics(yparity_test, parity_pred, parity_proba)}

    maj_flip = baseline_majority(yflip_pool.values, n_test)
    maj_par, _ = _reconstruct_parity(parity1600_test.values, maj_flip)
    test_results["Baseline: Majority"] = classification_metrics(yparity_test, maj_par)

    pers_flip = baseline_persistence_1600(n_test)
    pers_par, _ = _reconstruct_parity(parity1600_test.values, pers_flip)
    test_results["Baseline: Persistence-1600"] = classification_metrics(yparity_test, pers_par)

    rule_flip = baseline_rule(dist_pool.values, yflip_pool.values, dist_test.values)
    rule_par, _ = _reconstruct_parity(parity1600_test.values, rule_flip)
    test_results["Baseline: Rule"] = classification_metrics(yparity_test, rule_par)

    test_df = results_table(test_results, sort_by="Accuracy", ascending=False)
    print_table(test_df, f"Task A2 — {ticker}: ผลลัพธ์บน Test Set (ยิงนัดเดียว, n={n_test})")

    best_baseline_test = max(
        (nm for nm in test_df.index if nm.startswith("Baseline")),
        key=lambda nm: test_df.loc[nm, "Accuracy"])
    print("\n" + verdict(test_df.loc[best, "Accuracy"],
                        test_df.loc[best_baseline_test, "Accuracy"], n=n_test))
    print(f"  (เกณฑ์หลักที่ต้องชนะคือ Baseline: Persistence-1600 = "
          f"{test_df.loc['Baseline: Persistence-1600', 'Accuracy']:.4f})")

    all_preds_test = {best: parity_pred, "Baseline: Majority": maj_par,
                      "Baseline: Persistence-1600": pers_par,
                      "Baseline: Rule": rule_par}
    probas_test = {best: parity_proba} if parity_proba is not None else {}
    full_report(yparity_test, all_preds_test, probabilities=probas_test,
               baseline_name="Baseline: Persistence-1600")

    test_df.to_csv(os.path.join(RESULTS_DIR, f"A2_test_results_{safe}.csv"),
                  encoding="utf-8-sig")
    print(f"\n  [csv] {RESULTS_DIR}/A2_test_results_{safe}.csv")

    print_confusion(yparity_test, parity_pred, labels=("คู่", "คี่"),
                    title=f"({best}, refit บน dev_pool)")

    return {"ticker": ticker, "safe": safe, "split_info": split_info,
            "X": X, "targets": targets, "n_pool": n_pool, "n_test": n_test,
            "best": best, "cv_table": df, "test_table": test_df, "stage": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("  Task A2 — ทำนายคู่/คี่ ณ เวลา 16:00")
    if args.dev:
        print("  *** โหมด dev: ไม่แตะ test ***")
    print("=" * 78)

    split_records = []
    for ticker, safe in TICKERS.items():
        res = run_ticker(ticker, safe, dev=args.dev)
        split_records.append(res["split_info"])

    split_df = pd.DataFrame(split_records)
    split_path = os.path.join(RESULTS_DIR, "A2_train_test_split_dates.csv")
    split_df.to_csv(split_path, index=False, encoding="utf-8-sig")
    print(f"\n[บันทึกช่วงวันที่ train/test] {split_path}")
    print(split_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("  จบ CV — ยังไม่แตะ test เลย รอรีวิวผลก่อนยิง test")
    print("=" * 78)


if __name__ == "__main__":
    main()
