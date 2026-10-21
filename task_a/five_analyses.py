"""
five_analyses.py — งาน A และ A2: การวิเคราะห์เพิ่ม 5 ข้อ
=======================================================
  [1] roc        ROC + AUC จาก walk-forward OOF แยกหุ้น/ช่วงราคา (ทั้ง P(odd) และ P(flip))
  [2] threshold  ไล่ threshold 0.30-0.70 บน OOF เลือกค่าบน OOF แล้วค่อยดูผลบน holdout
  [3] label      เทียบ label 3 แบบ: ปัดตามกฎอาจารย์ (round) / floor / x2 (นับหน่วย 0.5 บาท)
  [4] finetune   (เฉพาะ A) pretrain 2016-2023 -> fine-tune ด้วยปี 2024 -> ทดสอบ 2025-2026
  [5] optuna     ค้น model/hyperparameter/feature group บน dev pool เท่านั้น แล้ววัด holdout ครั้งเดียว

ข้อมูล: ราคารายวันจาก task_b/raw_data/ (verified_daily.py) ไม่ใช่ data_cache/ ของ Yahoo ที่มีแถวเสียปี 2024
        A2 ใช้แท่ง 1h ของ task_a2/data/ เหมือน main_a2.py
การแบ่ง: เหมือน main.py / main_a2.py (A: dev pool 85% แรก, A2: 80% แรก) holdout = ส่วนที่เหลือ
         ข้อ [1] [3] ใช้แค่ dev pool · ข้อ [2] [5] เลือกบน dev pool แล้วดู holdout ทีหลัง

รัน:  cd task_a ; python five_analyses.py                 (ทุกข้อ)
      python five_analyses.py --only roc,threshold --trials 60
ผล:   results/five_analyses/*.csv และ results/five_analyses/figures/*.png
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "task_a2"))

from config import (TICKERS, START_DATE, END_DATE, TRAIN_RATIO, VAL_RATIO,  # noqa: E402
                    ANN_PARAMS, LGBM_PARAMS, LOGREG_PARAMS, RANDOM_STATE)
from features import build_features  # noqa: E402
from rounding import to_int_baht, parity  # noqa: E402
from splits import walk_forward_splits  # noqa: E402
from eval_stats import binom_ci, mcnemar_test  # noqa: E402
import verified_daily  # noqa: E402
import features_a2  # noqa: E402

OUT = HERE / "results" / "five_analyses"
FIG = OUT / "figures"
MODEL_NAMES = ["ANN (MLP)", "LightGBM", "Logistic Regression"]
COLORS = {"ANN (MLP)": "#4c78a8", "LightGBM": "#e45756", "Logistic Regression": "#54a24b"}
PRICE_BINS = [0, 100, 200, 400, np.inf]
PRICE_LABELS = ["<100 (tick 0.25)", "100-200 (tick 0.5)", "200-400 (tick 1)", ">=400 (tick 2)"]
TASK_CFG = {"A": {"n_splits": 5, "min_train": 250}, "A2": {"n_splits": 5, "min_train": 150}}


# =============================================================================
# label 3 แบบ (คืน parity 0=คู่ 1=คี่)
# =============================================================================
def _satang(p):
    return np.rint(np.asarray(p, dtype=float) * 100)


def label_round(p):
    """กฎอาจารย์ที่ใช้จริงใน production (rounding.to_int_baht)"""
    return parity(to_int_baht(pd.Series(np.asarray(p, dtype=float)))).values


def label_floor(p):
    return np.mod(np.floor(_satang(p) / 100), 2)


def label_x2(p):
    """round(close x 2) % 2 = คู่/คี่ของจำนวนหน่วย 0.5 บาท (ปัดครึ่งขึ้น ไม่ใช้ banker's rounding)"""
    return np.mod(np.floor((2 * _satang(p) + 50) / 100), 2)


LABELS = {"round": label_round, "floor": label_floor, "x2": label_x2}


# =============================================================================
# ชุดข้อมูล
# =============================================================================
class Data:
    """X, parity จริง (y), parity ฐาน (base = เมื่อวาน / ณ 16:00), flip = y xor base"""

    def __init__(self, task, ticker, label, X, y, base, price_ref, n_pool):
        self.task, self.ticker, self.label = task, ticker, label
        self.X = X
        self.y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.base = pd.Series(np.asarray(base, dtype=float), index=X.index)
        self.flip = (self.y + self.base) % 2
        self.price_ref = pd.Series(np.asarray(price_ref, dtype=float), index=X.index)
        self.n_pool = n_pool
        self.cfg = TASK_CFG[task]

    def pool(self, s):
        return s.iloc[:self.n_pool]

    def hold(self, s):
        return s.iloc[self.n_pool:]


_DAILY = {}


def load_daily(ticker):
    if ticker not in _DAILY:
        _DAILY[ticker] = verified_daily.load_verified_history(ticker, START_DATE)
    return _DAILY[ticker]


def dataset_A(ticker, label="round"):
    df = load_daily(ticker)
    df = df.loc[df.index <= pd.Timestamp(END_DATE)]   # ช่วงเดียวกับ main.py
    X = build_features(df, verbose=False).replace([np.inf, -np.inf], np.nan)
    close = df["Close"]
    y = pd.Series(LABELS[label](close), index=df.index)
    base = y.shift(1)
    ok = (X.isna().mean(axis=1) < 0.5) & base.notna()
    n = int(ok.sum())
    return Data("A", ticker, label, X[ok], y[ok], base[ok], close.shift(1)[ok],
                int(n * (TRAIN_RATIO + VAL_RATIO)))


def dataset_A2(ticker, label="round"):
    safe = ticker.replace(".", "_")
    df = load_daily(ticker)
    intra = features_a2.build_intraday_features(ticker, safe)
    ctx = features_a2.build_daily_context_features(ticker, df=df)
    close = df["Close"]
    common = intra.index.intersection(close.index).intersection(ctx.index)
    intra, ctx = intra.loc[common], ctx.loc[common]
    X = pd.concat([intra.drop(columns=["price_1600"]), ctx], axis=1)
    X = X.astype(float).replace([np.inf, -np.inf], np.nan)
    p1600 = intra["price_1600"].astype(float)
    n = len(X)
    return Data("A2", ticker, label, X, LABELS[label](close.loc[common]),
                LABELS[label](p1600), p1600, n - int(round(n * 0.20)))


def dataset(task, ticker, label="round"):
    return dataset_A(ticker, label) if task == "A" else dataset_A2(ticker, label)


# =============================================================================
# โมเดล + walk-forward
# =============================================================================
def _scaled(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("model", est)])


def _unscaled(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", est)])


DEFAULT_CTORS = {
    "ANN (MLP)": lambda: _scaled(MLPClassifier(**ANN_PARAMS)),
    "LightGBM": lambda: _unscaled(LGBMClassifier(**LGBM_PARAMS)),
    "Logistic Regression": lambda: _scaled(LogisticRegression(**LOGREG_PARAMS)),
}


def _fit_proba(ctor, X_tr, y_tr, X_te):
    """P(flip) — ถ้า train มีคลาสเดียว (เช่น label x2 ช่วง tick >= 1 บาท flip=0 ตลอด) ทายคลาสนั้น"""
    classes = np.unique(y_tr)
    if len(classes) < 2:
        return np.full(len(X_te), float(classes[0]))
    m = ctor()
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def oof_pflip(d, ctors=DEFAULT_CTORS, cols=None):
    """walk-forward บน dev pool คืน DataFrame P(flip) ของแถวที่เป็น out-of-fold"""
    X = d.pool(d.X if cols is None else d.X[cols])
    flip = d.pool(d.flip)
    out = {n: np.full(len(X), np.nan) for n in ctors}
    for tr, te in walk_forward_splits(X, **d.cfg):
        for n, c in ctors.items():
            out[n][te] = _fit_proba(c, X.iloc[tr], flip.iloc[tr].values, X.iloc[te])
    df = pd.DataFrame(out, index=X.index)
    return df.dropna()


def holdout_pflip(d, ctors=DEFAULT_CTORS, cols=None):
    X = d.X if cols is None else d.X[cols]
    Xp, Xh = d.pool(X), d.hold(X)
    return pd.DataFrame({n: _fit_proba(c, Xp, d.pool(d.flip).values, Xh) for n, c in ctors.items()},
                        index=Xh.index)


def p_odd(pflip, base):
    return np.where(np.asarray(base) == 0, pflip, 1 - np.asarray(pflip))


def parity_pred(pflip, base, thr=0.5):
    return (np.asarray(base) + (np.asarray(pflip) >= thr)) % 2


def acc(pred, y):
    return float(np.mean(np.asarray(pred) == np.asarray(y)))


def price_bucket(p):
    return pd.cut(p, PRICE_BINS, labels=PRICE_LABELS, right=False).astype(str)


def boot_auc_ci(y, s, n_boot=500, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


_CACHE = {}


def cached(task, ticker, label="round"):
    """(Data, OOF P(flip), holdout P(flip)) ของโมเดล default 3 ตัว — ใช้ร่วมกันหลายข้อ"""
    key = (task, ticker, label)
    if key not in _CACHE:
        t0 = time.time()
        d = dataset(task, ticker, label)
        _CACHE[key] = (d, oof_pflip(d), holdout_pflip(d))
        print(f"  [data] {task} {ticker} label={label}: n={len(d.X)} pool={d.n_pool} "
              f"holdout={len(d.X) - d.n_pool} ({time.time() - t0:.0f}s)")
    return _CACHE[key]


COMBOS = [(t, k) for t in ("A", "A2") for k in TICKERS]


# =============================================================================
# [1] ROC + AUC
# =============================================================================
def run_roc():
    print("\n[1] ROC + AUC")
    rows = []
    fig_p, axs_p = plt.subplots(2, 2, figsize=(11, 10))
    fig_f, axs_f = plt.subplots(2, 2, figsize=(11, 10))
    for ax_i, (task, ticker) in enumerate(COMBOS):
        d, oof, _ = cached(task, ticker)
        y = d.y.loc[oof.index].values
        base = d.base.loc[oof.index].values
        flip = d.flip.loc[oof.index].values
        bucket = price_bucket(d.price_ref.loc[oof.index]).values
        scores = {}
        for n in MODEL_NAMES:
            scores[(n, "P(odd) vs parity")] = (y, p_odd(oof[n].values, base))
            scores[(n, "P(flip) vs flip")] = (flip, oof[n].values)
        scores[("Baseline: Persistence", "P(odd) vs parity")] = (y, base)
        for (n, kind), (yt, s) in scores.items():
            for b in ["all"] + PRICE_LABELS:
                m = np.ones(len(yt), bool) if b == "all" else bucket == b
                if m.sum() < 30 or len(np.unique(yt[m])) < 2:
                    continue
                lo, hi = boot_auc_ci(yt[m], s[m])
                rows.append({"task": task, "ticker": ticker, "model": n, "score": kind,
                             "price_range": b, "n": int(m.sum()),
                             "AUC": round(roc_auc_score(yt[m], s[m]), 4),
                             "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
        for axs, kind in ((axs_p, "P(odd) vs parity"), (axs_f, "P(flip) vs flip")):
            ax = axs.flat[ax_i]
            for n in MODEL_NAMES:
                yt, s = scores[(n, kind)]
                fpr, tpr, _ = roc_curve(yt, s)
                ax.plot(fpr, tpr, lw=1.8, color=COLORS[n],
                        label=f"{n} AUC={roc_auc_score(yt, s):.3f}")
            if kind.startswith("P(odd)"):
                yt, s = scores[("Baseline: Persistence", kind)]
                fpr, tpr, _ = roc_curve(yt, s)
                ax.plot(fpr[1], tpr[1], "ks", ms=8,
                        label=f"Persistence AUC={roc_auc_score(yt, s):.3f}")
            ax.plot([0, 1], [0, 1], "k:", lw=1)
            ax.set_title(f"Task {task} - {ticker} (OOF n={len(oof)})")
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.legend(fontsize=8, loc="lower right")
    fig_p.suptitle("ROC of P(odd) vs actual parity (walk-forward OOF, dev pool)")
    fig_f.suptitle("ROC of P(flip) vs actual flip - skill beyond persistence")
    for f, name in ((fig_p, "roc_parity.png"), (fig_f, "roc_flip.png")):
        f.tight_layout(); f.savefig(FIG / name, dpi=150); plt.close(f)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "1_roc_auc.csv", index=False, encoding="utf-8-sig")

    # AUC ของ P(flip) แยกช่วงราคา
    sub = df[(df.score == "P(flip) vs flip") & (df.price_range != "all")]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for ax, (task, ticker) in zip(axs, COMBOS):
        s = sub[(sub.task == task) & (sub.ticker == ticker)]
        buckets = [b for b in PRICE_LABELS if b in set(s.price_range)]
        w = 0.27
        for j, n in enumerate(MODEL_NAMES):
            r = s[s.model == n].set_index("price_range").reindex(buckets)
            x = np.arange(len(buckets)) + (j - 1) * w
            ax.bar(x, r.AUC, w, color=COLORS[n], label=n,
                   yerr=[r.AUC - r.ci_low, r.ci_high - r.AUC], capsize=3)
        ax.axhline(0.5, color="k", ls=":")
        n_lab = [f"{b}\nn={int(s[s.price_range == b].n.iloc[0])}" for b in buckets]
        ax.set_xticks(range(len(buckets))); ax.set_xticklabels(n_lab, fontsize=8)
        ax.set_title(f"Task {task} - {ticker}"); ax.set_ylim(0.3, 0.8)
    axs[0].set_ylabel("AUC of P(flip) (95% bootstrap CI)")
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "auc_by_price.png", dpi=150); plt.close(fig)
    print(df[df.price_range == "all"].to_string(index=False))


# =============================================================================
# [2] Threshold sweep
# =============================================================================
THRESHOLDS = np.round(np.arange(0.30, 0.7001, 0.05), 2)


def _pred_at(score, pflip, base, t):
    if score == "P(odd)":
        return (p_odd(pflip, base) >= t).astype(float)
    return parity_pred(pflip, base, t)


def run_threshold():
    print("\n[2] Threshold sweep")
    rows, choice = [], []
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (task, ticker) in zip(axs.flat, COMBOS):
        d, oof, hold = cached(task, ticker)
        y, base = d.y.loc[oof.index].values, d.base.loc[oof.index].values
        yh, baseh = d.y.loc[hold.index].values, d.base.loc[hold.index].values
        pers_oof, pers_hold = acc(base, y), acc(baseh, yh)
        for n in MODEL_NAMES:
            for score in ("P(odd)", "P(flip)"):
                accs = []
                for t in THRESHOLDS:
                    pr = _pred_at(score, oof[n].values, base, t)
                    accs.append(acc(pr, y))
                    rows.append({"task": task, "ticker": ticker, "model": n, "score": score,
                                 "threshold": t, "oof_accuracy": round(accs[-1], 4),
                                 "pred_odd_rate": round(float(np.mean(pr)), 4),
                                 "persistence_oof": round(pers_oof, 4)})
                accs = np.array(accs)
                # เลือกบน OOF เท่านั้น เสมอกันเลือกค่าที่ใกล้ 0.5 ที่สุด
                best_i = min(np.flatnonzero(accs == accs.max()), key=lambda i: abs(THRESHOLDS[i] - 0.5))
                t_best = THRESHOLDS[best_i]
                h05 = _pred_at(score, hold[n].values, baseh, 0.5)
                hb = _pred_at(score, hold[n].values, baseh, t_best)
                choice.append({"task": task, "ticker": ticker, "model": n, "score": score,
                               "oof_acc_at_0.5": round(accs[THRESHOLDS == 0.5][0], 4),
                               "chosen_threshold": t_best, "oof_acc_at_chosen": round(accs.max(), 4),
                               "persistence_oof": round(pers_oof, 4),
                               "holdout_acc_at_0.5": round(acc(h05, yh), 4),
                               "holdout_acc_at_chosen": round(acc(hb, yh), 4),
                               "persistence_holdout": round(pers_hold, 4), "n_holdout": len(yh)})
                if score == "P(flip)":
                    ax.plot(THRESHOLDS, accs, "o-", color=COLORS[n], label=n)
        ax.axhline(pers_oof, color="k", ls="--", label=f"Persistence {pers_oof:.3f}")
        ax.set_title(f"Task {task} - {ticker} (OOF n={len(y)})")
        ax.set_xlabel("threshold on P(flip)  (predict flip if P >= t)")
        ax.set_ylabel("OOF accuracy"); ax.legend(fontsize=8)
    fig.suptitle("Threshold sweep on walk-forward OOF (dev pool)")
    fig.tight_layout(); fig.savefig(FIG / "threshold_sweep.png", dpi=150); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "2_threshold_sweep.csv", index=False, encoding="utf-8-sig")
    ch = pd.DataFrame(choice)
    ch.to_csv(OUT / "2_threshold_choice.csv", index=False, encoding="utf-8-sig")
    print(ch.to_string(index=False))


# =============================================================================
# [3] label 3 แบบ
# =============================================================================
def run_label():
    print("\n[3] Label: round vs floor vs x2")
    dist, perf = [], []
    for task, ticker in COMBOS:
        for lab in LABELS:
            d, oof, _ = cached(task, ticker, lab)
            bucket = price_bucket(d.price_ref).values
            for b in ["all"] + PRICE_LABELS:
                m = np.ones(len(d.y), bool) if b == "all" else bucket == b
                if m.sum() == 0:
                    continue
                dist.append({"task": task, "ticker": ticker, "label": lab, "price_range": b,
                             "n": int(m.sum()), "pct_odd": round(float(d.y[m].mean()), 4),
                             "flip_rate": round(float(d.flip[m].mean()), 4),
                             "persistence_acc": round(1 - float(d.flip[m].mean()), 4)})
            y, base = d.y.loc[oof.index].values, d.base.loc[oof.index].values
            pers = base
            row = {"task": task, "ticker": ticker, "label": lab, "n_oof": len(y),
                   "flip_rate_oof": round(float(np.mean(d.flip.loc[oof.index])), 4),
                   "persistence": round(acc(pers, y), 4)}
            for n in MODEL_NAMES:
                pr = parity_pred(oof[n].values, base)
                row[n] = round(acc(pr, y), 4)
            best = max(MODEL_NAMES, key=lambda n: row[n])
            mc = mcnemar_test(y, parity_pred(oof[best].values, base), pers)
            row.update({"best_model": best, "best_minus_persistence": round(row[best] - row["persistence"], 4),
                        "mcnemar_p": round(mc["pvalue"], 4)})
            perf.append(row)
    dist, perf = pd.DataFrame(dist), pd.DataFrame(perf)
    dist.to_csv(OUT / "3_label_distribution.csv", index=False, encoding="utf-8-sig")
    perf.to_csv(OUT / "3_label_model_accuracy.csv", index=False, encoding="utf-8-sig")

    fig, axs = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for ax, (task, ticker) in zip(axs, COMBOS):
        p = perf[(perf.task == task) & (perf.ticker == ticker)].set_index("label").loc[list(LABELS)]
        x = np.arange(len(p))
        ax.bar(x - 0.2, p.persistence, 0.4, color="#999", label="Persistence")
        ax.bar(x + 0.2, p[MODEL_NAMES].max(axis=1), 0.4, color="#e45756", label="Best ML model")
        for i, (pe, ml) in enumerate(zip(p.persistence, p[MODEL_NAMES].max(axis=1))):
            ax.text(i - 0.2, pe + 0.01, f"{pe:.3f}", ha="center", fontsize=8)
            ax.text(i + 0.2, ml + 0.01, f"{ml:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(["round (official)", "floor", "x2 (0.5-baht)"])
        ax.set_title(f"Task {task} - {ticker}"); ax.set_ylim(0.4, 1.05)
    axs[0].set_ylabel("OOF accuracy"); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "label_compare.png", dpi=150); plt.close(fig)
    print(perf.to_string(index=False))
    print(dist[dist.price_range != "all"].to_string(index=False))


# =============================================================================
# [4] Pretrain -> fine-tune (เฉพาะ A — A2 มีแท่ง 1h แค่ตั้งแต่ 2023-10 ไม่มีช่วง 2016-2023 ให้ pretrain)
# =============================================================================
PRE_END, FT_END, TEST_START = "2023-12-31", "2024-12-31", "2025-01-01"
FT_EXTRA_TREES = [50, 100, 200]   # 100 = ค่าหลักที่ตั้งไว้ก่อนดูผล อีก 2 ค่าเป็น sensitivity
FT_EXTRA_EPOCHS = 50


def _prep(X_fit, scale):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps).fit(X_fit)


def run_finetune():
    print("\n[4] Pretrain (2016-2023) -> fine-tune (2024) -> test (2025-01 .. 2026-08)")
    rows = []
    for ticker in TICKERS:
        d = dataset_A(ticker)
        idx = d.X.index
        pre, ft = idx <= pd.Timestamp(PRE_END), (idx > pd.Timestamp(PRE_END)) & (idx <= pd.Timestamp(FT_END))
        te = idx >= pd.Timestamp(TEST_START)
        X, f = d.X, d.flip.values
        y_te, base_te = d.y[te].values, d.base[te].values
        pers = base_te
        preds = {}

        # LightGBM: ต่อต้นไม้เพิ่มบน booster ที่ pretrain แล้ว (init_model)
        prep = _prep(X[pre], scale=False)
        m_pre = LGBMClassifier(**LGBM_PARAMS).fit(prep.transform(X[pre]), f[pre])
        preds["LightGBM | pretrain only (2016-2023)"] = m_pre.predict_proba(prep.transform(X[te]))[:, 1]
        for k in FT_EXTRA_TREES:
            m_ft = LGBMClassifier(**{**LGBM_PARAMS, "n_estimators": k}).fit(
                prep.transform(X[ft]), f[ft], init_model=m_pre.booster_)
            preds[f"LightGBM | fine-tune +{k} trees on 2024"] = m_ft.predict_proba(prep.transform(X[te]))[:, 1]
        for name, mask in (("recent only (2024)", ft), ("pooled (2016-2024)", pre | ft)):
            p2 = _prep(X[mask], scale=False)
            m = LGBMClassifier(**LGBM_PARAMS).fit(p2.transform(X[mask]), f[mask])
            preds[f"LightGBM | {name}"] = m.predict_proba(p2.transform(X[te]))[:, 1]

        # ANN: warm_start ต่อจาก weight ที่ pretrain แล้ว (scaler ของช่วง pretrain)
        prep = _prep(X[pre], scale=True)
        mlp = MLPClassifier(**{**ANN_PARAMS, "warm_start": True}).fit(prep.transform(X[pre]), f[pre])
        preds["ANN | pretrain only (2016-2023)"] = mlp.predict_proba(prep.transform(X[te]))[:, 1]
        mlp.set_params(max_iter=FT_EXTRA_EPOCHS)
        mlp.fit(prep.transform(X[ft]), f[ft])
        preds[f"ANN | fine-tune +{FT_EXTRA_EPOCHS} epochs on 2024"] = mlp.predict_proba(prep.transform(X[te]))[:, 1]
        for name, mask in (("recent only (2024)", ft), ("pooled (2016-2024)", pre | ft)):
            p2 = _prep(X[mask], scale=True)
            m = MLPClassifier(**ANN_PARAMS).fit(p2.transform(X[mask]), f[mask])
            preds[f"ANN | {name}"] = m.predict_proba(p2.transform(X[te]))[:, 1]

        rows.append({"ticker": ticker, "strategy": "Baseline: Persistence", "n_test": int(te.sum()),
                     "accuracy": round(acc(pers, y_te), 4)})
        for name, pf in preds.items():
            pr = parity_pred(pf, base_te)
            k = int(np.sum(pr == y_te))
            lo, hi = binom_ci(k, len(y_te))
            mc = mcnemar_test(y_te, pr, pers)
            rows.append({"ticker": ticker, "strategy": name, "n_test": len(y_te),
                         "accuracy": round(k / len(y_te), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                         "vs_persistence": round(k / len(y_te) - acc(pers, y_te), 4),
                         "mcnemar_p_vs_persistence": round(mc["pvalue"], 4),
                         "pred_flip_rate": round(float(np.mean(pf >= 0.5)), 4)})
        print(f"  {ticker}: pretrain n={int(pre.sum())} ({idx[pre][0].date()}..{idx[pre][-1].date()}) "
              f"fine-tune n={int(ft.sum())} test n={int(te.sum())} ({idx[te][0].date()}..{idx[te][-1].date()})")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "4_pretrain_finetune.csv", index=False, encoding="utf-8-sig")

    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    for ax, ticker in zip(axs, TICKERS):
        s = df[(df.ticker == ticker) & (df.strategy != "Baseline: Persistence")].iloc[::-1]
        pers = df[(df.ticker == ticker) & (df.strategy == "Baseline: Persistence")].accuracy.iloc[0]
        col = ["#e45756" if x.startswith("LightGBM") else "#4c78a8" for x in s.strategy]
        ax.barh(s.strategy, s.accuracy, color=col,
                xerr=[s.accuracy - s.ci_low, s.ci_high - s.accuracy], capsize=3)
        ax.axvline(pers, color="k", ls="--", label=f"Persistence {pers:.3f}")
        ax.axvline(0.5, color="#999", ls=":")
        ax.set_xlim(0.35, 0.65); ax.set_title(f"{ticker} - test 2025-01..2026-08 (n={s.n_test.iloc[0]})")
        ax.set_xlabel("accuracy (95% CI)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "pretrain_finetune.png", dpi=150); plt.close(fig)
    print(df.to_string(index=False))


# =============================================================================
# [5] Optuna (HyperOps)
# =============================================================================
A_GROUPS = {
    "state": ["n_mod2", "dec_flag", "tick_regime", "dist_to_boundary"],
    "history": ["dtick_lag1", "dtick_lag2", "dtick_lag3", "dtick_parity_lag1", "dtick_parity_lag2",
                "zero_rate_20"],
    "volatility": ["atr14_ticks", "std20_ticks", "range_ticks_lag1"],
    "volume_calendar": ["vol_ratio", "dow", "is_month_end"],
}
A2_GROUPS = {
    "intraday_state": ["n_mod2_1600", "dec_flag_1600", "tick_regime_1600", "dist_to_boundary_1600"],
    "intraday_move": ["ret_last1h", "ret_last2h", "ret_since_open", "range_ticks_so_far", "vol_so_far",
                      "n_bars_available"],
}


def feature_groups(task):
    g = {f"daily_{k}": [f"{c}_prev" for c in v] for k, v in A_GROUPS.items()}
    return g if task == "A" else {**A2_GROUPS, **g}


def _ctor_from_params(p):
    if p["model"] == "logreg":
        return lambda: _scaled(LogisticRegression(C=p["C"], max_iter=2000, random_state=RANDOM_STATE))
    if p["model"] == "lgbm":
        return lambda: _unscaled(LGBMClassifier(
            num_leaves=p["num_leaves"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
            n_estimators=p["n_estimators"], min_child_samples=p["min_child_samples"],
            reg_lambda=p["reg_lambda"], subsample=p["subsample"], subsample_freq=1,
            colsample_bytree=p["colsample_bytree"], random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))
    hidden = {"8": (8,), "16": (16,), "32": (32,), "16-8": (16, 8)}[p["hidden"]]
    return lambda: _scaled(MLPClassifier(hidden_layer_sizes=hidden, alpha=p["alpha"],
                                         learning_rate_init=p["lr"], max_iter=p["max_iter"],
                                         early_stopping=False, random_state=RANDOM_STATE))


def _suggest(trial, task):
    import optuna
    groups = feature_groups(task)
    use = [g for g in groups if trial.suggest_categorical(f"use_{g}", [True, False])]
    if not use:
        raise optuna.TrialPruned()
    model = trial.suggest_categorical("model", ["logreg", "lgbm", "mlp"])
    if model == "logreg":
        trial.suggest_float("C", 1e-3, 10, log=True)
    elif model == "lgbm":
        trial.suggest_int("num_leaves", 3, 31)
        trial.suggest_int("max_depth", 2, 6)
        trial.suggest_float("learning_rate", 0.005, 0.1, log=True)
        trial.suggest_int("n_estimators", 50, 600, step=50)
        trial.suggest_int("min_child_samples", 20, 300)
        trial.suggest_float("reg_lambda", 1e-2, 100, log=True)
        trial.suggest_float("subsample", 0.5, 1.0)
        trial.suggest_float("colsample_bytree", 0.5, 1.0)
    else:
        trial.suggest_categorical("hidden", ["8", "16", "32", "16-8"])
        trial.suggest_float("alpha", 1e-3, 10, log=True)
        trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        trial.suggest_int("max_iter", 100, 500, step=100)
    cols = [c for g in use for c in groups[g]]
    return _ctor_from_params(trial.params), cols


def run_optuna(n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\n[5] Optuna {n_trials} trials ต่อชุด (เลือกบน dev pool OOF เท่านั้น)")
    summary, trials_all = [], []
    m_holdout = len(COMBOS)   # จำนวนการทดสอบบน holdout -> Bonferroni
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (task, ticker) in zip(axs.flat, COMBOS):
        t0 = time.time()
        d, oof_def, hold_def = cached(task, ticker)
        y_oof_all = d.pool(d.y)

        def objective(trial):
            ctor, cols = _suggest(trial, task)
            oof = oof_pflip(d, {"m": ctor}, cols)
            return acc(parity_pred(oof["m"].values, d.base.loc[oof.index].values),
                       y_oof_all.loc[oof.index].values)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=n_trials)
        done = [t for t in study.trials if t.value is not None]
        y_oof, base_oof = d.y.loc[oof_def.index].values, d.base.loc[oof_def.index].values
        pers_oof = acc(base_oof, y_oof)
        for t in done:
            trials_all.append({"task": task, "ticker": ticker, "trial": t.number,
                               "oof_accuracy": round(t.value, 4), **t.params})

        best = study.best_trial
        ctor, cols = _suggest(optuna.trial.FixedTrial(best.params), task)
        oof_b = oof_pflip(d, {"m": ctor}, cols)
        pred_oof = parity_pred(oof_b["m"].values, base_oof)
        mc_cv = mcnemar_test(y_oof, pred_oof, base_oof)

        hold = holdout_pflip(d, {"m": ctor}, cols)
        yh, bh = d.y.loc[hold.index].values, d.base.loc[hold.index].values
        pred_h = parity_pred(hold["m"].values, bh)
        k = int(np.sum(pred_h == yh))
        lo, hi = binom_ci(k, len(yh))
        mc_h = mcnemar_test(yh, pred_h, bh)
        default_best = max(MODEL_NAMES, key=lambda n: acc(parity_pred(oof_def[n].values, base_oof), y_oof))
        summary.append({
            "task": task, "ticker": ticker, "n_trials": len(done),
            "trials_beating_persistence_oof": int(sum(t.value > pers_oof for t in done)),
            "persistence_oof": round(pers_oof, 4),
            "default_best_model": default_best,
            "default_best_oof": round(acc(parity_pred(oof_def[default_best].values, base_oof), y_oof), 4),
            "best_trial_oof": round(best.value, 4),
            "best_model": best.params["model"],
            "best_groups": "+".join(g for g in feature_groups(task) if best.params.get(f"use_{g}")),
            "oof_mcnemar_p": round(mc_cv["pvalue"], 4),
            "oof_p_bonferroni_x_trials": round(min(1.0, mc_cv["pvalue"] * len(done)), 4),
            "n_holdout": len(yh),
            "persistence_holdout": round(acc(bh, yh), 4),
            "default_best_holdout": round(acc(parity_pred(hold_def[default_best].values, bh), yh), 4),
            "best_trial_holdout": round(k / len(yh), 4),
            "holdout_ci": f"{lo:.3f}-{hi:.3f}",
            "holdout_mcnemar_p": round(mc_h["pvalue"], 4),
            "holdout_p_bonferroni": round(min(1.0, mc_h["pvalue"] * m_holdout), 4),
            "passes_bonferroni": bool(mc_h["pvalue"] * m_holdout < 0.05 and k / len(yh) > acc(bh, yh)),
            "best_params": {k_: v for k_, v in best.params.items() if not k_.startswith("use_")},
        })
        vals = [t.value for t in done]
        col = {"logreg": "#54a24b", "lgbm": "#e45756", "mlp": "#4c78a8"}
        ax.scatter([t.number for t in done], vals, c=[col[t.params["model"]] for t in done], s=18)
        ax.plot([t.number for t in done], np.maximum.accumulate(vals), "k-", lw=1, label="best so far")
        ax.axhline(pers_oof, color="k", ls="--", label=f"Persistence OOF {pers_oof:.3f}")
        ax.set_title(f"Task {task} - {ticker}: holdout best={k / len(yh):.3f} "
                     f"vs persistence {acc(bh, yh):.3f}", fontsize=10)
        ax.set_xlabel("trial (green=LogReg, red=LightGBM, blue=MLP)"); ax.set_ylabel("OOF accuracy")
        ax.legend(fontsize=8)
        print(f"  {task} {ticker}: best OOF={best.value:.4f} vs persistence {pers_oof:.4f} | "
              f"holdout {k / len(yh):.4f} vs {acc(bh, yh):.4f} p={mc_h['pvalue']:.3f} ({time.time() - t0:.0f}s)")
    fig.suptitle(f"Optuna search ({n_trials} trials each) - selected on dev-pool OOF only")
    fig.tight_layout(); fig.savefig(FIG / "optuna_history.png", dpi=150); plt.close(fig)
    pd.DataFrame(trials_all).to_csv(OUT / "5_optuna_trials.csv", index=False, encoding="utf-8-sig")
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "5_optuna_summary.csv", index=False, encoding="utf-8-sig")
    print(s.drop(columns=["best_params"]).T.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="roc,threshold,label,finetune,optuna")
    ap.add_argument("--trials", type=int, default=60)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    steps = args.only.split(",")
    t0 = time.time()
    if "roc" in steps:
        run_roc()
    if "threshold" in steps:
        run_threshold()
    if "label" in steps:
        run_label()
    if "finetune" in steps:
        run_finetune()
    if "optuna" in steps:
        run_optuna(args.trials)
    print(f"\nเสร็จ ({time.time() - t0:.0f}s) ผลอยู่ที่ {OUT}")


if __name__ == "__main__":
    main()
