"""
docs/feedback/calib_roc.py — feedback ข้อ 8 (calibration) และข้อ 12 (ROC + AUC CI)
==================================================================================

อ่าน P รายแถวจาก docs/feedback/preds/ (สร้างโดย main.py / main_a2.py / dtick_model.py
ที่ตั้ง PREDS_DUMP_DIR) · ขั้นแรกตรวจ accuracy ทุกชุดเทียบไฟล์ผลใน results_floor/
ต่างเกิน 0.3 จุด -> หยุด ไม่ใช้ไฟล์นั้น

    python docs/feedback/calib_roc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

OUT = Path(__file__).resolve().parent
PREDS = OUT / "preds"
ROOT = OUT.parent.parent
A_RES = ROOT / "task_a" / "results_floor" / "main"
A2_RES = ROOT / "task_a2" / "results_floor"
TICKERS = ["KBANK_BK", "ADVANC_BK"]
TOL = 0.003
N_BOOT = 2000
SEED = 42
H1_RANGE = (100, 200)


def load(name):
    return pd.read_csv(PREDS / f"{name}.csv", index_col=0, parse_dates=True)


def models_in(df, prefix):
    return [c.split("|", 1)[1] for c in df.columns if c.startswith(prefix + "|")]


# --------------------------------------------------------------------------
# 0. ตรวจ accuracy เทียบไฟล์ผลเดิม
# --------------------------------------------------------------------------
def check_accuracy():
    rows = []

    def cmp(task, ticker, split, df, saved):
        for m in models_in(df, "pred"):
            if m not in saved.index:
                continue
            acc = float((df[f"pred|{m}"] == df["y_parity"]).mean())
            rows.append({"task": task, "ticker": ticker, "split": split, "model": m,
                         "n": len(df), "acc_rerun": round(acc, 4),
                         "acc_saved": float(saved.loc[m]),
                         "diff_pts": round(100 * (acc - float(saved.loc[m])), 2)})

    for t in TICKERS:
        cmp("A", t, "dev OOF", load(f"A_{t}_oof"),
            pd.read_csv(A_RES / f"{t}_taskA_walkforward.csv", index_col=0)["Accuracy"])
        cmp("A", t, "test", load(f"A_{t}_test"),
            pd.read_csv(A_RES / f"{t}_taskA_parity.csv", index_col=0)["Accuracy"])
        cmp("A2", t, "dev OOF", load(f"A2_{t}_oof"),
            pd.read_csv(A2_RES / f"A2_cv_results_{t}.csv", index_col=0)["Accuracy"])
        cmp("A2", t, "test", load(f"A2_{t}_test"),
            pd.read_csv(A2_RES / f"A2_test_results_{t}.csv", index_col=0)["Accuracy"])
        dt = pd.read_csv(A2_RES / "dtick" / "dtick_results.csv")
        dt = dt[(dt.ticker == t.replace("_", ".")) & (dt.price_range == "all")]
        for split, name in (("dev OOF", "oof"), ("holdout", "holdout")):
            df = load(f"dtick_{t}_{name}")
            df["pred|Baseline: Persistence-1600"] = df["base_parity"]
            saved = dt[dt.split == split].set_index("model")["accuracy"]
            cmp("A2 Δtick", t, split, df, saved)
    chk = pd.DataFrame(rows)
    chk["ok"] = chk["diff_pts"].abs() <= 100 * TOL
    chk.to_csv(OUT / "0_accuracy_check.csv", index=False, encoding="utf-8-sig")
    return chk


# --------------------------------------------------------------------------
# ชุดที่ใช้: (กลุ่ม, ticker, split, df, โมเดล, P(คี่) ของ baseline คงที่)
# baseline คงที่ = สัดส่วนคี่ของชุด train: test ใช้ dev pool ทั้งก้อน
# OOF ใช้สัดส่วนคี่สะสมของแถว OOF ก่อนหน้า (ไม่เห็นอนาคต) เริ่มที่ 0.5
# --------------------------------------------------------------------------
def pool_odd_share():
    cb = pd.read_csv(OUT / "1_class_balance.csv")
    cb = cb[(cb.split == "dev pool") & (cb.price_range == "all")]
    return {(r.task, r.ticker.replace(".", "_")): r.pct_odd for r in cb.itertuples()}


def expanding_share(y):
    s = pd.Series(y).expanding().mean().shift(1).fillna(0.5)
    return s.values


def datasets():
    pool = pool_odd_share()
    out = []
    for t in TICKERS:
        for task, pre in (("A", "A"), ("A2", "A2")):
            oof, test = load(f"{pre}_{t}_oof"), load(f"{pre}_{t}_test")
            out.append((task, t, "dev OOF", oof, models_in(oof, "p_odd"),
                        expanding_share(oof["y_parity"].values)))
            out.append((task, t, "test", test, models_in(test, "p_odd"),
                        np.full(len(test), pool[(task, t)])))
        for split, name in (("dev OOF", "oof"), ("holdout", "holdout")):
            df = load(f"dtick_{t}_{name}")
            if split == "dev OOF":
                base = expanding_share(df["y_parity"].values)
                ms = models_in(df, "p_odd")
            else:
                # pool ของ Δtick = แถว OOF (dev) ทั้งหมด
                base = np.full(len(df), load(f"dtick_{t}_oof")["y_parity"].mean())
                ch = pd.read_csv(A2_RES / "dtick" / "dtick_model_choice.csv")
                ch = ch[ch.ticker == t.replace("_", ".")].iloc[0]
                ms = [ch.selected_dtick_model, ch.selected_flip_model]
            out.append(("A2 Δtick", t, split, df, ms, base))
    return out


# --------------------------------------------------------------------------
# 2. Calibration
# --------------------------------------------------------------------------
def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = [(p[idx == b].mean(), y[idx == b].mean(), int((idx == b).sum()))
            for b in range(bins) if (idx == b).any()]
    return np.array(rows)


def section_calibration(sets):
    rows = []
    groups = [("A", "calibration_A.png"), ("A2", "calibration_A2.png"),
              ("A2 Δtick", "calibration_A2_dtick.png")]
    for task, fname in groups:
        sub = [s for s in sets if s[0] == task]
        fig, axs = plt.subplots(2, 2, figsize=(11, 10))
        for ax, (tk, t, split, df, ms, base) in zip(axs.ravel(), sub):
            y = df["y_parity"].values
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            for m in ms:
                p = df[f"p_odd|{m}"].values
                r = reliability(p, y)
                r = r[r[:, 2] >= 5]          # bin ที่มี < 5 แถวไม่วาด (ECE ยังนับทุก bin)
                ax.plot(r[:, 0], r[:, 1], "o-", ms=3, label=f"{m} (ECE {ece(p, y):.3f})")
                rows.append({"task": tk, "ticker": t, "split": split, "model": m, "n": len(y),
                             "brier": round(float(np.mean((p - y) ** 2)), 4),
                             "ece": round(ece(p, y), 4),
                             "p_odd_min": round(float(p.min()), 3),
                             "p_odd_max": round(float(p.max()), 3)})
            rows.append({"task": tk, "ticker": t, "split": split,
                         "model": "Baseline: P = train odd share", "n": len(y),
                         "brier": round(float(np.mean((base - y) ** 2)), 4),
                         "ece": round(ece(base, y), 4),
                         "p_odd_min": round(float(base.min()), 3),
                         "p_odd_max": round(float(base.max()), 3)})
            ax.set_title(f"{tk} {t.replace('_BK', '')} {split} (n={len(y)})", fontsize=10)
            ax.set_xlabel("mean predicted P(odd)")
            ax.set_ylabel("observed share odd")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7, loc="upper left")
        fig.suptitle(f"Reliability diagram (10 bins, bins with n<5 not drawn) - {tk}")
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=130)
        plt.close(fig)
    cal = pd.DataFrame(rows)
    base = cal[cal.model.str.startswith("Baseline")].set_index(["task", "ticker", "split"])["brier"]
    cal["brier_minus_baseline"] = [round(r.brier - base.loc[(r.task, r.ticker, r.split)], 4)
                                   for r in cal.itertuples()]
    cal.to_csv(OUT / "2_calibration.csv", index=False, encoding="utf-8-sig")
    return cal


# --------------------------------------------------------------------------
# 3. ROC + AUC (bootstrap 2,000 รอบ)
# --------------------------------------------------------------------------
def boot_auc(y, s, rng):
    n = len(y)
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def point(y, pred):
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    return fp / max(np.sum(y == 0), 1), tp / max(np.sum(y == 1), 1)


def section_roc(sets):
    rng = np.random.default_rng(SEED)
    rows = []
    for task, fname in (("A", "roc_A.png"), ("A2", "roc_A2.png"), ("A2 Δtick", "roc_A2_dtick.png")):
        sub = [s for s in sets if s[0] == task]
        fig, axs = plt.subplots(2, 2, figsize=(11, 10))
        for ax, (tk, t, split, df, ms, _) in zip(axs.ravel(), sub):
            y = df["y_parity"].values.astype(int)
            yf = df["y_flip"].values.astype(int)
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            for m in ms:
                s = df[f"p_odd|{m}"].values
                a = roc_auc_score(y, s)
                lo, hi = boot_auc(y, s, rng)
                sf = df[f"p_flip|{m}"].values
                af = roc_auc_score(yf, sf)
                lof, hif = boot_auc(yf, sf, rng)
                fpr, tpr, _ = roc_curve(y, s)
                ax.plot(fpr, tpr, lw=1.4, label=f"{m}: AUC {a:.3f} [{lo:.3f}, {hi:.3f}]")
                rows.append({"task": tk, "ticker": t, "split": split, "model": m, "n": len(y),
                             "auc_p_odd": round(a, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                             "auc_p_flip": round(af, 4), "flip_ci_low": round(lof, 4),
                             "flip_ci_high": round(hif, 4)})
            # persistence = ทาย parity เท่ากับ base (0/1) -> จุดเดียว
            base = df["base_parity"].values.astype(int)
            x0, y0 = point(y, base)
            ax.plot(x0, y0, "s", color="k", ms=8,
                    label=f"Persistence (acc {np.mean(base == y):.3f})")
            rows.append({"task": tk, "ticker": t, "split": split, "model": "Persistence (point)",
                         "n": len(y), "fpr": round(x0, 4), "tpr": round(y0, 4),
                         "accuracy": round(float(np.mean(base == y)), 4)})
            if tk != "A" and "price_1600" in df:
                h1 = (df["price_1600"] >= H1_RANGE[0]) & (df["price_1600"] < H1_RANGE[1])
                if h1.sum() >= 10:
                    x1, y1 = point(y[h1.values], base[h1.values])
                    acc1 = float(np.mean(base[h1.values] == y[h1.values]))
                    ax.plot(x1, y1, "^", color="#d62728", ms=9,
                            label=f"H1 (100-200 only, n={int(h1.sum())}, acc {acc1:.3f})")
                    rows.append({"task": tk, "ticker": t, "split": split,
                                 "model": "H1 (point, 100-200 subset)", "n": int(h1.sum()),
                                 "fpr": round(x1, 4), "tpr": round(y1, 4), "accuracy": round(acc1, 4)})
            ax.set_title(f"{tk} {t.replace('_BK', '')} {split} (n={len(y)})", fontsize=10)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend(fontsize=6.5, loc="lower right")
        fig.suptitle(f"ROC of P(odd) - {tk} (95% CI: bootstrap {N_BOOT})")
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=130)
        plt.close(fig)
    roc = pd.DataFrame(rows)
    roc.to_csv(OUT / "3_roc_auc.csv", index=False, encoding="utf-8-sig")
    return roc


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    chk = check_accuracy()
    print(chk.to_string())
    if not chk["ok"].all():
        print("\n!! accuracy ไม่ตรงเกิน 0.3 จุด -- หยุด ไม่ใช้ไฟล์ P")
        sys.exit(1)
    sets = datasets()
    print(section_calibration(sets).to_string())
    print(section_roc(sets).to_string())
