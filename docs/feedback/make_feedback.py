"""
docs/feedback/make_feedback.py — feedback ข้ออาจารย์ (5 สัดส่วนคลาส, 10 breakeven)
================================================================================

ไม่เทรนโมเดลใหม่: อ่านแค่ label จากข้อมูลราคา (โค้ด split เดิม) + ตัวเลขที่บันทึกไว้ใน
task_a/results_floor/, task_a2/results_floor/ และ outcomes.csv (อ่านอย่างเดียว)
ข้อ 8 (calibration) และ 12 (ROC + CI) ทำไม่ได้ เพราะไม่มีไฟล์ P รายแถว (ดู README.md)

    python docs/feedback/make_feedback.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import norm  # noqa: E402

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
sys.path.insert(0, str(ROOT / "task_a"))
sys.path.insert(0, str(ROOT / "task_a2"))

from data_loader import load_stock  # noqa: E402
from features import build_features  # noqa: E402
from targets import build_targets  # noqa: E402
from splits import chronological_split  # noqa: E402
from main import _prepare  # noqa: E402
import features_a2  # noqa: E402

TICKERS = ["KBANK.BK", "ADVANC.BK"]
A_RES = ROOT / "task_a" / "results_floor" / "main"
A2_RES = ROOT / "task_a2" / "results_floor"
FEES = [0.0, 0.005, 0.01, 0.02, 0.05]
A2_TEST_FRACTION = 0.20   # = main_a2.TEST_FRACTION


def safe(t):
    return t.replace(".", "_")


def price_range(p):
    return np.where(p < 100, "<100", np.where(p < 200, "100-200", np.where(p < 400, "200-400", ">=400")))


# --------------------------------------------------------------------------
# label ของแต่ละชุด (split เดียวกับ main.py / main_a2.py)
# --------------------------------------------------------------------------
def labels_a(ticker):
    df = load_stock(ticker, verbose=False)
    X, targets = build_features(df), build_targets(df)
    X_, yflip, extra = _prepare(X, targets["y_flip"], targets[["prev_parity", "y_parity"]])
    parts = chronological_split(X_, yflip, verbose=False)
    pool_idx = parts["train"][0].index.append(parts["val"][0].index)
    test_idx = parts["test"][0].index
    out = {}
    for name, idx in (("dev pool", pool_idx), ("test", test_idx)):
        out[name] = pd.DataFrame({"y_parity": extra.loc[idx, "y_parity"],
                                  "y_flip": yflip.loc[idx]})
    return out


def labels_a2(ticker):
    _, t = features_a2.build_dataset(ticker, safe(ticker))
    n_test = int(round(len(t) * A2_TEST_FRACTION))
    n_pool = len(t) - n_test
    t = t.assign(price_range=price_range(t["price_1600"].values))
    return {"dev pool": t.iloc[:n_pool], "test": t.iloc[n_pool:]}


# --------------------------------------------------------------------------
# 1. สัดส่วนคลาส + balanced accuracy
# --------------------------------------------------------------------------
def class_row(task, ticker, split, rng, d, maj_from):
    n = len(d)
    odd = d["y_parity"].mean()
    maj = 1 if maj_from["y_parity"].mean() >= 0.5 else 0
    return {"task": task, "ticker": ticker, "split": split, "price_range": rng, "n": n,
            "pct_even": 1 - odd, "pct_odd": odd, "pct_flip": d["y_flip"].mean(),
            "train_majority": "คี่" if maj else "คู่",
            "acc_train_majority": (d["y_parity"] == maj).mean(),
            "acc_best_constant": max(odd, 1 - odd)}


def balanced_from_saved(row, n, prev):
    """confusion matrix จาก Accuracy/Recall/Pred_1_Rate ที่บันทึกไว้ + สัดส่วนคี่จริงของชุด"""
    pos = prev * n
    tp = row["Recall"] * pos
    fp = row["Pred_1_Rate"] * n - tp
    tn = (n - pos) - fp
    tnr = tn / (n - pos)
    acc_check = (tp + tn) / n
    return (row["Recall"] + tnr) / 2, acc_check


def section_class():
    rows, ba_rows = [], []
    for t in TICKERS:
        la = labels_a(t)
        for split in ("dev pool", "test"):
            rows.append(class_row("A", t, split, "all", la[split], la["dev pool"]))
        la2 = labels_a2(t)
        for split in ("dev pool", "test"):
            d = la2[split]
            rows.append(class_row("A2", t, split, "all", d, la2["dev pool"]))
            for rng, g in d.groupby("price_range"):
                rows.append(class_row("A2", t, split, rng, g, la2["dev pool"]))

        # balanced accuracy ของทุกแถวในไฟล์ผล (ไม่มี P ก็คำนวณได้)
        for task, split, path, lab in (
                ("A", "dev OOF", A_RES / f"{safe(t)}_taskA_walkforward.csv", None),
                ("A", "test", A_RES / f"{safe(t)}_taskA_parity.csv", la["test"]),
                ("A2", "dev OOF", A2_RES / f"A2_cv_results_{safe(t)}.csv", None),
                ("A2", "test", A2_RES / f"A2_test_results_{safe(t)}.csv", la2["test"])):
            res = pd.read_csv(path, index_col=0)
            if lab is None:
                # OOF: สัดส่วนคี่หาจาก Precision*Pred_1_Rate/Recall ของแถวที่ Recall > 0
                r = res[res["Recall"] > 0].iloc[0]
                prev = r["Precision"] * r["Pred_1_Rate"] / r["Recall"]
                n = None
            else:
                prev, n = lab["y_parity"].mean(), len(lab)
            for model, r in res.iterrows():
                nn = n or 1000
                ba, acc_check = balanced_from_saved(r, nn, prev)
                ba_rows.append({"task": task, "ticker": t, "split": split, "model": model,
                                "accuracy": r["Accuracy"], "balanced_accuracy": round(ba, 4),
                                "accuracy_recomputed": round(acc_check, 4),
                                "pct_odd_used": round(prev, 4)})
    cls = pd.DataFrame(rows)
    ba = pd.DataFrame(ba_rows)
    bad = ba[(ba["accuracy"] - ba["accuracy_recomputed"]).abs() > 0.003]
    assert bad.empty, f"confusion matrix ที่ประกอบกลับไม่ตรง accuracy เดิม:\n{bad}"
    cls.round(4).to_csv(OUT / "1_class_balance.csv", index=False, encoding="utf-8-sig")
    ba.drop(columns="accuracy_recomputed").to_csv(
        OUT / "1_balanced_accuracy.csv", index=False, encoding="utf-8-sig")

    # กราฟ: % คี่ ใน dev pool vs test
    sub = cls[cls["price_range"] == "all"]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r.task}\n{r.ticker.replace('.BK', '')}" for r in sub[sub.split == "test"].itertuples()]
    x = np.arange(len(labels))
    for i, (split, c) in enumerate((("dev pool", "#9aa5b1"), ("test", "#3b6ea8"))):
        v = sub[sub.split == split]["pct_odd"].values * 100
        ax.bar(x + (i - 0.5) * 0.38, v, 0.38, label=split, color=c)
    ax.axhline(50, color="k", lw=0.8, ls=":")
    ax.set_xticks(x, labels)
    ax.set_ylabel("% odd (floor label)")
    ax.set_ylim(35, 65)
    ax.set_title("Class balance: share of odd closes, dev pool vs test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "1_class_balance.png", dpi=150)
    plt.close(fig)
    return cls, ba


# --------------------------------------------------------------------------
# 4. Breakeven (สมมติ)
# --------------------------------------------------------------------------
def days_needed(p, p0, alpha=0.05, power=0.80):
    """n ที่ทดสอบ H0: acc = p0 แบบด้านเดียว ได้ power 80% ถ้า acc จริง = p"""
    if p <= p0:
        return np.nan
    za, zb = norm.ppf(1 - alpha), norm.ppf(power)
    return int(np.ceil(((za * np.sqrt(p0 * (1 - p0)) + zb * np.sqrt(p * (1 - p))) / (p - p0)) ** 2))


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def section_breakeven():
    be = pd.DataFrame({"fee_f": FEES, "breakeven_accuracy": [(1 + f) / 2 for f in FEES]})
    be.to_csv(OUT / "4_breakeven_table.csv", index=False, encoding="utf-8-sig")

    sets = []
    for t in TICKERS:
        a_wf = pd.read_csv(A_RES / f"{safe(t)}_taskA_walkforward.csv", index_col=0)
        a_te = pd.read_csv(A_RES / f"{safe(t)}_taskA_parity.csv", index_col=0)
        best = [m for m in a_te.index if not m.startswith("Baseline")][0]
        for m in (best, "Baseline: Persistence"):
            sets.append(("A", t, "dev OOF", m, a_wf.loc[m, "Accuracy"], 1815))
            sets.append(("A", t, "test", m, a_te.loc[m, "Accuracy"], 365))
        a2_cv = pd.read_csv(A2_RES / f"A2_cv_results_{safe(t)}.csv", index_col=0)
        a2_te = pd.read_csv(A2_RES / f"A2_test_results_{safe(t)}.csv", index_col=0)
        for m in a2_cv.index:
            if m in ("Baseline: Majority", "Baseline: Rule"):
                continue
            sets.append(("A2", t, "dev CV", m, a2_cv.loc[m, "Accuracy"], 410))
            if m in a2_te.index:
                sets.append(("A2", t, "test", m, a2_te.loc[m, "Accuracy"], 140))
    dt = pd.read_csv(A2_RES / "dtick" / "dtick_results.csv")
    dt = dt[dt["model"].str.startswith(("Δtick", "Baseline")) & (dt["n"] >= 30)]
    for r in dt.itertuples():
        rng = "" if r.price_range == "all" else f" [{r.price_range.split(' ')[0]}]"
        name = ("H1 / " if r.model.startswith("Baseline") and r.price_range.startswith("100-200")
                else "") + r.model + rng
        sets.append(("A2 Δtick-split", r.ticker, r.split, name, r.accuracy, r.n))

    rows = []
    for task, t, split, m, p, n in sets:
        lo, hi = wilson(p, n)
        for f in FEES:
            p0 = (1 + f) / 2
            rows.append({"task": task, "ticker": t, "split": split, "model": m, "n": n,
                         "accuracy": round(p, 4), "fee_f": f,
                         "profit_per_bet": round(2 * p - 1 - f, 4),
                         "profit_ci_low": round(2 * lo - 1 - f, 4),
                         "profit_ci_high": round(2 * hi - 1 - f, 4),
                         "ci_excludes_zero": bool(2 * lo - 1 - f > 0),
                         "days_needed_power80": days_needed(p, p0)})
    prof = pd.DataFrame(rows)
    prof.to_csv(OUT / "4_profit_by_model.csv", index=False, encoding="utf-8-sig")

    # กราฟ: accuracy + 95% CI ของชุด test/holdout เทียบเส้น breakeven
    s = prof[(prof.fee_f == 0) & prof.split.isin(["test", "holdout"])].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(s) + 1.5))
    y = np.arange(len(s))[::-1]
    lo = (s.profit_ci_low + 1) / 2
    hi = (s.profit_ci_high + 1) / 2
    ax.errorbar(s.accuracy, y, xerr=[s.accuracy - lo, hi - s.accuracy], fmt="o", color="#3b6ea8",
                capsize=3)
    for f, c in zip(FEES, ["k", "#999", "#c88", "#c55", "#a00"]):
        ax.axvline((1 + f) / 2, color=c, ls=":", lw=1, label=f"breakeven f={f:.1%}")
    ax.set_yticks(y, [f"{r.task} {r.ticker.replace('.BK', '')} {r.split}: {r.model} (n={r.n})"
                      for r in s.itertuples()], fontsize=7)
    ax.set_xlabel("accuracy (95% Wilson CI)")
    ax.set_title("Hypothetical bet: test/holdout accuracy vs breakeven", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "4_breakeven_accuracy.png", dpi=150)
    plt.close(fig)
    return be, prof


def section_live():
    """กำไรสะสมสมมติจาก outcomes.csv (อ่านอย่างเดียว) ใช้ label ที่ระบบ live ให้คะแนนจริง"""
    rows, curves = [], {}
    for task, path, base in (("A", ROOT / "task_a/results/outcomes.csv", "Baseline: Persistence"),
                             ("A2", ROOT / "task_a2/results/outcomes.csv",
                              "Baseline: Persistence-1600")):
        d = pd.read_csv(path)
        d = d[d["status"] == "scored"].copy()
        use_floor = d["correct_floor"].notna() if "correct_floor" in d else False
        d["hit"] = np.where(use_floor, d.get("correct_floor"), d["correct"])
        d["base_hit"] = np.where(use_floor, d.get("baseline_correct_floor"), d["baseline_correct"])
        d["target_date"] = pd.to_datetime(d["target_date"])
        ml = d[~d["model"].str.startswith("Baseline")]
        series = {m: g.sort_values("target_date") for m, g in ml.groupby("model")}
        # baseline: นับทุกวัน-หุ้นที่มีผล (ไม่ใช่แค่แถว Baseline ที่ log บางวัน)
        series[base] = (d.drop_duplicates(["ticker", "target_date"])
                        .assign(hit=lambda x: x["base_hit"]).sort_values("target_date"))
        for m, g in series.items():
            hit = g["hit"].astype(float).values
            for f in (0.0, 0.01):
                pnl = np.where(hit == 1, 1.0, -1.0) - f
                rows.append({"task": task, "model": m, "fee_f": f, "n_bets": len(hit),
                             "n_correct": int(hit.sum()), "accuracy": round(hit.mean(), 4),
                             "cum_profit_units": round(pnl.sum(), 2),
                             "profit_per_bet": round(pnl.mean(), 4)})
            daily = pd.Series(np.where(hit == 1, 1.0, -1.0), index=g["target_date"]).groupby(level=0).sum()
            curves[(task, m)] = (daily.index, daily.cumsum().values)   # รวม 2 หุ้นต่อวัน กันเส้นกระตุก
    live = pd.DataFrame(rows)
    live.to_csv(OUT / "4_live_hypothetical_pnl.csv", index=False, encoding="utf-8-sig")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, task in zip(axs, ("A", "A2")):
        for (tk, m), (x, y) in curves.items():
            if tk == task:
                ax.step(x, y, where="post", label=m, lw=2 if m.startswith("Baseline") else 1.3,
                        ls="--" if m.startswith("Baseline") else "-")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(f"Task {task} live: hypothetical cumulative P&L (f = 0)")
        ax.set_ylabel("units (+1 correct / -1 wrong)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "4_live_hypothetical_pnl.png", dpi=150)
    plt.close(fig)
    return live


if __name__ == "__main__":
    os.chdir(ROOT / "task_a")   # data_loader ใช้ path cache แบบ relative
    pd.set_option("display.width", 200)
    cls, ba = section_class()
    print(cls.round(3).to_string())
    print(ba.to_string())
    be, prof = section_breakeven()
    print(prof[prof.fee_f == 0].to_string())
    print(section_live().to_string())
