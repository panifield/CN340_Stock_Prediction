"""
supplementary_analysis.py — งาน A (คู่/คี่)
===========================================
รวม 4 การวิเคราะห์เสริมที่อาจารย์ขอ ([1],[2],[3],[4] ตามที่คุยกันไว้)
ทั้งหมดใช้ walk-forward out-of-fold (OOF) ของ pipeline หลัก (main.py)
เป็นฐาน — **ไม่แตะ test set เลยสักบรรทัดเดียว**

  [1] Subperiod power analysis (ตั้งแต่ SUBPERIOD_START ใน config.py)
      *** ช่วงนี้ทับกับ test เดิม (2025-02-21 -> 2026-08-28) ที่เห็นผล
      ไปแล้ว — เป็นการวิเคราะห์เสริม/robustness check เท่านั้น
      ห้ามอ้างเป็นการทดสอบใหม่ที่ยังไม่เห็นผล ***
  [2] Threshold sweep — coverage vs accuracy จาก OOF, เลือก threshold
      จาก OOF เท่านั้น ไม่ใช้ test
  [3] ROC curve จาก OOF + bootstrap 95% CI ของ AUC + เทียบ oracle control
  [4] Breakeven / commission analysis (ระบุสมมติฐานการแปลงเป็นการเทรด
      ชัดเจน เพราะ parity ไม่มี payoff โดยตรง)

รัน: python supplementary_analysis.py
บันทึกกราฟ -> results/figures/, ตาราง -> results/
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore")

from config import (
    TICKERS, RANDOM_STATE, TRAIN_RATIO, VAL_RATIO, SUBPERIOD_START,
    OUTPUT_DIR,
)
from data_loader import load_stock
from features import build_features
from targets import build_targets
from splits import chronological_split
from main import _prepare, _walk_forward_eval
import power_analysis as pa
from experiment_matrix import build_X as build_rep_X, _wf_eval_with_proba

FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_COLORS = {
    "ANN (MLP)": "#4c78a8", "LightGBM": "#e45756",
    "Logistic Regression": "#54a24b", "Oracle (R13)": "#7f7f7f",
}


# =============================================================================
# ค่าตั้งของข้อ [4] — commission / breakeven (แก้ตัวเลขได้ตรงนี้)
# =============================================================================
COMMISSION_RATE = 0.00157      # ค่าคอมมิชชั่นต่อขา (~0.157% ทั่วไปของโบรกไทย)
VAT_RATE = 0.07                # VAT บนค่าคอมมิชชั่น
MARKET_FEE_RATE = 0.00001      # ค่าธรรมเนียมตลาด/clearing (ประมาณ)

# ต้นทุนไปกลับ (ซื้อ+ขาย) เป็นสัดส่วน = 2 ขา x (comm x (1+VAT) + market fee)
ROUNDTRIP_COST = 2 * (COMMISSION_RATE * (1 + VAT_RATE) + MARKET_FEE_RATE)


# =============================================================================
# ตัวช่วยกลาง: ดึง walk-forward OOF ของ 3 โมเดลหลัก ต่อหุ้น (ใช้ร่วมกัน)
# =============================================================================

def _get_oof_for_ticker(ticker, verbose=False):
    df = load_stock(ticker, verbose=verbose)
    X = build_features(df, verbose=verbose)
    targets = build_targets(df, verbose=verbose)

    X_, yflip_, extra_ = _prepare(X, targets["y_flip"],
                                  targets[["prev_parity", "y_parity"]])
    parts = chronological_split(X_, yflip_, verbose=verbose, name=ticker)
    dev_X = pd.concat([parts["train"][0], parts["val"][0]])
    dev_yflip = pd.concat([parts["train"][1], parts["val"][1]])

    y_oof, preds_oof, probas_oof = _walk_forward_eval(
        dev_X, dev_yflip, extra_, verbose=verbose)
    return df, y_oof, preds_oof, probas_oof


def _best_baseline_acc(y_oof, preds_oof):
    base_names = [n for n in preds_oof if n.startswith("Baseline")]
    accs = {n: float(np.mean(preds_oof[n].values == y_oof.values))
           for n in base_names}
    best = max(accs, key=accs.get)
    return best, accs[best]


MODEL_NAMES = ["ANN (MLP)", "LightGBM", "Logistic Regression"]


# =============================================================================
# [1] Subperiod power analysis
# =============================================================================

def run_subperiod_analysis():
    print("\n" + "=" * 78)
    print("  [1] Subperiod power analysis")
    print("=" * 78)

    if SUBPERIOD_START is None:
        print("  SUBPERIOD_START ยังไม่ตั้งค่าใน config.py -> ข้ามขั้นตอนนี้")
        return None

    rows = []
    for ticker in TICKERS:
        df = load_stock(ticker, verbose=False)
        X = build_features(df, verbose=False)
        targets = build_targets(df, verbose=False)
        X_, yflip_, extra_ = _prepare(X, targets["y_flip"],
                                      targets[["prev_parity", "y_parity"]])

        # ขอบ test เดิมของ pipeline หลัก (ใช้เทียบว่า subperiod ทับไหม)
        parts = chronological_split(X_, yflip_, verbose=False, name=ticker)
        test_start = parts["test"][0].index[0]
        test_end = parts["test"][0].index[-1]

        sub_start = pd.Timestamp(SUBPERIOD_START)
        sub_mask = X_.index >= sub_start
        n_sub = int(sub_mask.sum())
        overlaps_test = (sub_start <= test_end)

        print(f"\n  [{ticker}] subperiod ตั้งแต่ {SUBPERIOD_START}: "
              f"n = {n_sub} วัน ({X_.index[sub_mask][0].date()} -> "
              f"{X_.index[sub_mask][-1].date()})")
        print(f"    test เดิมของ pipeline หลัก: "
              f"{test_start.date()} -> {test_end.date()}")
        if overlaps_test:
            print(f"    !!! ทับกับ test เดิมที่เห็นผลไปแล้ว — "
                  f"ผลจากช่วงนี้เป็นแค่ supplementary/robustness check "
                  f"ห้ามอ้างเป็นการทดสอบใหม่ !!!")
        else:
            print(f"    ok: ไม่ทับกับ test เดิม")

        rows.append({"ticker": ticker, "n_days": n_sub,
                     "start": X_.index[sub_mask][0].date(),
                     "end": X_.index[sub_mask][-1].date(),
                     "overlaps_old_test": overlaps_test})

    summary = pd.DataFrame(rows)
    n_avg = int(summary["n_days"].mean())

    print(f"\n  --- Power / MDE ที่ n = {n_avg} วัน (เฉลี่ย 2 หุ้น) ---")
    table = pa.run(n_avg, observed=None, outdir=OUTPUT_DIR)
    pa.plot(n_avg, FIG_DIR)

    summary.to_csv(os.path.join(OUTPUT_DIR, "subperiod_summary.csv"),
                   index=False, encoding="utf-8-sig")
    print(f"\n  [csv] {OUTPUT_DIR}/subperiod_summary.csv")
    print("  *** คำเตือนซ้ำ: subperiod นี้ทับกับ test เดิม ***"
          if summary["overlaps_old_test"].any() else "")
    return summary, table


# =============================================================================
# [2] Threshold sweep (coverage vs accuracy) — จาก OOF เท่านั้น
# =============================================================================

def run_threshold_sweep(oof_cache):
    print("\n" + "=" * 78)
    print("  [2] Threshold sweep (OOF เท่านั้น ไม่แตะ test)")
    print("=" * 78)

    coverages = [1.00, 0.75, 0.50, 0.30, 0.20, 0.10, 0.05]
    rows = []

    fig, axes = plt.subplots(1, len(TICKERS), figsize=(6.5 * len(TICKERS), 4.8),
                             squeeze=False)

    for ax, ticker in zip(axes[0], TICKERS):
        y_oof, preds_oof, probas_oof = oof_cache[ticker]
        base_name, base_acc = _best_baseline_acc(y_oof, preds_oof)

        for name in MODEL_NAMES:
            if name not in probas_oof:
                continue
            proba = probas_oof[name].values
            y_true = y_oof.loc[probas_oof[name].index].values
            pred = (proba >= 0.5).astype(int)
            conf = np.abs(proba - 0.5)

            accs, ns = [], []
            for cov in coverages:
                thr = np.quantile(conf, 1 - cov)
                m = conf >= thr
                accs.append(float(np.mean(pred[m] == y_true[m])))
                ns.append(int(m.sum()))
                rows.append({"ticker": ticker, "model": name, "coverage": cov,
                            "n": int(m.sum()),
                            "accuracy": round(accs[-1], 4)})

            ax.plot([c * 100 for c in coverages], accs, "o-", lw=2,
                    color=MODEL_COLORS.get(name), label=name)

        ax.axhline(base_acc, color="black", ls="--", lw=1.5,
                  label=f"Best baseline ({base_name}={base_acc:.3f})")
        ax.axhline(0.5, color="grey", ls=":", lw=1, label="Random = 0.50")
        ax.invert_xaxis()
        ax.set_xlabel("Coverage (% ของวันที่ทำนาย)")
        ax.set_ylabel("Accuracy (walk-forward OOF)")
        ax.set_title(f"{ticker} — Threshold sweep")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "A_threshold_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "threshold_sweep.csv"), index=False,
              encoding="utf-8-sig")
    print(f"  [csv] {OUTPUT_DIR}/threshold_sweep.csv")
    print(out.to_string(index=False))
    return out


# =============================================================================
# [3] ROC curve + bootstrap CI + oracle control overlay
# =============================================================================

def _bootstrap_auc_ci(y_true, proba, n_boot=1000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y_true[idx], proba[idx]
        if len(np.unique(yb)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yb, pb)
        aucs.append(auc(fpr, tpr))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def _oracle_oof(ticker):
    """OOF probability ของ R13 oracle control (LightGBM) ใช้ทำ ROC เทียบ"""
    df = load_stock(ticker, verbose=False)
    targets = build_targets(df, verbose=False)
    X_raw = build_rep_X("R13_oracle_control", df)
    X_, yflip_, extra_ = _prepare(X_raw, targets["y_flip"],
                                  targets[["prev_parity", "y_parity"]])
    parts = chronological_split(X_, yflip_, verbose=False, name=ticker)
    dev_X = pd.concat([parts["train"][0], parts["val"][0]])
    dev_yflip = pd.concat([parts["train"][1], parts["val"][1]])

    from experiment_matrix import _model_ctors
    ctors = {"LightGBM": _model_ctors()["LightGBM"]}
    y_oof, preds_oof, probas_oof = _wf_eval_with_proba(dev_X, dev_yflip, extra_, ctors)
    return y_oof.values, probas_oof["LightGBM"].values


def run_roc_analysis(oof_cache):
    print("\n" + "=" * 78)
    print("  [3] ROC curve + bootstrap 95% CI + oracle control")
    print("=" * 78)

    rows = []
    fig, axes = plt.subplots(1, len(TICKERS), figsize=(6.5 * len(TICKERS), 5.5),
                             squeeze=False)

    for ax, ticker in zip(axes[0], TICKERS):
        y_oof, preds_oof, probas_oof = oof_cache[ticker]

        for name in MODEL_NAMES:
            if name not in probas_oof:
                continue
            y_true = y_oof.loc[probas_oof[name].index].values
            proba = probas_oof[name].values
            fpr, tpr, _ = roc_curve(y_true, proba)
            auc_val = auc(fpr, tpr)
            lo, hi = _bootstrap_auc_ci(y_true, proba)
            ax.plot(fpr, tpr, lw=2, color=MODEL_COLORS.get(name),
                    label=f"{name} (AUC={auc_val:.3f} [{lo:.3f},{hi:.3f}])")
            rows.append({"ticker": ticker, "model": name, "AUC": round(auc_val, 4),
                        "ci_low": round(lo, 4), "ci_high": round(hi, 4)})

        # oracle control overlay
        y_oracle, proba_oracle = _oracle_oof(ticker)
        fpr_o, tpr_o, _ = roc_curve(y_oracle, proba_oracle)
        auc_o = auc(fpr_o, tpr_o)
        lo_o, hi_o = _bootstrap_auc_ci(y_oracle, proba_oracle)
        ax.plot(fpr_o, tpr_o, lw=2, ls="--", color=MODEL_COLORS["Oracle (R13)"],
                label=f"Oracle R13 (AUC={auc_o:.3f} [{lo_o:.3f},{hi_o:.3f}])")
        rows.append({"ticker": ticker, "model": "Oracle (R13)",
                    "AUC": round(auc_o, 4), "ci_low": round(lo_o, 4),
                    "ci_high": round(hi_o, 4)})

        ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random (AUC=0.50)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{ticker} — ROC (walk-forward OOF)")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "A_roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "roc_auc.csv"), index=False,
              encoding="utf-8-sig")
    print(f"  [csv] {OUTPUT_DIR}/roc_auc.csv")
    print(out.to_string(index=False))
    print("\n  วิธีอ่าน: ถ้า CI ของโมเดลจริงคร่อม 0.50 = ไม่ต่างจากสุ่ม "
          "ถ้า CI ของ Oracle ไม่คร่อม 0.50 (ควรเป็นแบบนี้เสมอ) = "
          "pipeline ตรวจจับ signal ได้จริงเวลามันมี")
    return out


# =============================================================================
# [4] Breakeven / commission
# =============================================================================

def run_breakeven_analysis(oof_cache):
    print("\n" + "=" * 78)
    print("  [4] Breakeven / Commission analysis")
    print("=" * 78)
    print(f"  ต้นทุนไป-กลับ (roundtrip cost) = {ROUNDTRIP_COST*100:.4f}% "
          f"(commission={COMMISSION_RATE*100:.3f}% x2 ขา, "
          f"VAT={VAT_RATE:.0%}, market fee={MARKET_FEE_RATE*100:.4f}% x2 ขา)")
    print("  *** สมมติฐานสำคัญ: parity ไม่มี payoff ทางการเงินโดยตรง ***")
    print("  สูตรนี้ตั้งสมมติฐานว่าแปลงคำทำนายเป็นการเทรดทิศทางแบบ")
    print("  symmetric (เช่น ทายคี่=long, ทายคู่=short) ซึ่งเป็นสมมติฐาน")
    print("  ที่ต้องระบุไว้ชัดเจน ไม่ใช่กฎที่มีอยู่จริงในตลาด")

    rows = []
    for ticker in TICKERS:
        y_oof, preds_oof, probas_oof = oof_cache[ticker]
        # ใช้ raw close จริงคำนวณ E|r| (ผลตอบแทนสัมบูรณ์เฉลี่ยรายวัน)
        df_full = load_stock(ticker, verbose=False)
        ret = df_full["Close"].pct_change().dropna()
        e_abs_r = float(ret.abs().mean())

        c = ROUNDTRIP_COST
        p_star = 0.5 + c / (2 * e_abs_r)

        for name in MODEL_NAMES:
            if name not in preds_oof:
                continue
            acc = float(np.mean(preds_oof[name].values == y_oof.values))
            rows.append({
                "ticker": ticker, "model": name,
                "oof_accuracy": round(acc, 4),
                "E_abs_return": round(e_abs_r, 5),
                "roundtrip_cost_pct": round(c * 100, 4),
                "breakeven_accuracy": round(p_star, 4),
                "beats_breakeven": "YES" if acc > p_star else "no",
            })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "breakeven_commission.csv"),
              index=False, encoding="utf-8-sig")
    print(f"\n{out.to_string(index=False)}")
    print(f"\n  [csv] {OUTPUT_DIR}/breakeven_commission.csv")
    return out


# =============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("  Supplementary analysis — งาน A ([1] subperiod, [2] threshold,")
    print("  [3] ROC, [4] breakeven) — ใช้ walk-forward OOF เท่านั้น")
    print("  ไม่แตะ test set เลย")
    print("=" * 78)

    print("\n  กำลังคำนวณ walk-forward OOF ของ pipeline หลัก (ครั้งเดียว "
          "ใช้ซ้ำทุกข้อ) ...")
    oof_cache = {}
    for ticker in TICKERS:
        _, y_oof, preds_oof, probas_oof = _get_oof_for_ticker(ticker, verbose=False)
        oof_cache[ticker] = (y_oof, preds_oof, probas_oof)
        print(f"    [{ticker}] เสร็จ (n={len(y_oof)})")

    run_subperiod_analysis()
    run_threshold_sweep(oof_cache)
    run_roc_analysis(oof_cache)
    run_breakeven_analysis(oof_cache)

    print("\nเสร็จสิ้น — ไม่แตะ test set เลยตลอดการวิเคราะห์นี้")
