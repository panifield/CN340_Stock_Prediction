"""
calibration_and_curve.py  — A2 + A5
===================================
สองการวิเคราะห์ที่ใช้ walk-forward ร่วมกัน จึงรวมไว้ไฟล์เดียว

A2 — Calibration
----------------
accuracy บอกแค่ว่า "ทายถูกกี่ครั้ง" แต่ไม่บอกว่าโมเดล *มั่นใจ* แค่ไหน
ตัวชี้วัดที่แข็งกว่าคือ

    LogLoss เทียบกับ log(2) = 0.6931

log(2) คือค่าที่ได้จากการทายความน่าจะเป็น 0.5 ทุกครั้งโดยไม่คิดอะไรเลย
ถ้า LogLoss ของโมเดล > log(2) แปลว่า **โมเดลแย่กว่าการไม่ทำอะไรเลย**
ซึ่งเป็นข้อสรุปที่หนักกว่า "accuracy ประมาณ 50%" มาก
เพราะมันบอกว่าโมเดลไม่ได้แค่ทายไม่ถูก แต่ทายผิดอย่างมั่นใจ

รายงานผลของคุณมีตัวเลขนี้อยู่แล้ว (LogReg บน ADVANC = 1.6644) แต่ยังไม่ได้ใช้

A5 — Learning curve
-------------------
accuracy เทียบกับขนาด training set
ถ้าเส้นแบนอยู่ที่ 50% ไม่ว่าจะให้ข้อมูลเท่าไหร่ = ข้อมูลเพิ่มก็ไม่ช่วย
เป็นหลักฐานเชิงภาพว่าปัญหาอยู่ที่ตัวปัญหา ไม่ใช่ที่ปริมาณข้อมูล

ไม่แตะ test set

วิธีใช้
    python calibration_and_curve.py
    python calibration_and_curve.py --models lr,lgbm --skip-curve
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from data_adapter import load_investing_csv
from tick_utils import dev_cutoff, resolve_data_path
from run_permutation_fixed import build_features, make_model

warnings.filterwarnings("ignore")

LOG2 = float(np.log(2))          # 0.6931 — เกณฑ์อ้างอิงของ binary logloss


# ---------------------------------------------------------------
def walk_forward_indices(n: int, n_folds: int = 5, min_train: int = 250):
    """สูตรเดียวกับ splits.walk_forward_splits() ที่ main.py ใช้"""
    fold_size = (n - min_train) // n_folds
    for i in range(n_folds):
        train_end = min_train + i * fold_size
        pred_end = min(train_end + fold_size, n)
        if pred_end <= train_end:
            break
        yield np.arange(0, train_end), np.arange(train_end, pred_end)


def split_xy(data: pd.DataFrame):
    cols = [c for c in data.columns
            if c not in {"y_parity", "y_flip", "prev_parity"}]
    return (data[cols].to_numpy(float),
            data["y_flip"].to_numpy(int),
            data["prev_parity"].to_numpy(int),
            data["y_parity"].to_numpy(int))


# ---------------------------------------------------------------
# A2 — Calibration
# ---------------------------------------------------------------
def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """
    ECE: เฉลี่ยความต่างระหว่าง "ความมั่นใจ" กับ "ความถูกจริง" ในแต่ละ bin
    ยิ่งใกล้ 0 ยิ่งดี
    """
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(proba, bins[1:-1])
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        total += mask.mean() * abs(proba[mask].mean() - y_true[mask].mean())
    return float(total)


def reliability_bins(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(proba, bins[1:-1])
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() < 5:
            continue
        rows.append({
            "bin_center": (bins[b] + bins[b + 1]) / 2,
            "mean_predicted": float(proba[mask].mean()),
            "observed_freq": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def calibration_report(data: pd.DataFrame, name: str, kinds: list[str],
                       n_folds: int, min_train: int):
    """เก็บ probability แบบ out-of-fold แล้ววัด calibration บน y_flip"""
    X, y_flip, prev_parity, y_parity = split_xy(data)

    rows, reliab = [], {}
    for kind in kinds:
        probs, truths, par_pred, par_true = [], [], [], []

        for tr, pr in walk_forward_indices(len(y_flip), n_folds, min_train):
            if len(np.unique(y_flip[tr])) < 2:
                continue
            model = make_model(kind)
            model.fit(X[tr], y_flip[tr])
            p = model.predict_proba(X[pr])[:, 1]
            probs.append(p)
            truths.append(y_flip[pr])
            par_pred.append((prev_parity[pr] + (p >= 0.5).astype(int)) % 2)
            par_true.append(y_parity[pr])

        p = np.concatenate(probs)
        t = np.concatenate(truths).astype(float)
        pp, pt = np.concatenate(par_pred), np.concatenate(par_true)

        eps = 1e-15
        p_clip = np.clip(p, eps, 1 - eps)
        logloss = float(-np.mean(t * np.log(p_clip)
                                 + (1 - t) * np.log(1 - p_clip)))
        brier = float(np.mean((p - t) ** 2))
        ece = expected_calibration_error(t, p)

        rows.append({
            "ticker": name,
            "model": kind,
            "parity_accuracy": round(float(np.mean(pp == pt)), 4),
            "flip_accuracy": round(float(np.mean((p >= 0.5) == t)), 4),
            "LogLoss": round(logloss, 4),
            "log2_reference": round(LOG2, 4),
            "excess_over_log2": round(logloss - LOG2, 4),
            "worse_than_nothing": "YES" if logloss > LOG2 else "no",
            "Brier": round(brier, 4),
            "brier_reference": 0.25,
            "ECE": round(ece, 4),
            "mean_confidence": round(float(np.mean(np.maximum(p, 1 - p))), 4),
        })
        reliab[kind] = reliability_bins(t, p)

    return pd.DataFrame(rows), reliab


# ---------------------------------------------------------------
# A5 — Learning curve
# ---------------------------------------------------------------
def learning_curve(data: pd.DataFrame, name: str, kinds: list[str],
                   fractions=(0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0),
                   n_folds: int = 5, min_train: int = 250):
    """
    วัด accuracy เมื่อให้ข้อมูล train เพียงบางส่วน

    สำคัญ: ตัดข้อมูลจาก *ท้าย* ของ train prefix (เอาข้อมูลที่ใกล้ที่สุด)
    ไม่ใช่สุ่มตัด เพราะเป็น time series การเอาช่วงที่ติดกับวันทำนาย
    คือสิ่งที่สมจริงที่สุด
    """
    X, y_flip, prev_parity, y_parity = split_xy(data)
    rows = []

    for kind in kinds:
        for frac in fractions:
            preds, truths, sizes = [], [], []
            for tr, pr in walk_forward_indices(len(y_flip), n_folds, min_train):
                keep = max(60, int(len(tr) * frac))
                tr_sub = tr[-keep:]                  # เอาช่วงที่ใกล้ที่สุด
                if len(np.unique(y_flip[tr_sub])) < 2:
                    continue
                model = make_model(kind)
                model.fit(X[tr_sub], y_flip[tr_sub])
                flip_hat = model.predict(X[pr])
                preds.append((prev_parity[pr] + flip_hat) % 2)
                truths.append(y_parity[pr])
                sizes.append(len(tr_sub))

            if not truths:
                continue
            pred, truth = np.concatenate(preds), np.concatenate(truths)
            acc = float(np.mean(pred == truth))
            half = 1.96 * np.sqrt(0.25 / len(truth))
            rows.append({
                "ticker": name,
                "model": kind,
                "train_fraction": frac,
                "avg_train_size": int(np.mean(sizes)),
                "accuracy": round(acc, 4),
                "ci_low": round(acc - half, 4),
                "ci_high": round(acc + half, 4),
                "n_eval": len(truth),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
def plot_calibration(reliab_by_ticker: dict, outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [(t, k, d) for t, m in reliab_by_ticker.items()
             for k, d in m.items() if len(d)]
    if not items:
        return
    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 4.6),
                             squeeze=False)
    for ax, (ticker, kind, d) in zip(axes[0], items):
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="Perfect calibration")
        ax.plot(d["mean_predicted"], d["observed_freq"], "o-", lw=2,
                color="#4c78a8", label=kind.upper())
        ax.axhline(0.5, color="grey", ls="--", lw=0.8)
        ax.axvline(0.5, color="grey", ls="--", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"{ticker} — {kind.upper()}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "A2_reliability_diagram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


def plot_curve(curve: pd.DataFrame, outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tickers = curve["ticker"].unique()
    fig, axes = plt.subplots(1, len(tickers), figsize=(6 * len(tickers), 4.6),
                             squeeze=False)
    for ax, ticker in zip(axes[0], tickers):
        g = curve[curve.ticker == ticker]
        for kind, gg in g.groupby("model"):
            ax.plot(gg["avg_train_size"], gg["accuracy"], "o-", lw=2,
                    label=kind.upper())
            ax.fill_between(gg["avg_train_size"], gg["ci_low"], gg["ci_high"],
                            alpha=0.15)
        ax.axhline(0.5, color="red", ls="--", lw=1.5, label="Random = 0.50")
        ax.set_xlabel("Training set size (days)")
        ax.set_ylabel("Walk-forward parity accuracy")
        ax.set_title(f"{ticker} — learning curve")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "A5_learning_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


# ---------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*",
                    default=["KBANK_10Y_Cleaned.csv", "ADVANC_10Y_Cleaned.csv"])
    ap.add_argument("--models", default="lr,lgbm")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--min-train", type=int, default=250)
    ap.add_argument("--round-mode", default="gt", choices=["gt", "gte"])
    ap.add_argument("--date-format", default="%m/%d/%Y")
    ap.add_argument("--outdir", default="results/dev")
    ap.add_argument("--skip-curve", action="store_true")
    args = ap.parse_args()

    kinds = [k.strip() for k in args.models.split(",")]
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print("  A2 — Calibration  |  A5 — Learning curve   (dev เท่านั้น)")
    print(f"  เกณฑ์อ้างอิง: LogLoss ของการทาย 0.5 ทุกครั้ง = log(2) = {LOG2:.4f}")
    print("=" * 78)

    calib_all, curve_all, reliab_all = [], [], {}

    for raw in args.files:
        path = resolve_data_path(raw)
        name = os.path.basename(path).split("_")[0]
        frame = load_investing_csv(path, date_format=args.date_format,
                                   verbose=False)
        data = build_features(frame, round_mode=args.round_mode)

        cut = dev_cutoff(len(data))
        dev = data.iloc[:cut]
        assert dev.index[-1] < data.index[cut], "dev ล้ำเข้า test"
        print(f"\n[{name}] dev = {len(dev)} แถว "
              f"({dev.index[0].date()} -> {dev.index[-1].date()})")

        calib, reliab = calibration_report(dev, name, kinds,
                                           args.n_folds, args.min_train)
        calib_all.append(calib)
        reliab_all[name] = reliab

        if not args.skip_curve:
            print(f"[{name}] learning curve ...", end=" ", flush=True)
            curve_all.append(learning_curve(dev, name, kinds,
                                            n_folds=args.n_folds,
                                            min_train=args.min_train))
            print("เสร็จ")

    calib = pd.concat(calib_all, ignore_index=True)
    print(f"\n{'='*78}\n  A2 — Calibration\n{'='*78}")
    print(calib.to_string(index=False))
    calib.to_csv(os.path.join(args.outdir, "A2_calibration.csv"),
                 index=False, encoding="utf-8-sig")

    print(f"\n  การตีความ")
    for r in calib.itertuples():
        if r.worse_than_nothing == "YES":
            print(f"    [{r.ticker}/{r.model}] LogLoss {r.LogLoss:.4f} "
                  f"> log(2) {LOG2:.4f} (เกิน {r.excess_over_log2:+.4f})")
            print(f"      -> แย่กว่าการทายความน่าจะเป็น 0.5 ทุกครั้ง "
                  f"= ทายผิดอย่างมั่นใจ")
        else:
            print(f"    [{r.ticker}/{r.model}] LogLoss {r.LogLoss:.4f} "
                  f"< log(2) เล็กน้อย แต่ยังไม่แปลว่ามี signal")

    plot_calibration(reliab_all, args.outdir)

    if curve_all:
        curve = pd.concat(curve_all, ignore_index=True)
        print(f"\n{'='*78}\n  A5 — Learning curve\n{'='*78}")
        print(curve.to_string(index=False))
        curve.to_csv(os.path.join(args.outdir, "A5_learning_curve.csv"),
                     index=False, encoding="utf-8-sig")

        print(f"\n  การตีความ")
        for (ticker, kind), g in curve.groupby(["ticker", "model"]):
            g = g.sort_values("avg_train_size")
            slope = g["accuracy"].iloc[-1] - g["accuracy"].iloc[0]
            crosses = ((g["ci_low"] <= 0.5) & (g["ci_high"] >= 0.5)).all()
            print(f"    [{ticker}/{kind}] ข้อมูลน้อยสุด -> มากสุด: "
                  f"{g['accuracy'].iloc[0]:.4f} -> "
                  f"{g['accuracy'].iloc[-1]:.4f} ({slope:+.4f})")
            if crosses:
                print(f"      -> ทุกจุดคร่อม 0.50 = เพิ่มข้อมูลไม่ช่วย")
        plot_curve(curve, args.outdir)

    print(f"\n  [csv] {args.outdir}/")
    print("  *** ยังไม่มีตัวเลขไหนมาจาก test set ***")
