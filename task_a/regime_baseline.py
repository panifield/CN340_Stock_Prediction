"""
regime_baseline.py  — A4
========================
Baseline ตัวใหม่: Regime-aware majority

แนวคิด
------
จาก dev_report เราพบว่า base rate ของ parity ไม่คงที่ แต่เปลี่ยนตาม
tick regime ชัดเจน (ADVANC: tick=0.50 -> คู่ 46.97% แต่ tick=1.00 -> คู่ 56.11%)

baseline ตัวนี้จึงทายตาม "ฝั่งที่เยอะกว่าภายใน regime นั้น" แทนที่จะใช้
ฝั่งเดียวตลอดทั้งชุดข้อมูล

ทำไมถึงใส่ได้โดยไม่ถือว่าเป็นการ tune
-------------------------------------
มันไม่ใช่โมเดลที่ปรับจนได้ตัวเลขสวย แต่เป็น baseline ที่มาจาก
*ความเข้าใจโครงสร้างตลาด* (tick size ของ SET เป็นกฎที่ประกาศไว้ล่วงหน้า
ไม่ใช่สิ่งที่เราค้นพบจากข้อมูล) และ regime ของวัน t รู้ได้จากราคาวัน t-1
จึงไม่มี leak

และถ้ามันชนะโมเดล ML ทุกตัว (ซึ่งน่าจะเป็นแบบนั้น) ข้อสรุปของรายงาน
จะยิ่งหนักแน่น เพราะแสดงว่า accuracy ที่เกิน 50% มาจากโครงสร้างตลาด
ไม่ใช่จากการทำนาย

การประเมิน
----------
ใช้ walk-forward เหมือน pipeline หลัก: เรียนสัดส่วนจาก train prefix
เท่านั้น แล้วทายช่วงถัดไป ไม่มีการมองอนาคต

ไม่แตะ test set

วิธีใช้
    python regime_baseline.py
    python regime_baseline.py --min-train 250 --n-folds 5
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from data_adapter import load_investing_csv
from tick_utils import (
    dev_cutoff, tick_size, to_int_baht, parity, resolve_data_path,
)


def build_frame(df: pd.DataFrame, round_mode: str = "gt") -> pd.DataFrame:
    """
    เตรียมคอลัมน์ที่ baseline ต้องใช้

    regime_prev = tick regime ของ *เมื่อวาน* — รู้ได้ตอนทำนายวันนี้
    ใช้ตัวนี้ ไม่ใช่ regime ของวันนี้ เพื่อไม่ให้มี leak
    """
    close = df["Close"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["close"] = close
    out["regime"] = tick_size(close)
    out["regime_prev"] = out["regime"].shift(1)
    out["y"] = parity(to_int_baht(close, mode=round_mode))
    out["prev_y"] = out["y"].shift(1)
    return out.dropna()


def walk_forward_indices(n: int, n_folds: int = 5, min_train: int = 250):
    """
    สูตรเดียวกับ splits.walk_forward_splits() ที่ main.py ใช้
        fold_size = (n - min_train) // n_folds
    """
    fold_size = (n - min_train) // n_folds
    for i in range(n_folds):
        train_end = min_train + i * fold_size
        pred_end = min(train_end + fold_size, n)
        if pred_end <= train_end:
            break
        yield np.arange(0, train_end), np.arange(train_end, pred_end)


# ---------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------
def predict_global_majority(y_train, n_pred):
    """ทายฝั่งที่เยอะกว่าในทั้ง train set (baseline เดิมของคุณ)"""
    return np.full(n_pred, float(pd.Series(y_train).mode().iloc[0]))


def predict_regime_majority(y_train, regime_train, regime_pred,
                            fallback: float):
    """
    ทายฝั่งที่เยอะกว่า *ภายใน regime เดียวกัน*

    ถ้า regime ไหนไม่เคยเห็นใน train (เช่นราคาเพิ่งข้ามขอบ 200 บาท
    เป็นครั้งแรก) ให้ถอยไปใช้ global majority
    """
    lookup = {}
    frame = pd.DataFrame({"r": regime_train, "y": y_train})
    for r, g in frame.groupby("r"):
        if len(g) >= 30:                      # ต้องมีข้อมูลพอสมควร
            lookup[r] = float(g["y"].mean() >= 0.5)

    return np.array([lookup.get(r, fallback) for r in regime_pred])


def predict_persistence(prev_y):
    """ทายว่าวันนี้เหมือนเมื่อวาน"""
    return np.asarray(prev_y, dtype=float)


# ---------------------------------------------------------------
def evaluate(data: pd.DataFrame, name: str, n_folds: int, min_train: int):
    y = data["y"].to_numpy(float)
    regime = data["regime_prev"].to_numpy(float)
    prev_y = data["prev_y"].to_numpy(float)

    collected = {k: ([], []) for k in
                 ["Global majority", "Regime-aware majority", "Persistence"]}
    fold_rows = []

    for fold, (tr, pr) in enumerate(
            walk_forward_indices(len(y), n_folds, min_train), start=1):
        fallback = float(pd.Series(y[tr]).mode().iloc[0])

        preds = {
            "Global majority": predict_global_majority(y[tr], len(pr)),
            "Regime-aware majority": predict_regime_majority(
                y[tr], regime[tr], regime[pr], fallback),
            "Persistence": predict_persistence(prev_y[pr]),
        }

        row = {"fold": fold, "n": len(pr),
               "start": data.index[pr[0]].date(),
               "end": data.index[pr[-1]].date(),
               "regimes_seen": sorted(set(regime[pr]))}
        for key, p in preds.items():
            collected[key][0].append(p)
            collected[key][1].append(y[pr])
            row[key] = round(float(np.mean(p == y[pr])), 4)
        fold_rows.append(row)

    results = []
    for key, (ps, ys) in collected.items():
        pred = np.concatenate(ps)
        truth = np.concatenate(ys)
        acc = float(np.mean(pred == truth))
        n = len(truth)
        half = 1.96 * np.sqrt(0.25 / n)
        results.append({
            "ticker": name,
            "baseline": key,
            "accuracy": round(acc, 4),
            "ci_low": round(acc - half, 4),
            "ci_high": round(acc + half, 4),
            "n": n,
            "pred_1_rate": round(float(pred.mean()), 4),
            "beats_50": "YES" if acc - half > 0.50 else "no",
        })

    return pd.DataFrame(results), pd.DataFrame(fold_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*",
                    default=["KBANK_10Y_Cleaned.csv", "ADVANC_10Y_Cleaned.csv"])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--min-train", type=int, default=250)
    ap.add_argument("--round-mode", default="gt", choices=["gt", "gte"])
    ap.add_argument("--date-format", default="%m/%d/%Y")
    ap.add_argument("--outdir", default="results/dev")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 78)
    print("  A4 — Regime-aware majority baseline  (dev เท่านั้น)")
    print("=" * 78)

    all_results, all_folds = [], []

    for raw in args.files:
        path = resolve_data_path(raw)
        name = os.path.basename(path).split("_")[0]
        frame = load_investing_csv(path, date_format=args.date_format,
                                   verbose=False)
        data = build_frame(frame, args.round_mode)

        cut = dev_cutoff(len(data))
        dev = data.iloc[:cut]
        assert dev.index[-1] < data.index[cut], "dev ล้ำเข้า test"

        print(f"\n[{name}] dev = {len(dev)} แถว "
              f"({dev.index[0].date()} -> {dev.index[-1].date()})")
        print(f"         regime ที่พบใน dev: "
              f"{sorted(dev['regime_prev'].unique())}")

        res, folds = evaluate(dev, name, args.n_folds, args.min_train)
        folds.insert(0, "ticker", name)
        all_results.append(res)
        all_folds.append(folds)

        print(f"\n  ผลรายfold:")
        print(folds.drop(columns="regimes_seen").to_string(index=False))
        print(f"\n  ผลรวม (out-of-fold):")
        print(res.to_string(index=False))

    results = pd.concat(all_results, ignore_index=True)
    folds = pd.concat(all_folds, ignore_index=True)

    results.to_csv(os.path.join(args.outdir, "A4_regime_baseline.csv"),
                   index=False, encoding="utf-8-sig")
    folds.drop(columns="regimes_seen").to_csv(
        os.path.join(args.outdir, "A4_regime_baseline_folds.csv"),
        index=False, encoding="utf-8-sig")

    print(f"\n{'='*78}")
    print("  สรุปสำหรับรายงาน")
    print(f"{'='*78}")
    for ticker, g in results.groupby("ticker"):
        best = g.loc[g["accuracy"].idxmax()]
        plain = g[g.baseline == "Global majority"].iloc[0]
        gain = (best["accuracy"] - plain["accuracy"]) * 100
        print(f"  {ticker}: baseline ที่ดีที่สุด = {best['baseline']} "
              f"({best['accuracy']:.4f})")
        if best["baseline"] == "Regime-aware majority":
            print(f"    ดีกว่า global majority {gain:+.2f} pp "
                  f"-> โครงสร้าง tick regime มีข้อมูลจริง")
        print(f"    -> โมเดล ML ต้องชนะ {best['accuracy']:.4f} "
              f"ถึงจะมีความหมาย")
    print(f"\n  [csv] {args.outdir}/A4_regime_baseline.csv")
