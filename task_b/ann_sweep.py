"""
ann_sweep.py — งาน B (ราคาปิด / return)
=======================================
กวาดหา hyperparameter ของ ANN บน validation set

    cd task_b
    python ann_sweep.py                      # กวาดบน 29 features (ปิดราคาดิบ)
    python ann_sweep.py --use-raw-price-levels   # กวาดบน 34 features (ของเดิม)

*** ใช้แค่ train / val เท่านั้น ไม่แตะ test เด็ดขาด ***
หยิบเฉพาะ parts["train"] กับ parts["val"] ไม่มีบรรทัดไหนอ้าง parts["test"]

---------------------------------------------------------------------------
*** ทำไมต้องกวาดใหม่ ? ***

ค่า alpha = 7.0 ที่ใช้อยู่ถูกกวาดหาบน **34 features** พอ ablation H1 ตัด
กลุ่มราคาดิบออกเหลือ 29 features ปรากฏว่า ANN ยุบเข้าใกล้ค่าคงที่:

    StdRatio  KBANK  0.1219 -> 0.0378   (ยุบ 69%)
    StdRatio  ADVANC 0.1224 -> 0.0458   (ยุบ 63%)

เพราะ L2 penalty ลงโทษผลรวมกำลังสองของ weight ทั้งหมด พอจำนวน input
ลดลง 15% ค่า alpha เดิมจึงกลายเป็นแรงเกินไปเมื่อเทียบกับ capacity ที่เหลือ

ถ้ารายงานตัวเลข ANN บน 29 features โดยใช้ alpha ที่จูนมาจาก 34 features
= การเปรียบเทียบที่ไม่เป็นธรรม (ANN ถูกมัดมือข้างหนึ่งไว้)

---------------------------------------------------------------------------
*** กฎการเลือก — ยกมาจาก config.py ตามเดิมทุกตัวอักษร ***

    "ในบรรดาชุดที่ StdRatio อยู่ใน 0.10-0.20 ทั้งสองหุ้น
     เลือกตัวที่ MAE_return เฉลี่ยต่ำสุด เสมอกันเลือกตัวเล็กกว่า"

ประกาศไว้ก่อนดูผลรอบนี้เช่นกัน ไม่มีการเปลี่ยนกฎกลางทาง

เหตุผลที่เกณฑ์เป็น StdRatio ไม่ใช่ MAE ล้วน: alpha ที่สูงมากจะกด StdRatio
ลงเหลือ 0 ซึ่งได้ MAE ต่ำที่สุดเสมอ แต่นั่นคือ Mean Return baseline ที่ใส่
neural network ครอบไว้ ไม่ใช่โมเดลที่เรียนรู้อะไรจริง
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import warnings
warnings.filterwarnings("ignore")

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from config import TICKERS, OUTPUT_DIR, ANN_PARAMS
from data_loader import load_stock
from features import build_features
from targets import build_targets
from splits import chronological_split
from models import _scaled
from evaluate import regression_metrics
from main import _prepare


# --- ตารางที่จะกวาด (ประกาศไว้ก่อนรัน) ---
SIZES = [(8, 4), (16, 8), (32, 16), (64, 32)]
COARSE_ALPHAS = [0.001, 0.1, 1.0, 10.0, 100.0]
REFINE_ALPHAS = [0.3, 0.5, 2.0, 3.0, 5.0, 7.0]
SOLVERS = ["adam", "lbfgs"]

TARGET_BAND = (0.10, 0.20)      # ช่วง StdRatio ที่ยอมรับ


def n_params(sizes, n_input):
    """จำนวนพารามิเตอร์ของ MLP (weight + bias) เอาไว้ใช้ตัดสินตอนเสมอกัน"""
    dims = [n_input] + list(sizes) + [1]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


def load_split(ticker, use_raw):
    """คืน (X_train, y_train, X_val, y_val, prev_close_val) -- ไม่แตะ test"""
    df = load_stock(ticker, verbose=False)
    X = build_features(df, verbose=False, use_raw_price_levels=use_raw)
    targets = build_targets(df, verbose=False)
    extra = targets[["prev_close", "close"]]
    X_, y_, extra_ = _prepare(X, targets["y_return"], extra, verbose=False)

    parts = chronological_split(X_, y_, verbose=False)
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]        # <-- ไม่แตะ parts["test"] เด็ดขาด
    return X_train, y_train, X_val, y_val, extra_.loc[X_val.index, "prev_close"]


def evaluate_config(data, sizes, alpha, solver):
    """เทรน ANN หนึ่งชุดแล้ววัดผลบน val"""
    X_train, y_train, X_val, y_val, prev_val = data

    params = dict(ANN_PARAMS)
    params.update(hidden_layer_sizes=sizes, alpha=alpha, solver=solver)

    wrapped = TransformedTargetRegressor(
        regressor=_scaled(MLPRegressor(**params)),
        transformer=StandardScaler(),
    )
    wrapped.fit(X_train, y_train)
    pred = wrapped.predict(X_val)
    return regression_metrics(y_val, pred, prev_val)


def run_sweep(use_raw, verbose=True):
    data = {t: load_split(t, use_raw) for t in TICKERS}
    n_input = data[TICKERS[0]][0].shape[1]
    print(f"[setup] ใช้ {n_input} features | "
          f"train {len(data[TICKERS[0]][0])} แถว / val {len(data[TICKERS[0]][2])} แถว")

    grid = [("coarse", s, a, sv) for s in SIZES
            for a in COARSE_ALPHAS for sv in SOLVERS]
    grid += [("refine", s, a, sv) for s in SIZES
             for a in REFINE_ALPHAS for sv in SOLVERS]
    print(f"[setup] กวาด {len(grid)} config x {len(TICKERS)} หุ้น = "
          f"{len(grid)*len(TICKERS)} การเทรน")

    rows, t0 = [], time.time()
    for i, (stage, sizes, alpha, solver) in enumerate(grid, 1):
        for ticker in TICKERS:
            m = evaluate_config(data[ticker], sizes, alpha, solver)
            rows.append({
                "sweep_stage": stage,
                "hidden": f"({sizes[0]},{sizes[1]})",
                "n_params": n_params(sizes, n_input),
                "alpha": alpha, "solver": solver, "stock": ticker,
                "n_features": n_input,
                **{k: m[k] for k in ("StdRatio", "Rho", "Bias", "R2_return",
                                     "MAE_return", "RMSE_return", "MAE_baht",
                                     "RMSE_baht", "DirAcc")},
            })
        if verbose and i % 10 == 0:
            print(f"    {i}/{len(grid)} config  ({time.time()-t0:.0f} วินาที)")

    return pd.DataFrame(rows)


def select_best(df):
    """
    ใช้กฎที่ประกาศไว้: StdRatio อยู่ใน 0.10-0.20 ทั้งสองหุ้น
    -> MAE_return เฉลี่ยต่ำสุด -> เสมอกันเลือก n_params น้อยกว่า
    """
    lo, hi = TARGET_BAND
    df = df.copy()
    df["in_band"] = df["StdRatio"].between(lo, hi)

    key = ["hidden", "n_params", "alpha", "solver"]
    agg = df.groupby(key, as_index=False).agg(
        n_in_band=("in_band", "sum"),
        n_stock=("stock", "nunique"),
        mean_mae_return=("MAE_return", "mean"),
        min_stdratio=("StdRatio", "min"),
        max_stdratio=("StdRatio", "max"),
        mean_rho=("Rho", "mean"),
    )
    agg["passes"] = agg["n_in_band"] == agg["n_stock"]

    ok = agg[agg["passes"]]
    if ok.empty:
        return None, agg

    best = ok.sort_values(["mean_mae_return", "n_params"]).iloc[0]
    return best, agg


def main():
    p = argparse.ArgumentParser(description="กวาด hyperparameter ANN บน val")
    p.add_argument("--use-raw-price-levels", action="store_true",
                   help="กวาดบน 34 features (ของเดิม) แทน 29 features")
    args = p.parse_args()
    use_raw = args.use_raw_price_levels

    print("=" * 78)
    print("  งาน B : กวาดหา alpha ของ ANN บน validation set")
    print(f"  feature ราคาดิบ: {'เปิด (34)' if use_raw else 'ปิด (29)'}")
    print("  *** ใช้ train/val เท่านั้น ไม่แตะ test ***")
    print("=" * 78)
    print("\nกฎการเลือก (ประกาศก่อนดูผล):")
    print(f"  1) StdRatio ต้องอยู่ใน {TARGET_BAND} ทั้งสองหุ้น")
    print("  2) ในบรรดาที่ผ่าน เลือก MAE_return เฉลี่ยต่ำสุด")
    print("  3) เสมอกัน เลือกโมเดลที่พารามิเตอร์น้อยกว่า\n")

    df = run_sweep(use_raw)
    best, agg = select_best(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = "34feat" if use_raw else "29feat"
    path = os.path.join(OUTPUT_DIR, f"ann_sweep_val_{tag}.csv")

    if best is not None:
        sel = ((df["hidden"] == best["hidden"]) & (df["alpha"] == best["alpha"])
               & (df["solver"] == best["solver"]))
        df["selected"] = sel
    else:
        df["selected"] = False
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("  ผลการกวาด")
    print("=" * 78)
    print(f"  config ที่ผ่านเกณฑ์ StdRatio ทั้งสองหุ้น: "
          f"{int(agg['passes'].sum())} จาก {len(agg)}")

    if best is None:
        print("  !! ไม่มี config ไหนผ่านเกณฑ์เลย -- ต้องทบทวนช่วง alpha ที่กวาด")
    else:
        print(f"\n  >>> เลือก: hidden={best['hidden']}  alpha={best['alpha']}  "
              f"solver={best['solver']}")
        print(f"      ({int(best['n_params'])} พารามิเตอร์)  "
              f"MAE_return เฉลี่ย = {best['mean_mae_return']:.6f}")
        print(f"      StdRatio อยู่ระหว่าง {best['min_stdratio']:.4f} - "
              f"{best['max_stdratio']:.4f}")

        print("\n  รายละเอียดของ config ที่เลือก:")
        cols = ["stock", "StdRatio", "Rho", "Bias", "R2_return",
                "MAE_return", "MAE_baht", "DirAcc"]
        print(df[df["selected"]][cols].to_string(index=False))

        print("\n  5 อันดับแรกที่ผ่านเกณฑ์ (เรียงตาม MAE_return เฉลี่ย):")
        top = agg[agg["passes"]].sort_values(
            ["mean_mae_return", "n_params"]).head(5)
        print(top[["hidden", "n_params", "alpha", "solver",
                   "mean_mae_return", "min_stdratio", "max_stdratio",
                   "mean_rho"]].to_string(index=False))

    print(f"\n[save] {path}")
    print("\nเสร็จสิ้น")
    return df, best


if __name__ == "__main__":
    main()
