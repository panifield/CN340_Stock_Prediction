"""
xgb_sweep.py — งาน B (ราคาปิด / return)
=======================================
กวาดหา n_estimators / learning_rate / max_depth ของ XGBoost บน validation set
+ ตรวจความไวของ Random Forest 3 config (ไม่ได้ใช้เลือกค่า)

    cd task_b
    python xgb_sweep.py

*** ใช้แค่ train / val เท่านั้น ไม่แตะ test เด็ดขาด ***
load_split() ลบ parts["test"] ทิ้งทันทีหลัง split ไม่มีทางถูกเรียกใช้

---------------------------------------------------------------------------
*** ทำไมต้องกวาด XGBoost ***

1) ความเป็นธรรมของการเปรียบเทียบ
   ANN ถูกกวาด 140 ชุดบน val ส่วน RF/XGB ใช้ค่าที่ตั้งเองชุดเดียว
   การรายงานว่า "XGBoost แย่ที่สุด" ทั้งที่มันได้โอกาสน้อยกว่า ANN 140 เท่า
   เป็นการเทียบที่ไม่เท่ากัน (เหตุผลเดียวกับที่ config.py ใช้อธิบายว่าทำไม
   ต้องกวาด ANN ใหม่หลังตัดราคาดิบ -- ที่นี่คือการใช้กฎเดิมให้สม่ำเสมอ)

2) หลักฐาน 3 ชิ้นที่เป็นอิสระจากกัน ชี้ว่า XGB ปรับเทียบผิดแบบเดียวกับ
   ANN ตอนก่อนกวาด (ทำนายแกว่งเกินจริง ไม่ใช่ทายผิดทิศ):
     ก. StdRatio / Rho = 0.414/0.143 (KBANK) และ 0.369/0.128 (ADVANC)
        คือแกว่งเกินราว 2.9 เท่าของค่าที่เหมาะสมทางทฤษฎี (StdRatio = Rho)
        ส่วน RF อยู่ที่ราว 1.0 เท่า
     ข. DM test เทียบ Naive: XGB z = +3.07 (p = 0.002) บน ADVANC
        = แย่กว่า Naive อย่างมีนัยสำคัญ ส่วน RF ได้ z = -0.73 (p = 0.46)
     ค. permutation guard ใน feature_selection.py: การคัด feature บน
        target สุ่มทำให้ XGB "ดีขึ้น" ได้เฉลี่ย 7.18% (สูงสุด 17.31%)
        เทียบกับ RF ที่ได้แค่ 1.87% -- โมเดลที่แกว่งเกินจริงจะไวต่อการ
        เปลี่ยน feature มาก จึงสร้าง "การปรับปรุงปลอม" ได้มากกว่า

---------------------------------------------------------------------------
*** กริดที่กวาด (ประกาศไว้ก่อนรัน) ***

ตรึงทุกอย่างที่ไม่เกี่ยวกับ "ความแรงในการ fit": subsample, colsample_bytree,
reg_lambda, random_state คงเดิมตาม config.py แล้วกวาดเฉพาะ 3 ปุ่มที่กำหนด
ว่าโมเดลจะไล่เรียน residual แรงแค่ไหน:

    n_estimators   [100, 200, 400]
    learning_rate  [0.01, 0.03, 0.05]
    max_depth      [2, 3, 4]          <- ค่าปัจจุบันคือ 4

max_depth อยู่ในกริดด้วย (ไม่ตรึงไว้ที่ 3) เพราะค่าที่ใช้อยู่จริงคือ 4
การตรึงไว้ที่ 3 เท่ากับเปลี่ยนค่าที่ใช้อยู่โดยไม่ได้ทดสอบ และ depth ก็เป็น
"ปุ่มหด" เหมือนอีกสองตัว -- รวม 27 ชุด ยังน้อยกว่าที่ ANN ได้ (140) มาก

*** กฎการเลือก — ยกมาจาก config.py ทุกตัวอักษร ประกาศก่อนดูผล ***

    "ในบรรดาชุดที่ StdRatio อยู่ใน 0.10-0.20 ทั้งสองหุ้น
     เลือกตัวที่ MAE_return เฉลี่ยต่ำสุด เสมอกันเลือกตัวเล็กกว่า"

(เล็กกว่า = n_estimators น้อยกว่า ถ้ายังเสมอให้ดู max_depth)

*** ถ้าไม่มีชุดไหนผ่านเกณฑ์ทั้งสองหุ้น -> คงค่าเดิมไว้ และรายงานตามจริง ***
กติกาข้อนี้สำคัญ เพราะมีความเป็นไปได้จริงที่ KBANK กับ ADVANC จะต้องการค่า
คนละทาง ถ้าเกิดขึ้นจริงนั่นคือผลลัพธ์ที่มีความหมาย (ไม่มีค่ากลางที่ดีกับ
ทั้งสองหุ้น = ความต่างที่เห็นคือ noise) ห้ามเลือกแยกรายหุ้นเพื่อให้ตัวเลขสวย

---------------------------------------------------------------------------
*** ส่วนที่ 2: Random Forest — ตรวจความไว ไม่ใช่การเลือกค่า ***

RF หดค่าทำนายเข้าหาค่ากลางด้วยตัวเองอยู่แล้ว (เฉลี่ย 400 ต้น x ใบละ >= 20
วัน) StdRatio จึงใกล้ Rho อยู่แล้วโดยไม่ต้องจูน ส่วนนี้ลองแค่ 3 config
เพื่อ "ยืนยันว่าปรับแล้วไม่ขยับ" **ผลส่วนนี้จะไม่ถูกนำไปเปลี่ยน RF_PARAMS
ไม่ว่าออกมาเป็นอย่างไร** จึงไม่ใช่การเลือกค่าจาก val
"""

import os
import sys

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from config import TICKERS, OUTPUT_DIR, XGB_PARAMS, RF_PARAMS
from data_loader import load_stock
from features import build_features
from targets import build_targets
from splits import chronological_split
from models import _unscaled
from evaluate import regression_metrics
from main import _prepare


# --- กริดที่จะกวาด (ประกาศไว้ก่อนรัน) ---
N_ESTIMATORS = [100, 200, 400]
LEARNING_RATES = [0.01, 0.03, 0.05]
MAX_DEPTHS = [2, 3, 4]

TARGET_BAND = (0.10, 0.20)      # ช่วง StdRatio ที่ยอมรับ (เดียวกับ ANN)

# --- config ของ RF ที่ใช้ "ตรวจความไว" เท่านั้น ---
RF_CHECKS = [
    ("ปัจจุบัน", {}),
    ("max_features=0.3", {"max_features": 0.3}),
    ("min_samples_leaf=50", {"min_samples_leaf": 50}),
]


def load_split(ticker):
    """คืน (X_train, y_train, X_val, y_val, prev_close_val) -- ไม่แตะ test"""
    df = load_stock(ticker, verbose=False)
    X = build_features(df, verbose=False)
    targets = build_targets(df, verbose=False)
    extra = targets[["prev_close", "close"]]
    X_, y_, extra_ = _prepare(X, targets["y_return"], extra, verbose=False)

    parts = chronological_split(X_, y_, verbose=False)
    del parts["test"]                  # <-- ลบทิ้งทันที ไม่มีทางถูกใช้
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]
    return X_train, y_train, X_val, y_val, extra_.loc[X_val.index, "prev_close"]


def evaluate_model(data, estimator):
    """เทรนหนึ่งชุดบน train แล้ววัดผลบน val"""
    X_train, y_train, X_val, y_val, prev_val = data
    wrapped = TransformedTargetRegressor(
        regressor=_unscaled(estimator), transformer=StandardScaler()
    )
    wrapped.fit(X_train, y_train)
    pred = wrapped.predict(X_val)
    return regression_metrics(y_val, pred, prev_val)


def run_xgb_sweep(data_by_ticker):
    grid = [(n, lr, d) for n in N_ESTIMATORS
            for lr in LEARNING_RATES for d in MAX_DEPTHS]
    print(f"[setup] กวาด {len(grid)} config x {len(TICKERS)} หุ้น = "
          f"{len(grid)*len(TICKERS)} การเทรน")

    rows = []
    for n, lr, depth in grid:
        for ticker in TICKERS:
            params = dict(XGB_PARAMS)
            params.update(n_estimators=n, learning_rate=lr, max_depth=depth)
            m = evaluate_model(data_by_ticker[ticker], XGBRegressor(**params))
            rows.append({
                "n_estimators": n, "learning_rate": lr, "max_depth": depth,
                "stock": ticker,
                **{k: m[k] for k in ("StdRatio", "Rho", "Bias", "R2_return",
                                     "MAE_return", "RMSE_return", "MAE_baht",
                                     "RMSE_baht", "DirAcc")},
            })
    return pd.DataFrame(rows)


def select_best(df):
    """
    กฎที่ประกาศไว้: StdRatio อยู่ใน 0.10-0.20 ทั้งสองหุ้น
    -> MAE_return เฉลี่ยต่ำสุด -> เสมอกันเลือก n_estimators แล้ว max_depth ที่น้อยกว่า
    """
    lo, hi = TARGET_BAND
    df = df.copy()
    df["in_band"] = df["StdRatio"].between(lo, hi)

    key = ["n_estimators", "learning_rate", "max_depth"]
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

    best = ok.sort_values(["mean_mae_return", "n_estimators", "max_depth"]).iloc[0]
    return best, agg


def run_rf_check(data_by_ticker):
    """ตรวจความไวของ RF -- ผลส่วนนี้ไม่ถูกนำไปเปลี่ยน RF_PARAMS"""
    rows = []
    for label, override in RF_CHECKS:
        for ticker in TICKERS:
            params = dict(RF_PARAMS)
            params.update(override)
            m = evaluate_model(data_by_ticker[ticker],
                               RandomForestRegressor(**params))
            rows.append({
                "config": label, "stock": ticker,
                **{k: m[k] for k in ("StdRatio", "Rho", "MAE_return",
                                     "MAE_baht", "DirAcc")},
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print("  งาน B : กวาดหาพารามิเตอร์ของ XGBoost บน validation set")
    print("  *** ใช้ train/val เท่านั้น ไม่แตะ test ***")
    print("=" * 78)
    print("\nกฎการเลือก (ประกาศก่อนดูผล):")
    print(f"  1) StdRatio ต้องอยู่ใน {TARGET_BAND} ทั้งสองหุ้น")
    print("  2) ในบรรดาที่ผ่าน เลือก MAE_return เฉลี่ยต่ำสุด")
    print("  3) เสมอกัน เลือก n_estimators แล้ว max_depth ที่น้อยกว่า")
    print("  4) ถ้าไม่มีชุดไหนผ่านทั้งสองหุ้น -> คงค่าเดิมไว้ รายงานตามจริง\n")

    data = {t: load_split(t) for t in TICKERS}
    n_input = data[TICKERS[0]][0].shape[1]
    print(f"[setup] ใช้ {n_input} features | "
          f"train {len(data[TICKERS[0]][0])} แถว / val {len(data[TICKERS[0]][2])} แถว")

    df = run_xgb_sweep(data)
    best, agg = select_best(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = f"{n_input}feat"

    if best is not None:
        sel = ((df["n_estimators"] == best["n_estimators"])
               & (df["learning_rate"] == best["learning_rate"])
               & (df["max_depth"] == best["max_depth"]))
        df["selected"] = sel
    else:
        df["selected"] = False
    df["n_features"] = n_input
    path = os.path.join(OUTPUT_DIR, f"xgb_sweep_val_{tag}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("  ผลการกวาด XGBoost")
    print("=" * 78)
    print(f"  config ที่ผ่านเกณฑ์ StdRatio ทั้งสองหุ้น: "
          f"{int(agg['passes'].sum())} จาก {len(agg)}")

    # ค่าที่ config ใช้อยู่ ณ ตอนรัน (ก่อนนำผลรอบนี้ไปใช้จะเป็นค่าเดิม
    # หลังนำไปใช้แล้วบรรทัดนี้จะกลายเป็นค่าที่เลือกไว้ ซึ่งถูกต้องตามชื่อ)
    print("\n  ค่าที่ config ใช้อยู่ตอนรัน "
          f"(n_estimators={XGB_PARAMS['n_estimators']}, "
          f"lr={XGB_PARAMS['learning_rate']}, "
          f"depth={XGB_PARAMS['max_depth']}):")
    cur = df[(df["n_estimators"] == XGB_PARAMS["n_estimators"])
             & (df["learning_rate"] == XGB_PARAMS["learning_rate"])
             & (df["max_depth"] == XGB_PARAMS["max_depth"])]
    cols = ["stock", "StdRatio", "Rho", "Bias", "MAE_return", "MAE_baht", "DirAcc"]
    print(cur[cols].to_string(index=False))

    if best is None:
        print("\n  !! ไม่มี config ไหนผ่านเกณฑ์ทั้งสองหุ้น -> คงค่าเดิมไว้ตามกฎข้อ 4")
        print("     (ผลลัพธ์นี้มีความหมาย: ไม่มีค่ากลางที่เหมาะกับทั้งสองหุ้น)")
    else:
        print(f"\n  >>> เลือก: n_estimators={int(best['n_estimators'])}  "
              f"learning_rate={best['learning_rate']}  "
              f"max_depth={int(best['max_depth'])}")
        print(f"      MAE_return เฉลี่ย = {best['mean_mae_return']:.6f}  "
              f"StdRatio {best['min_stdratio']:.4f} - {best['max_stdratio']:.4f}")
        print("\n  รายละเอียดของ config ที่เลือก:")
        print(df[df["selected"]][cols].to_string(index=False))

        print("\n  5 อันดับแรกที่ผ่านเกณฑ์ (เรียงตาม MAE_return เฉลี่ย):")
        top = agg[agg["passes"]].sort_values(
            ["mean_mae_return", "n_estimators", "max_depth"]).head(5)
        print(top[["n_estimators", "learning_rate", "max_depth",
                   "mean_mae_return", "min_stdratio", "max_stdratio",
                   "mean_rho"]].to_string(index=False))

    print("\n  config ที่ StdRatio ใกล้เกณฑ์ที่สุด 8 อันดับแรก (ดูภาพรวม):")
    near = agg.assign(gap=(agg["min_stdratio"] - TARGET_BAND[0]).abs()
                      + (agg["max_stdratio"] - TARGET_BAND[1]).abs())
    print(near.sort_values("gap").head(8)[
        ["n_estimators", "learning_rate", "max_depth", "min_stdratio",
         "max_stdratio", "mean_mae_return", "mean_rho", "passes"]
    ].to_string(index=False))

    print(f"\n[save] {path}")

    # ---------------- ส่วนที่ 2: RF ----------------
    print("\n" + "=" * 78)
    print("  Random Forest — ตรวจความไว (ไม่ใช่การเลือกค่า)")
    print("  *** ผลส่วนนี้จะไม่ถูกนำไปเปลี่ยน RF_PARAMS ไม่ว่าออกมาอย่างไร ***")
    print("=" * 78)
    rf = run_rf_check(data)
    print(rf.round(6).to_string(index=False))

    base = rf[rf["config"] == "ปัจจุบัน"].set_index("stock")["MAE_return"]
    print("\n  ส่วนต่างจาก config ปัจจุบัน:")
    for label, _ in RF_CHECKS[1:]:
        sub = rf[rf["config"] == label].set_index("stock")["MAE_return"]
        d = (sub - base) / base * 100
        print(f"    {label:22s} " + "  ".join(
            f"{t}: {d[t]:+.2f}%" for t in TICKERS))

    rf_path = os.path.join(OUTPUT_DIR, f"rf_check_val_{tag}.csv")
    rf.to_csv(rf_path, index=False, encoding="utf-8-sig")
    print(f"\n[save] {rf_path}")

    print("\nเสร็จสิ้น")
    return df, best, rf


if __name__ == "__main__":
    main()
