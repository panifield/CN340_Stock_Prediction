"""
feature_selection.py — งาน B (ราคาปิด / return)
===============================================
Backward elimination + **การทดสอบด้วย target สุ่ม (permutation guard)**

    cd task_b
    python feature_selection.py                 # group mode (ค่าเริ่มต้น)
    python feature_selection.py --mode both     # เทียบ group กับ single

*** ใช้แค่ train / val เท่านั้น ไม่แตะ test เด็ดขาด ***
สคริปต์นี้เรียก chronological_split แล้วหยิบเฉพาะ parts["train"] กับ
parts["val"] ไม่มีบรรทัดไหนอ้างถึง parts["test"] เลย

---------------------------------------------------------------------------
*** ทำไมต้องตัด feature แบบ "กลุ่ม" ไม่ใช่ทีละตัว ? ***

1. จำนวนการเปรียบเทียบ = ต้นทุนของ selection bias
   - แบบกลุ่ม  (8 กลุ่ม) : 8*9/2  =  36 การเปรียบเทียบ
   - แบบทีละตัว (34 ตัว) : 34*35/2 = 595 การเปรียบเทียบ
   ค่าสูงสุดของ noise ที่สุ่ม N ครั้งโตราว sigma*sqrt(2*ln N)
   -> sqrt(2*ln 595) = 3.57  เทียบกับ  sqrt(2*ln 36) = 2.68
   แปลว่าแบบทีละตัวจะเจอ "การปรับปรุง" ที่พองกว่าราว 33% ทั้งที่เป็น
   noise ล้วน ๆ ในงานที่สัญญาณจริงมีแค่ rho^2 ~ 0.01-0.02 นี่คือหายนะ

2. feature ในโปรเจกต์นี้ correlate กันเองสูงมาก
   close_over_sma5 กับ close_over_ema5 แทบเป็นตัวเดียวกัน ตัดทีละตัว
   อีกตัวก็รับงานแทน MAE แทบไม่ขยับ -> ไม่มีตัวไหนถูกตัด หรือเลือกตัด
   ตัวไหนก็ได้แบบสุ่ม การตัดทั้งกลุ่มถามคำถามที่มีความหมายกว่าว่า
   "ข้อมูล MA ทั้งชุดมีประโยชน์ไหม"

3. เขียนลงรายงานแล้วอธิบายได้
   "กลุ่ม MA ไม่ช่วยอะไรเลย" = ข้อสรุปที่ตรวจสอบซ้ำได้
   "ตัด close_over_ema10 แต่เก็บ close_over_ema5" = อธิบายไม่ได้

ข้อเสียของแบบกลุ่ม (ต้องพูดให้ครบ): หยาบกว่า ถ้ามีแค่ 1 ใน 7 ตัวของกลุ่ม
MA ที่มีประโยชน์จริง แบบกลุ่มจะมองไม่เห็น -- แต่ด้วย rho^2 ~ 0.01 เราไม่มี
statistical power พอจะตรวจจับ effect ระดับ feature เดี่ยวอยู่แล้ว
ข้อเสียนี้จึงแทบไม่มีต้นทุนจริง

---------------------------------------------------------------------------
*** ทำไมต้องมี permutation guard ? ***

backward elimination จะ "หาการปรับปรุงเจอเสมอ" แม้ในข้อมูลที่ไม่มีสัญญาณ
เลย เพราะมันเลือกตัวที่ดีที่สุดจากการลองหลายสิบครั้ง = การขุด noise

วิธีพิสูจน์: สับ y_train แบบสุ่ม (ทำลายความสัมพันธ์กับ X ทิ้งทั้งหมด)
แล้วรันขั้นตอนเดิมเป๊ะ ๆ ถ้ายังได้ "การปรับปรุง" ใกล้เคียงของจริง
=> การปรับปรุงที่เห็นในข้อมูลจริงก็คือ noise ไม่ใช่สัญญาณ

หมายเหตุ: สับเฉพาะ y_train (val ยังเป็นของจริง) เพื่อถามว่า "ถ้าโมเดล
ไม่มีอะไรให้เรียนรู้เลย การคัด feature จะยังดูเหมือนช่วยไหม"

*** ใช้ MAE_return ไม่ใช่ MAE_baht ***
MAE_baht คูณด้วยระดับราคาซึ่งไต่ขึ้นตลอดช่วง val (ADVANC 195 -> 304 บาท)
ทำให้วันปลายชุดมีน้ำหนักมากกว่าวันต้นชุดโดยไม่มีเหตุผล
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# บน Windows คอนโซลเป็น cp1252 พอ print ภาษาไทยจะ crash -- ดู main.py
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import warnings
warnings.filterwarnings("ignore")

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from config import (TICKERS, OUTPUT_DIR, RANDOM_STATE,
                    LAG_DAYS, MA_WINDOWS, VOL_WINDOWS, USE_DAY_OF_WEEK)
from data_loader import load_stock
from features import build_features
from targets import build_targets
from splits import chronological_split
from models import get_regressors
from main import _prepare


# ---------------------------------------------------------------
# นิยาม 8 กลุ่ม feature (สร้างจาก config เพื่อให้ตรงกับ features.py เสมอ)
# ---------------------------------------------------------------
def build_feature_groups():
    """คืน dict {ชื่อกลุ่ม: [ชื่อคอลัมน์ (ยังไม่ใส่ _prev)]}"""
    ma_cols = []
    for w in MA_WINDOWS:
        ma_cols += [f"close_over_sma{w}", f"close_over_ema{w}"]
    if len(MA_WINDOWS) >= 2:
        ma_cols.append(f"sma{MA_WINDOWS[0]}_over_sma{MA_WINDOWS[-1]}")

    groups = {
        "ราคาดิบ": ["close", "open", "high", "low", "volume"],
        "Return lag": [f"ret_{lag}d" for lag in LAG_DAYS],
        "รูปแท่งเทียน": ["hl_range", "oc_change", "close_pos_in_range"],
        "อัตราส่วน MA": ma_cols,
        "ความผันผวน": [f"volatility_{w}d" for w in VOL_WINDOWS],
        "Momentum": ["rsi", "macd", "macd_signal", "macd_hist", "bb_position"],
        "ปริมาณซื้อขาย": ["volume_change", "volume_over_ma20"],
    }
    if USE_DAY_OF_WEEK:
        groups["วันในสัปดาห์"] = [f"dow_{d}" for d in range(5)]

    # features.py เติม _prev ให้ทุกคอลัมน์หลัง shift(1)
    return {g: [f"{c}_prev" for c in cols] for g, cols in groups.items()}


def make_units(X, mode):
    """
    แปลงคอลัมน์ของ X เป็น "หน่วยที่จะถูกตัด"
    mode="group"  -> 8 หน่วย (กลุ่ม)
    mode="single" -> 34 หน่วย (ทีละคอลัมน์)
    """
    if mode == "single":
        return {c: [c] for c in X.columns}

    groups = build_feature_groups()
    covered = [c for cols in groups.values() for c in cols]

    missing = [c for c in covered if c not in X.columns]
    extra = [c for c in X.columns if c not in covered]
    if missing or extra:
        raise ValueError(
            f"กลุ่ม feature ไม่ตรงกับ X\n  ไม่พบใน X: {missing}\n  ไม่ถูกจัดกลุ่ม: {extra}"
        )
    return groups


# ---------------------------------------------------------------
# การประเมินหนึ่งชุด feature
# ---------------------------------------------------------------
def evaluate_subset(X_tr, y_tr, X_va, y_va, cols, model_name):
    """เทรนบน train แล้ววัด MAE_return บน val -- คืนค่ายิ่งต่ำยิ่งดี"""
    model = get_regressors()[model_name]          # instance ใหม่ทุกครั้ง
    wrapped = TransformedTargetRegressor(
        regressor=model, transformer=StandardScaler()
    )
    wrapped.fit(X_tr[cols], y_tr)
    pred = wrapped.predict(X_va[cols])
    return float(mean_absolute_error(y_va, pred))


def backward_eliminate(X_tr, y_tr, X_va, y_va, units, model_name,
                       verbose=False):
    """
    ไล่ตัดหน่วยที่ตัดแล้วดีที่สุดออกทีละหน่วย จนเหลือหน่วยเดียว
    เก็บคะแนนไว้ทุกขั้น แล้วคืนจุดที่ดีที่สุดบนเส้นทาง

    จำนวนการประเมิน = 1 (ชุดเต็ม) + k + (k-1) + ... + 2 = k(k+1)/2
    """
    remaining = dict(units)
    cols_of = lambda d: [c for cols in d.values() for c in cols]

    full_score = evaluate_subset(X_tr, y_tr, X_va, y_va,
                                 cols_of(remaining), model_name)
    n_eval = 1

    path = [{"n_units": len(remaining), "removed": "(ชุดเต็ม)",
             "n_features": len(cols_of(remaining)), "mae_return": full_score}]
    best = {"score": full_score, "units": list(remaining)}

    while len(remaining) > 1:
        scores = {}
        for name in list(remaining):
            trial = {k: v for k, v in remaining.items() if k != name}
            scores[name] = evaluate_subset(X_tr, y_tr, X_va, y_va,
                                           cols_of(trial), model_name)
            n_eval += 1

        drop = min(scores, key=scores.get)      # ตัดตัวที่ตัดแล้ว MAE ต่ำสุด
        remaining.pop(drop)
        score = scores[drop]

        path.append({"n_units": len(remaining), "removed": drop,
                     "n_features": len(cols_of(remaining)),
                     "mae_return": score})
        if score < best["score"]:
            best = {"score": score, "units": list(remaining)}

        if verbose:
            print(f"      ตัด {drop:16s} -> เหลือ {len(remaining)} หน่วย  "
                  f"MAE_return = {score:.6f}")

    improvement = (full_score - best["score"]) / full_score * 100.0
    return {"path": path, "full_score": full_score, "best_score": best["score"],
            "best_units": best["units"], "improvement_pct": improvement,
            "n_eval": n_eval}


# ---------------------------------------------------------------
# permutation guard
# ---------------------------------------------------------------
def permutation_null(X_tr, y_tr, X_va, y_va, units, model_name, n_perm,
                     seed=RANDOM_STATE, verbose=True):
    """
    สับ y_train แบบสุ่ม n_perm ครั้ง แล้วรัน backward elimination ชุดเดิม
    คืน list ของ improvement_pct = การกระจายของ "การปรับปรุงที่เกิดจาก noise"
    """
    rng = np.random.default_rng(seed)
    nulls = []
    for i in range(n_perm):
        y_shuffled = pd.Series(rng.permutation(y_tr.values), index=y_tr.index)
        res = backward_eliminate(X_tr, y_shuffled, X_va, y_va, units,
                                 model_name)
        nulls.append(res["improvement_pct"])
        if verbose:
            print(f"      สุ่มรอบ {i+1}/{n_perm}: "
                  f"การปรับปรุง = {res['improvement_pct']:.2f}%")
    return nulls


# ---------------------------------------------------------------
# รันหนึ่งหุ้น
# ---------------------------------------------------------------
def run_ticker(ticker, mode, model_name, n_perm, verbose=True):
    print("\n" + "#" * 78)
    print(f"#  {ticker}  |  mode = {mode}  |  โมเดล = {model_name}")
    print("#" * 78)

    df = load_stock(ticker, verbose=False)
    X = build_features(df, verbose=False)
    targets = build_targets(df, verbose=False)
    X_, y_ = _prepare(X, targets["y_return"])

    parts = chronological_split(X_, y_, verbose=False)
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]        # <-- ไม่แตะ parts["test"] เด็ดขาด
    print(f"[data] train {len(X_train)} แถว / val {len(X_val)} แถว "
          f"(test ไม่ถูกเรียกใช้)")

    units = make_units(X_, mode)
    k = len(units)
    print(f"[setup] {k} หน่วย -> {k*(k+1)//2} การเปรียบเทียบ")

    t0 = time.time()
    print("\n  [1/2] backward elimination บน target จริง")
    real = backward_eliminate(X_train, y_train, X_val, y_val, units,
                              model_name, verbose=verbose)
    print(f"      ชุดเต็ม     MAE_return = {real['full_score']:.6f}")
    print(f"      ชุดที่ดีสุด MAE_return = {real['best_score']:.6f} "
          f"({len(real['best_units'])} หน่วย)")
    print(f"      => การปรับปรุง = {real['improvement_pct']:.2f}%")
    print(f"      หน่วยที่เหลือ: {', '.join(real['best_units'])}")

    print(f"\n  [2/2] permutation guard ({n_perm} รอบ) — สับ y_train ทิ้งสัญญาณ")
    nulls = permutation_null(X_train, y_train, X_val, y_val, units,
                             model_name, n_perm, verbose=verbose)

    nulls = np.array(nulls)
    # p-value = สัดส่วนของ noise ที่ทำได้ดีเท่าหรือดีกว่าของจริง
    # ใช้สูตร (จำนวนที่ >= จริง + 1) / (n + 1) ตามมาตรฐาน permutation test
    p_value = (np.sum(nulls >= real["improvement_pct"]) + 1) / (len(nulls) + 1)

    print(f"\n  --- สรุป {ticker} ({mode}) ---")
    print(f"  การปรับปรุงจาก target จริง : {real['improvement_pct']:.2f}%")
    print(f"  การปรับปรุงจาก target สุ่ม : เฉลี่ย {nulls.mean():.2f}%  "
          f"(SD {nulls.std():.2f}, สูงสุด {nulls.max():.2f}%)")
    print(f"  p-value                    : {p_value:.3f}")
    if p_value > 0.05:
        print("  => สรุปไม่ได้ว่าการคัด feature เจอสัญญาณจริง "
              "การปรับปรุงอยู่ในระดับที่ noise ทำได้")
    else:
        print("  => การปรับปรุงมากกว่าที่ noise ทำได้อย่างมีนัยสำคัญ")

    print(f"  (ใช้เวลา {time.time()-t0:.1f} วินาที)")

    return {"ticker": ticker, "mode": mode, "model": model_name,
            "real": real, "nulls": nulls, "p_value": p_value}


# ---------------------------------------------------------------
def save_results(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path_rows, summary_rows = [], []
    for r in results:
        t = r["ticker"].replace(".", "_")
        for step in r["real"]["path"]:
            path_rows.append({"stock": r["ticker"], "mode": r["mode"], **step})
        summary_rows.append({
            "stock": r["ticker"], "mode": r["mode"], "model": r["model"],
            "n_units": r["real"]["path"][0]["n_units"],
            "n_eval": r["real"]["n_eval"],
            "mae_full": r["real"]["full_score"],
            "mae_best": r["real"]["best_score"],
            "improvement_pct": r["real"]["improvement_pct"],
            "null_mean_pct": float(r["nulls"].mean()),
            "null_sd_pct": float(r["nulls"].std()),
            "null_max_pct": float(r["nulls"].max()),
            "n_perm": len(r["nulls"]),
            "p_value": r["p_value"],
            "best_units": " | ".join(r["real"]["best_units"]),
        })

    p1 = os.path.join(OUTPUT_DIR, "feature_selection_path.csv")
    p2 = os.path.join(OUTPUT_DIR, "feature_selection_summary.csv")
    pd.DataFrame(path_rows).to_csv(p1, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(p2, index=False, encoding="utf-8-sig")
    print(f"\n[save] {p1}")
    print(f"[save] {p2}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Backward elimination + permutation guard (train/val เท่านั้น)"
    )
    p.add_argument("--mode", choices=["group", "single", "both"],
                   default="group",
                   help="group = 8 กลุ่ม (แนะนำ), single = ทีละ feature")
    p.add_argument("--model", default="Random Forest",
                   choices=["Random Forest", "XGBoost", "ANN (MLP)"])
    p.add_argument("--n-perm", type=int, default=20,
                   help="จำนวนรอบสับ target (ยิ่งมากยิ่งแม่น แต่ช้า)")
    p.add_argument("--n-perm-single", type=int, default=None,
                   help="จำนวนรอบสับเฉพาะ single mode (ค่าเริ่มต้น = --n-perm) "
                        "ตั้งน้อยกว่าได้เพราะ single ใช้การประเมิน 595 ครั้ง "
                        "เทียบกับ group ที่ใช้ 36 ครั้ง = ช้ากว่า ~17 เท่า")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    modes = ["group", "single"] if args.mode == "both" else [args.mode]

    print("=" * 78)
    print("  งาน B : Feature Selection + Permutation Guard")
    print("  *** ใช้ train/val เท่านั้น ไม่แตะ test ***")
    print("=" * 78)

    results = []
    for mode in modes:
        n_perm = args.n_perm
        if mode == "single" and args.n_perm_single is not None:
            n_perm = args.n_perm_single
        for ticker in TICKERS:
            results.append(run_ticker(ticker, mode, args.model,
                                      n_perm, verbose=not args.quiet))

    if len(modes) == 2:
        print("\n" + "=" * 78)
        print("  เทียบ group กับ single — ทำไมกลุ่มถึงน่าเชื่อถือกว่า")
        print("=" * 78)
        for ticker in TICKERS:
            rows = [r for r in results if r["ticker"] == ticker]
            print(f"\n  {ticker}")
            for r in rows:
                print(f"    {r['mode']:7s} ({r['real']['n_eval']:3d} เปรียบเทียบ)  "
                      f"จริง {r['real']['improvement_pct']:5.2f}%  |  "
                      f"สุ่มเฉลี่ย {r['nulls'].mean():5.2f}%  |  "
                      f"p = {r['p_value']:.3f}")

    save_results(results)
    print("\nเสร็จสิ้น")
    return results


if __name__ == "__main__":
    main()
