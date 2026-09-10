"""
ablation_h1.py — งาน B (ราคาปิด / return)
=========================================
ทดสอบสมมติฐาน H1 ด้วยการทดลองแบบ A/B ที่ควบคุมตัวแปร

    cd task_b
    python ablation_h1.py

*** ใช้แค่ train / val เท่านั้น ไม่แตะ test เด็ดขาด ***
หยิบเฉพาะ parts["train"] กับ parts["val"] ไม่มีบรรทัดไหนอ้าง parts["test"]

---------------------------------------------------------------------------
*** สมมติฐาน H1 ***

    "โมเดลเอียง (|Bias| สูง) เพราะ feature ราคาดิบ (close/open/high/low/
     volume) ใน val หลุดออกนอกช่วงที่เคยเห็นตอน train
     tree model extrapolate ไม่ได้ จึงทำนายตันอยู่ที่ขอบของ train"

ราคาดิบเป็น feature กลุ่มเดียวที่ไม่ stationary -- กลุ่มอื่นเป็นอัตราส่วน
(close/sma20) หรือ return ซึ่งอยู่รอบ ๆ ค่าคงที่เสมอ ไม่ว่าราคาจะไต่ไปถึงไหน

*** คำทำนายที่ประกาศไว้ก่อนรัน (pre-registration) ***

ถ้า H1 เป็นจริง การปิด flag (ตัดราคาดิบทิ้ง) จะทำให้:
  1. |Bias| ของ ADVANC ลดลงชัดเจน (จาก ~0.42 ของ XGBoost เหลือ < 0.15)
  2. KBANK แทบไม่เปลี่ยน
เพราะ ADVANC มีราคาใน val หลุดกรอบ train ราว 36% ส่วน KBANK หลุด 0%

ถ้าผลออกมาว่า **ทั้งสองหุ้นเปลี่ยนพอ ๆ กัน** = H1 ผิด (ไม่ได้เกิดจากการ
หลุดกรอบ แต่เกิดจากอย่างอื่น)
ถ้า **ADVANC ดีขึ้นแต่ KBANK แย่ลง/ไม่เปลี่ยน** = H1 ถูก และ KBANK ทำหน้าที่
เป็นกลุ่มควบคุม (control group) ที่ดี

การมีกลุ่มควบคุมคือสิ่งที่ทำให้นี่เป็น "การทดลอง" ไม่ใช่แค่ "การสังเกต"

---------------------------------------------------------------------------
*** ต่างจากผลที่ได้จาก feature_selection.py อย่างไร ***

เส้นทางการตัดกลุ่มใน feature_selection บอกใบ้ทิศทางเดียวกันอยู่แล้ว
(ADVANC ตัดราคาดิบเป็นกลุ่มแรก / KBANK หวงไว้จนเกือบท้าย) แต่ **เทียบกัน
ไม่ได้** เพราะวัดคนละจุดของเส้นทาง -- ADVANC วัดตอนมี 8 กลุ่ม ส่วน KBANK
วัดตอนเหลือ 2 กลุ่ม จำนวน feature อื่น ๆ ต่างกันคนละเรื่อง

ไฟล์นี้แก้ปัญหานั้นด้วยการเปลี่ยน "เฉพาะ" กลุ่มราคาดิบ ให้กลุ่มอื่นครบ
เหมือนกันทุกประการทั้งสองเงื่อนไข = การเปรียบเทียบที่ควบคุมตัวแปรจริง
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

from config import TICKERS, OUTPUT_DIR
from data_loader import load_stock
from features import build_features, verify_no_leak
from targets import build_targets
from splits import chronological_split
from models import get_regressors
from evaluate import regression_metrics
from main import _prepare


def out_of_range_stat(prev_close_train, prev_close_val):
    """
    วัดว่าราคาใน val หลุดออกนอกช่วง [min, max] ของ train กี่ %
    นี่คือ "ตัวแปรต้น" ที่ H1 บอกว่าเป็นสาเหตุ
    """
    lo, hi = float(prev_close_train.min()), float(prev_close_train.max())
    v = np.asarray(prev_close_val, dtype=float)
    outside = (v < lo) | (v > hi)
    return {
        "train_min": lo, "train_max": hi,
        "val_min": float(v.min()), "val_max": float(v.max()),
        "out_of_range_pct": float(outside.mean() * 100.0),
    }


def run_condition(df, targets, use_raw, verbose=False):
    """เทรน 3 โมเดลบน train แล้ววัดผลบน val -- คืน dict ของ metric"""
    X = build_features(df, verbose=False, use_raw_price_levels=use_raw)
    verify_no_leak(df, X, sample_idx=100, use_raw_price_levels=use_raw)

    y = targets["y_return"]
    extra = targets[["prev_close", "close"]]
    X_, y_, extra_ = _prepare(X, y, extra, verbose=verbose)

    parts = chronological_split(X_, y_, verbose=False)
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]          # <-- ไม่แตะ parts["test"] เด็ดขาด
    prev_val = extra_.loc[X_val.index, "prev_close"]
    prev_train = extra_.loc[X_train.index, "prev_close"]

    rows = {}
    for name, model in get_regressors().items():
        wrapped = TransformedTargetRegressor(
            regressor=model, transformer=StandardScaler()
        )
        wrapped.fit(X_train, y_train)
        pred = wrapped.predict(X_val)
        rows[name] = regression_metrics(y_val, pred, prev_val)

    return rows, X_.shape[1], out_of_range_stat(prev_train, prev_val)


def main():
    print("=" * 78)
    print("  งาน B : Ablation H1 — feature ราคาดิบทำให้โมเดลเอียงจริงไหม")
    print("  *** ใช้ train/val เท่านั้น ไม่แตะ test ***")
    print("=" * 78)
    print("\nคำทำนายที่ประกาศก่อนดูผล:")
    print("  ถ้า H1 ถูก -> ปิดราคาดิบแล้ว |Bias| ของ ADVANC ลดลงชัดเจน")
    print("               ส่วน KBANK แทบไม่เปลี่ยน (ทำหน้าที่เป็นกลุ่มควบคุม)")

    records = []
    for ticker in TICKERS:
        print("\n" + "#" * 78)
        print(f"#  {ticker}")
        print("#" * 78)

        df = load_stock(ticker, verbose=False)
        targets = build_targets(df, verbose=False)

        res = {}
        for use_raw in (True, False):
            rows, n_feat, oor = run_condition(df, targets, use_raw)
            res[use_raw] = rows
            label = "เปิด (34 features)" if use_raw else "ปิด (29 features)"
            print(f"\n  [ราคาดิบ {label}] ใช้ {n_feat} features")
            for name, m in rows.items():
                print(f"    {name:16s} Bias={m['Bias']:+.4f}  "
                      f"MAE_return={m['MAE_return']:.6f}  "
                      f"StdRatio={m['StdRatio']:.4f}  Rho={m['Rho']:+.4f}")

        print(f"\n  ตัวแปรต้นของ H1 — ราคาใน val หลุดกรอบ train:")
        print(f"    ช่วง train = [{oor['train_min']:.2f}, {oor['train_max']:.2f}] บาท")
        print(f"    ช่วง val   = [{oor['val_min']:.2f}, {oor['val_max']:.2f}] บาท")
        print(f"    หลุดกรอบ   = {oor['out_of_range_pct']:.2f}% ของวันใน val")

        print(f"\n  ผลของการปิดราคาดิบ (ค่าลบ = |Bias| ลดลง = ดีขึ้นตาม H1):")
        for name in res[True]:
            b_on = abs(res[True][name]["Bias"])
            b_off = abs(res[False][name]["Bias"])
            d_bias = (b_off - b_on) / b_on * 100 if b_on else np.nan
            mae_on = res[True][name]["MAE_return"]
            mae_off = res[False][name]["MAE_return"]
            d_mae = (mae_off - mae_on) / mae_on * 100
            print(f"    {name:16s} |Bias| {b_on:.4f} -> {b_off:.4f} "
                  f"({d_bias:+.1f}%)   MAE_return {d_mae:+.2f}%")

            records.append({
                "stock": ticker, "model": name,
                "out_of_range_pct": oor["out_of_range_pct"],
                "bias_on": res[True][name]["Bias"],
                "bias_off": res[False][name]["Bias"],
                "abs_bias_on": b_on, "abs_bias_off": b_off,
                "abs_bias_change_pct": d_bias,
                "mae_return_on": mae_on, "mae_return_off": mae_off,
                "mae_return_change_pct": d_mae,
                "stdratio_on": res[True][name]["StdRatio"],
                "stdratio_off": res[False][name]["StdRatio"],
                "rho_on": res[True][name]["Rho"],
                "rho_off": res[False][name]["Rho"],
                "mae_baht_on": res[True][name]["MAE_baht"],
                "mae_baht_off": res[False][name]["MAE_baht"],
            })

    out = pd.DataFrame(records)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "ablation_h1.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("  สรุปผลการทดสอบ H1")
    print("=" * 78)
    for ticker in TICKERS:
        sub = out[out.stock == ticker]
        print(f"\n  {ticker}  (ราคาหลุดกรอบ train "
              f"{sub['out_of_range_pct'].iloc[0]:.2f}%)")
        print(f"    |Bias| เปลี่ยนเฉลี่ย {sub['abs_bias_change_pct'].mean():+.1f}%  "
              f"(ต่ำสุด {sub['abs_bias_change_pct'].min():+.1f}%, "
              f"สูงสุด {sub['abs_bias_change_pct'].max():+.1f}%)")
        print(f"    MAE_return เปลี่ยนเฉลี่ย {sub['mae_return_change_pct'].mean():+.2f}%")

    print(f"\n[save] {path}")
    print("\nเสร็จสิ้น")
    return out


if __name__ == "__main__":
    main()
