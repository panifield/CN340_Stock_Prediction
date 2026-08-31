"""
main.py — งาน A (คู่/คี่)
========================
ไฟล์หลักของงาน A เท่านั้น รันไฟล์นี้ไฟล์เดียวจบ

    cd task_a
    python main.py

โฟลเดอร์นี้เป็นอิสระจาก task_b / task_c โดยสมบูรณ์
(feature / target / โมเดล / config แยกกันคนละชุด)

ลำดับการทำงาน:
  1. โหลดข้อมูล
  2. สร้าง feature (shift แล้ว) + target (parity)
  3. ตรวจ leak เชิงโครงสร้าง
  4. วิเคราะห์ข้อมูลก่อนเทรน
  5. เทรน 3 โมเดล + เทียบ baseline
  6. บันทึกผลลง csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

from config import TICKERS, OUTPUT_DIR
from data_loader import load_stock
from features import build_features, verify_no_leak
from targets import build_targets
from diagnostics import run_all_diagnostics
from splits import chronological_split
from models import get_classifiers
from baselines import get_classification_baselines
from evaluate import (
    classification_metrics, results_table, print_table,
    print_confusion, compare_to_baseline,
)
import rounding


def _prepare(X, y, extra=None):
    """จัด X และ y ให้ index ตรงกัน แล้วตัดแถวที่มี NaN ออก"""
    idx = X.index.intersection(y.dropna().index)
    X = X.loc[idx]
    y = y.loc[idx]

    # ตัดแถวที่ feature เป็น NaN เกินครึ่ง (ช่วงต้นที่ rolling ยังไม่ครบ)
    ok = X.isna().mean(axis=1) < 0.5
    X, y = X[ok], y[ok]

    if extra is not None:
        extra = extra.loc[X.index]
        return X, y, extra
    return X, y


def run_task_a(X, targets, verbose=True, dev=False):
    print("\n" + "=" * 78)
    print("  งาน A : ทำนายราคาปิดปัดเป็นจำนวนเต็มแล้ว เป็นเลขคู่หรือคี่")
    if dev:
        print("  (โหมด dev — ยังไม่แตะ test)")
    print("=" * 78)

    y = targets["y_parity"]
    extra = targets[["prev_parity", "prev_close", "close"]]
    X_, y_, extra_ = _prepare(X, y, extra)

    parts = chronological_split(X_, y_, verbose=verbose, name="งาน A")
    y_train = parts["train"][1]

    models = get_classifiers()
    X_train, _ = parts["train"]
    X_val, y_val = parts["val"]

    val_results, val_preds = {}, {}
    test_results, test_preds = {}, {}
    if not dev:
        X_test, y_test = parts["test"]

    for name, model in models.items():
        if verbose:
            print(f"    เทรน {name} ...", end=" ", flush=True)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        try:
            y_val_proba = model.predict_proba(X_val)[:, 1]
        except Exception:
            y_val_proba = None
        val_results[name] = classification_metrics(y_val, y_val_pred, y_val_proba)
        val_preds[name] = y_val_pred

        if not dev:
            y_test_pred = model.predict(X_test)
            try:
                y_test_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_test_proba = None
            test_results[name] = classification_metrics(y_test, y_test_pred, y_test_proba)
            test_preds[name] = y_test_pred

        if verbose:
            if dev:
                print(f"เสร็จ (val acc={val_results[name]['Accuracy']:.4f})")
            else:
                print(f"เสร็จ (val acc={val_results[name]['Accuracy']:.4f}, "
                      f"test acc={test_results[name]['Accuracy']:.4f})")

    best = max(val_results, key=lambda n: val_results[n]["Accuracy"])
    if verbose:
        print(f"\n    เลือกโมเดลที่ดีที่สุดจาก val set: {best} "
              f"(val Accuracy={val_results[best]['Accuracy']:.4f})")

    eval_split = "val" if dev else "test"
    X_eval, y_eval = parts[eval_split]
    eval_results = val_results if dev else test_results
    eval_preds = val_preds if dev else test_preds
    prev_parity_eval = extra_.loc[X_eval.index, "prev_parity"]

    base_preds = get_classification_baselines(
        y_train, y_eval, prev_values=prev_parity_eval.fillna(0)
    )
    for name, p in base_preds.items():
        eval_results[name] = classification_metrics(y_eval, p)

    df = results_table(eval_results, sort_by="Accuracy", ascending=False)
    label = "Val Set (โหมด dev)" if dev else "Test Set"
    print_table(df, f"งาน A : ผลลัพธ์บน {label}")

    print("\n" + compare_to_baseline(df, "Accuracy", higher_is_better=True,
                                     model_name=best))

    print_confusion(y_eval, eval_preds[best], labels=("คู่", "คี่"),
                    title=f"({best}, เลือกจาก val)")

    return {"table": df, "preds": eval_preds, "y_test": y_eval,
            "test_index": X_eval.index, "best_model": best, "stage": eval_split}


def run_one_ticker(ticker, dev=False):
    print("\n\n" + "#" * 78)
    print(f"#  หุ้น: {ticker}  (งาน A)")
    if dev:
        print("#  โหมด dev — ใช้แค่ train/val เพื่อพัฒนา ยังไม่แตะ test")
    print("#" * 78)

    df = load_stock(ticker)
    X = build_features(df)
    targets = build_targets(df)

    verify_no_leak(df, X, sample_idx=100)

    if dev:
        from config import TRAIN_RATIO, VAL_RATIO, SPLIT_BY_DATE
        if SPLIT_BY_DATE is not None:
            val_end = pd.Timestamp(SPLIT_BY_DATE["val_end"])
            cutoff = int((df.index <= val_end).sum())
        else:
            cutoff = int(len(df) * (TRAIN_RATIO + VAL_RATIO))
        print(f"\n[main] โหมด dev: diagnostics ใช้แค่ train+val "
              f"({cutoff}/{len(df)} แถวแรก) ตัด test ออก")
        diag = run_all_diagnostics(df.iloc[:cutoff], targets.iloc[:cutoff],
                                   X.iloc[:cutoff])
    else:
        diag = run_all_diagnostics(df, targets, X)

    res = run_task_a(X, targets, dev=dev)
    return {"ticker": ticker, "diag": diag, "a": res}


def save_results(all_results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for r in all_results:
        t = r["ticker"].replace(".", "_").replace("^", "")
        path = os.path.join(OUTPUT_DIR, f"{t}_taskA_parity.csv")
        r["a"]["table"].to_csv(path, encoding="utf-8-sig")
    print(f"\n[main] บันทึกตารางผลลัพธ์ไว้ที่โฟลเดอร์ '{OUTPUT_DIR}/'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="รันงาน A (คู่/คี่) — ปกติจะแตะ test ครั้งเดียวตอนจบ"
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="โหมดพัฒนา: เทรน+ประเมินบน train/val เท่านั้น "
             "ไม่แตะ test ไม่บันทึกผล",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 78)
    print("  งาน A : ทำนายคู่/คี่ (Classification)")
    if args.dev:
        print("  *** โหมด dev: ไม่แตะ test, ไม่บันทึกผล ***")
    print("=" * 78)

    rounding.self_test()

    all_results = []
    for ticker in TICKERS:
        try:
            all_results.append(run_one_ticker(ticker, dev=args.dev))
        except Exception as e:
            print(f"\n!! {ticker} รันไม่ผ่าน: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    if all_results and not args.dev:
        save_results(all_results)
    elif args.dev:
        print("\n[main] โหมด dev เสร็จแล้ว — ไม่บันทึก csv")

    print("\nเสร็จสิ้น")
    return all_results


if __name__ == "__main__":
    main()
