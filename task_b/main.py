"""
main.py — งาน B (ราคาปิด / return)
==================================
ไฟล์หลักของงาน B เท่านั้น รันไฟล์นี้ไฟล์เดียวจบ

    cd task_b
    python main.py

โฟลเดอร์นี้เป็นอิสระจาก task_a / task_c โดยสมบูรณ์
(feature / target / โมเดล / config แยกกันคนละชุด)

ลำดับการทำงาน:
  1. โหลดข้อมูล
  2. สร้าง feature (shift แล้ว) + target (return)
  3. ตรวจ leak เชิงโครงสร้าง
  4. วิเคราะห์ข้อมูลก่อนเทรน
  5. เทรน 3 โมเดล (ทำนายผ่าน return แล้วแปลงกลับเป็นราคา) + เทียบ baseline
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

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from config import TICKERS, OUTPUT_DIR
from data_loader import load_stock
from features import build_features, verify_no_leak
from targets import build_targets
from diagnostics import run_all_diagnostics
from splits import chronological_split
from models import get_regressors
from baselines import get_regression_baselines
from evaluate import regression_metrics, results_table, print_table, compare_to_baseline


def _prepare(X, y, extra=None):
    """จัด X และ y ให้ index ตรงกัน แล้วตัดแถวที่มี NaN ออก"""
    idx = X.index.intersection(y.dropna().index)
    X = X.loc[idx]
    y = y.loc[idx]

    ok = X.isna().mean(axis=1) < 0.5
    X, y = X[ok], y[ok]

    if extra is not None:
        extra = extra.loc[X.index]
        return X, y, extra
    return X, y


def run_task_b(X, targets, verbose=True, dev=False):
    print("\n" + "=" * 78)
    print("  งาน B : ทำนายราคาปิด (ทำนายผ่าน return แล้วแปลงกลับ)")
    if dev:
        print("  (โหมด dev — ยังไม่แตะ test)")
    print("=" * 78)

    y = targets["y_return"]
    extra = targets[["prev_close", "close"]]
    X_, y_, extra_ = _prepare(X, y, extra)

    parts = chronological_split(X_, y_, verbose=verbose, name="งาน B")
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]
    prev_close_val = extra_.loc[X_val.index, "prev_close"]

    if not dev:
        X_test, y_test = parts["test"]
        prev_close_test = extra_.loc[X_test.index, "prev_close"]

    # ห่อด้วย TransformedTargetRegressor เพื่อ scale target
    # (return มีขนาดเล็กมาก ~0.01 ANN จะเทรนไม่ดีถ้าไม่ scale)
    val_results, val_preds = {}, {}
    test_results, test_preds = {}, {}

    for name, model in get_regressors().items():
        if verbose:
            print(f"    เทรน {name} ...", end=" ", flush=True)
        wrapped = TransformedTargetRegressor(
            regressor=model, transformer=StandardScaler()
        )
        wrapped.fit(X_train, y_train)

        y_val_pred = wrapped.predict(X_val)
        val_results[name] = regression_metrics(y_val, y_val_pred, prev_close_val)
        val_preds[name] = y_val_pred

        if not dev:
            y_test_pred = wrapped.predict(X_test)
            test_results[name] = regression_metrics(y_test, y_test_pred, prev_close_test)
            test_preds[name] = y_test_pred

        if verbose:
            if dev:
                print(f"เสร็จ (val MAE={val_results[name]['MAE_baht']:.4f} บาท)")
            else:
                print(f"เสร็จ (val MAE={val_results[name]['MAE_baht']:.4f} บาท, "
                      f"test MAE={test_results[name]['MAE_baht']:.4f} บาท)")

    best = min(val_results, key=lambda n: val_results[n]["MAE_baht"])
    if verbose:
        print(f"\n    เลือกโมเดลที่ดีที่สุดจาก val set: {best} "
              f"(val MAE_baht={val_results[best]['MAE_baht']:.4f} บาท)")

    if dev:
        eval_split, y_eval, eval_results, eval_preds, prev_close_eval = (
            "val", y_val, val_results, val_preds, prev_close_val
        )
    else:
        eval_split, y_eval, eval_results, eval_preds, prev_close_eval = (
            "test", y_test, test_results, test_preds, prev_close_test
        )

    for name, p in get_regression_baselines(y_train, y_eval).items():
        eval_results[name] = regression_metrics(y_eval, p, prev_close_eval)
        eval_preds[name] = p

    df = results_table(eval_results, sort_by="MAE_baht", ascending=True)
    label = "Val Set (โหมด dev)" if dev else "Test Set"
    print_table(df, f"งาน B : ผลลัพธ์บน {label}")

    print("\n" + compare_to_baseline(df, "MAE_baht", higher_is_better=False,
                                     model_name=best))

    print("\n  หมายเหตุการอ่านผล:")
    print("  - R2_price ที่สูงมาก (>0.95) ไม่ได้แปลว่าโมเดลเก่ง")
    print("    เพราะมันมาจากการที่ราคาพรุ่งนี้ใกล้เคียงราคาวันนี้อยู่แล้ว")
    print("  - ให้ดู MAE_baht เทียบกับ Baseline: Naive (RW) เป็นหลัก")
    print("  - DirAcc (ทายทิศทางถูกกี่ %) มีความหมายกว่า R2 มาก")

    best_rmse = float(eval_results[best]["RMSE_baht"])

    return {"table": df, "preds": eval_preds, "y_test": y_eval,
            "prev_close_test": prev_close_eval,
            "best_rmse_baht": best_rmse,
            "test_index": y_eval.index, "best_model": best, "stage": eval_split}


def run_one_ticker(ticker, dev=False):
    print("\n\n" + "#" * 78)
    print(f"#  หุ้น: {ticker}  (งาน B)")
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

    res = run_task_b(X, targets, dev=dev)
    return {"ticker": ticker, "diag": diag, "b": res}


def save_results(all_results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for r in all_results:
        t = r["ticker"].replace(".", "_").replace("^", "")
        path = os.path.join(OUTPUT_DIR, f"{t}_taskB_price.csv")
        r["b"]["table"].to_csv(path, encoding="utf-8-sig")
    print(f"\n[main] บันทึกตารางผลลัพธ์ไว้ที่โฟลเดอร์ '{OUTPUT_DIR}/'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="รันงาน B (ราคาปิด) — ปกติจะแตะ test ครั้งเดียวตอนจบ"
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
    print("  งาน B : ทำนายราคาปิด (Regression)")
    if args.dev:
        print("  *** โหมด dev: ไม่แตะ test, ไม่บันทึกผล ***")
    print("=" * 78)

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
