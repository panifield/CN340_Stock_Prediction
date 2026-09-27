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

import pandas as pd

# ไม่ suppress warning (E1): อาจซ่อน ConvergenceWarning ของ MLP ที่ต้องรู้
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from config import (
    TICKERS, OUTPUT_DIR, SPLIT_BY_DATE, LOCK_PATH, REQUIRED_LOCK_FIELDS,
)
from data_loader import load_stock
from features import build_features, verify_no_leak
from targets import build_targets
from diagnostics import run_all_diagnostics
from splits import chronological_split, prepare_xy
from models import get_regressors
from baselines import get_regression_baselines
from evaluate import (
    regression_metrics, results_table, print_table, compare_to_baseline,
    prediction_shape,
)

PRIMARY_METRIC = "MAE_return"


def verify_lock(path=LOCK_PATH):
    """ตรวจว่า PRE_TEST_LOCK.md มีอยู่และมีหัวข้อครบ ก่อนอนุญาตให้เปิด test (0.5)"""
    if not path.exists():
        raise RuntimeError(
            f"ปฏิเสธการเปิด test set: ไม่พบ {path}\n"
            "ต้องเขียนไฟล์ล็อกให้เสร็จก่อนจึงจะเปิด test ได้\n"
            ">>> ถ้าต้องการรันเพื่อพัฒนา ให้ใช้:  python main.py --dev"
        )
    text = path.read_text(encoding="utf-8")

    missing, empty = [], []
    for field in REQUIRED_LOCK_FIELDS:
        if field not in text:
            missing.append(field)
            continue
        # ตรวจว่ามีค่าอยู่หลัง ":" จริง ไม่ใช่หัวข้อเปล่า
        value = text.split(field, 1)[1].splitlines()[0].strip()
        if not value:
            empty.append(field)

    if missing or empty:
        raise RuntimeError(
            f"ปฏิเสธการเปิด test set: {path.name} ยังไม่สมบูรณ์\n"
            f"  ขาดหัวข้อ : {missing}\n"
            f"  หัวข้อว่าง: {empty}\n"
            "ไฟล์ล็อกที่ไม่ครบ = ยังไม่ได้ประกาศสิ่งที่จะรายงานล่วงหน้าจริง"
        )
    print(f"[lock] ผ่านการตรวจ: {path.name} มีหัวข้อครบ {len(REQUIRED_LOCK_FIELDS)} ข้อ")


def run_task_b(X, targets, use_test, verbose=True):
    dev = not use_test
    print("\n" + "=" * 78)
    print("  งาน B : ทำนายราคาปิด (ทำนายผ่าน return แล้วแปลงกลับ)")
    if dev:
        print("  (โหมด dev — ไม่มี test rows ใน pipeline)")
    print("=" * 78)

    y = targets["y_return"]
    extra = targets[["prev_close", "close"]]
    X_, y_, extra_ = prepare_xy(X, y, extra)

    parts = chronological_split(X_, y_, verbose=verbose, name="งาน B",
                                include_test=use_test)
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]
    prev_close_val = extra_.loc[X_val.index, "prev_close"]

    # ต้องอยู่หลัง split เสมอ และใช้ index ของ train เท่านั้น (A3)
    diag = run_all_diagnostics(targets.loc[X_train.index], X_train)

    if use_test:
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
            print(f"เสร็จ (val {PRIMARY_METRIC}="
                  f"{val_results[name][PRIMARY_METRIC]:.6f})")

    # เลือกด้วย MAE_return (A4): target ของโมเดลคือ return และ scale-normalized
    # MAE_baht เก็บไว้เป็น secondary interpretability metric
    best = min(val_results, key=lambda n: val_results[n][PRIMARY_METRIC])
    if verbose:
        print(f"\n    เลือกโมเดลที่ดีที่สุดจาก val set: {best} "
              f"(val {PRIMARY_METRIC}={val_results[best][PRIMARY_METRIC]:.6f})")

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

    df = results_table(eval_results, sort_by=PRIMARY_METRIC, ascending=True)
    label = "Val Set (โหมด dev)" if dev else "Test Set"
    print_table(df, f"งาน B : ผลลัพธ์บน {label}")

    print("\n" + compare_to_baseline(df, PRIMARY_METRIC, model_name=best))

    print("\n  หมายเหตุการอ่านผล:")
    print("  - ตัดสินด้วย MAE_return เทียบกับ Baseline: Naive (RW) เป็นหลัก")
    print("  - MAE_baht / RMSE_baht ใช้อธิบายขนาด error เป็นบาทเท่านั้น")
    print("  - R2_return < 0 แปลว่าแย่กว่าการทายค่าเฉลี่ยของชุดนั้น")

    # ตารางที่ 2 (§2) -- diagnostic แยกจากตารางหลัก พิมพ์อย่างเดียว ไม่ลง CSV
    # ไม่ถูกใช้เลือกโมเดล (best เลือกด้วย MAE_return ไปแล้วด้านบน)
    shape_df = results_table(
        {name: prediction_shape(y_eval, p) for name, p in eval_preds.items()}
    )
    print_table(shape_df.loc[df.index],       # เรียงลำดับเดียวกับตารางหลัก
                "งาน B : รูปร่างการทำนาย (diagnostic -- ไม่ใช้เลือกโมเดล)")
    print("\n  อ่านตารางนี้อย่างไร:")
    print("  - StdRatio ใกล้ 0 = โมเดลยุบเป็นค่าคงที่ (ถึงจะได้ MAE ดีก็ไม่นับว่าทำนายได้)")
    print("  - ตารางนี้ไม่ถูกใช้เลือกโมเดล -- การเลือกใช้ MAE_return เท่านั้น")

    return {"table": df, "preds": eval_preds, "y_eval": y_eval,
            "prev_close_eval": prev_close_eval, "diag": diag,
            "eval_index": y_eval.index, "best_model": best, "stage": eval_split}


def run_one_ticker(ticker, use_test):
    print("\n\n" + "#" * 78)
    print(f"#  หุ้น: {ticker}  (งาน B)")
    if not use_test:
        print("#  โหมด dev — ใช้แค่ train/val เพื่อพัฒนา ไม่แตะ test")
    print("#" * 78)

    df = load_stock(ticker)

    if not use_test:
        # ตัดตั้งแต่ก่อนสร้าง feature -- test rows ไม่เคยเข้าสู่ pipeline (0.2)
        # ปลอดภัยเพราะ feature ทุกตัวมองย้อนหลังอย่างเดียว (rolling / shift)
        n_before = len(df)
        df = df.loc[:SPLIT_BY_DATE["val_end"]].copy()
        print(f"[main] โหมด dev: ตัด test rows ทิ้ง {n_before - len(df)} แถว "
              f"เหลือ {len(df)} แถว (ถึง {df.index[-1].date()})")

    X = build_features(df)
    targets = build_targets(df)

    verify_no_leak(df, X, sample_idx=100)

    res = run_task_b(X, targets, use_test=use_test)
    return {"ticker": ticker, "b": res}


def save_results(all_results):
    """บันทึกตารางของ split ที่ประเมิน: --dev -> *_val.csv / โหมดปกติ -> *_test.csv"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for r in all_results:
        t = r["ticker"].replace(".", "_").replace("^", "")
        path = OUTPUT_DIR / f"{t}_taskB_{r['b']['stage']}.csv"
        r["b"]["table"].to_csv(path, encoding="utf-8-sig")
        print(f"[main] บันทึก {path.relative_to(OUTPUT_DIR.parent)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="รันงาน B (ราคาปิด) — โหมดปกติจะเปิด test ซึ่งต้องผ่าน lock ก่อน"
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="โหมดพัฒนา: เทรน+ประเมินบน train/val เท่านั้น ไม่แตะ test",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_test = not args.dev      # มีแค่ flag --dev ไม่มี --test (ดูข้อ 0.4)

    if use_test:
        verify_lock()            # เฟส 1 ยังไม่มีไฟล์ lock -> หยุดที่บรรทัดนี้

    print("=" * 78)
    print("  งาน B : ทำนายราคาปิด (Regression)")
    if not use_test:
        print("  *** โหมด dev: ไม่แตะ test ***")
    print("=" * 78)

    # fail fast (E3): หุ้นไหนพัง -> หยุดทันที ไม่พิมพ์ "เสร็จสิ้น" หลอกๆ
    all_results = [run_one_ticker(ticker, use_test=use_test) for ticker in TICKERS]

    save_results(all_results)
    print("\nเสร็จสิ้น")
    return all_results


if __name__ == "__main__":
    main()
