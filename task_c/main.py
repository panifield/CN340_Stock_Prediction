"""
main.py — งาน C (ขึ้น/ลง)
========================
ไฟล์หลักของงาน C เท่านั้น รันไฟล์นี้ไฟล์เดียวจบ

    cd task_c
    python main.py

โฟลเดอร์นี้เป็นอิสระจาก task_a / task_b โดยสมบูรณ์
(feature / target / โมเดล / config แยกกันคนละชุด)

ลำดับการทำงาน:
  1. โหลดข้อมูล
  2. สร้าง feature (shift แล้ว) + target (ขึ้น/ลง)
  3. ตรวจ leak เชิงโครงสร้าง
  4. วิเคราะห์ข้อมูลก่อนเทรน
  5. เทรน 3 โมเดล + เทียบ baseline
  6. บันทึกผลลง csv
"""

import argparse
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

from config import (
    TICKERS, OUTPUT_DIR, SUSPICIOUS_ACCURACY, FEATURE_SELECTION_METHOD,
)
from data_loader import load_stock
from features import build_features, verify_no_leak
from targets import build_targets
from diagnostics import run_all_diagnostics
from splits import chronological_split
from models import get_classifiers
from feature_selection import VALID_METHODS, select_feature_names
from baselines import get_classification_baselines
from loss_function import binary_cross_entropy, l2_penalty
from evaluate import (
    classification_metrics, results_table, print_table,
    print_confusion, compare_to_baseline,
)


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


def _ann_loss_report(pipeline, X_train, y_train, y_val, y_val_proba):
    """คืนค่า loss ของ ANN หลัง fit เพื่อรายงานผลการเทรนจริง.

    train objective = train BCE + L2 penalty
    validation รายงานเฉพาะ BCE เพราะ validation ไม่ควรบวก penalty ของ model.
    """
    ann = pipeline.named_steps["model"]
    train_proba = pipeline.predict_proba(X_train)[:, 1]
    train_bce = binary_cross_entropy(y_train, train_proba)
    val_bce = binary_cross_entropy(y_val, y_val_proba)
    l2_term = ann.alpha * l2_penalty(ann.coefs_) / (2 * len(y_train))
    return {
        "train_bce": train_bce,
        "l2_penalty": l2_term,
        "train_objective": train_bce + l2_term,
        "val_bce": val_bce,
        # ค่า loss ของ iteration สุดท้ายที่ sklearn เก็บไว้
        "sklearn_last_loss": float(ann.loss_),
        "n_iter": int(ann.n_iter_),
    }


def _validation_prediction_table(index, y_true, y_pred, y_proba):
    """สร้างตารางผลทำนาย validation ที่อ่านง่ายสำหรับโหมด dev."""
    table = pd.DataFrame({
        "Actual": y_true.astype(int).to_numpy(),
        "Probability_Up": y_proba,
        "Predicted": np.asarray(y_pred, dtype=int),
    }, index=index)
    table.index.name = "Date"
    table["Correct"] = (table["Actual"] == table["Predicted"])
    table["Actual_Label"] = table["Actual"].map({0: "ลง/นิ่ง", 1: "ขึ้น"})
    table["Predicted_Label"] = table["Predicted"].map({0: "ลง/นิ่ง", 1: "ขึ้น"})
    return table[["Actual_Label", "Probability_Up", "Predicted_Label", "Correct"]]


def run_task_c(X, targets, drop_flat=False, verbose=True, dev=False,
               feature_selection_method=FEATURE_SELECTION_METHOD):
    print("\n" + "=" * 78)
    print("  งาน C : ทำนายว่าราคาปิดขึ้นหรือลงจากเมื่อวาน")
    if dev:
        print("  (โหมด dev — ยังไม่แตะ test)")
    print("=" * 78)

    y = targets["y_updown"]
    extra = targets[["is_flat", "prev_close"]]
    X_, y_, extra_ = _prepare(X, y, extra)

    if drop_flat:
        mask = extra_["is_flat"] == 0
        print(f"    ตัดวันราคานิ่งออก {(~mask).sum()} แถว")
        X_, y_, extra_ = X_[mask], y_[mask], extra_[mask]

    # persistence: ทิศทางเมื่อวาน
    prev_updown = y_.shift(1)

    parts = chronological_split(X_, y_, verbose=verbose, name="งาน C")
    y_train = parts["train"][1]
    X_train, _ = parts["train"]
    X_val, y_val = parts["val"]
    if not dev:
        X_test, y_test = parts["test"]

    # Feature selection fit บน train เท่านั้น; X_ ดั้งเดิมไม่ถูกแก้ไข
    selected_features, feature_ranking = select_feature_names(
        X_train, y_train, feature_selection_method
    )
    if feature_selection_method != "none":
        parts = {
            split_name: (X_part.loc[:, selected_features], y_part)
            for split_name, (X_part, y_part) in parts.items()
        }
        X_train, y_train = parts["train"]
        X_val, y_val = parts["val"]
        if not dev:
            X_test, y_test = parts["test"]
        if verbose:
            print(f"[feature selection] {feature_selection_method}: "
                  f"เลือก {len(selected_features)}/{X_.shape[1]} features "
                  "จาก train set เท่านั้น")
            print("    " + ", ".join(selected_features))

    models = get_classifiers()
    val_results, val_preds, val_probas = {}, {}, {}
    test_results, test_preds = {}, {}
    ann_losses = {}

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
        val_probas[name] = y_val_proba

        if name == "ANN (MLP)":
            ann_losses["val"] = _ann_loss_report(
                model, X_train, y_train, y_val, y_val_proba
            )

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

            if name == "ANN (MLP)":
                loss = ann_losses["val"]
                print("      ANN loss: "
                      f"train BCE={loss['train_bce']:.4f} + "
                      f"L2={loss['l2_penalty']:.4f} "
                      f"= objective {loss['train_objective']:.4f}; "
                      f"val BCE={loss['val_bce']:.4f} "
                      f"(iteration {loss['n_iter']}, "
                      f"sklearn loss ล่าสุด={loss['sklearn_last_loss']:.4f})")

    best = max(val_results, key=lambda n: val_results[n]["Accuracy"])
    if verbose:
        print(f"\n    เลือกโมเดลที่ดีที่สุดจาก val set: {best} "
              f"(val Accuracy={val_results[best]['Accuracy']:.4f})")

    eval_split = "val" if dev else "test"
    X_eval, y_eval = parts[eval_split]
    eval_results = val_results if dev else test_results
    eval_preds = val_preds if dev else test_preds

    validation_predictions = None
    if dev:
        # เก็บ prediction ครบทุกแถวของโมเดล ML ทั้ง 3 ตัวเพื่อ export เป็น CSV
        # ไม่พิมพ์ใน terminal เพราะหุ้นละ 365 แถวต่อโมเดล
        validation_predictions = {}
        for name in models:
            table = _validation_prediction_table(
                X_val.index, y_val, val_preds[name], val_probas[name]
            )
            validation_predictions[name] = table

    base_preds = get_classification_baselines(
        y_train, y_eval,
        prev_values=prev_updown.loc[X_eval.index].fillna(1),
        include_always_up=True,
    )
    for name, p in base_preds.items():
        eval_results[name] = classification_metrics(y_eval, p)

    df = results_table(eval_results, sort_by="Accuracy", ascending=False)
    label = "Val Set (โหมด dev)" if dev else "Test Set"
    print_table(df, f"งาน C : ผลลัพธ์บน {label}")

    print("\n" + compare_to_baseline(df, "Accuracy", higher_is_better=True,
                                     model_name=best))

    print_confusion(y_eval, eval_preds[best], labels=("ลง", "ขึ้น"),
                    title=f"({best}, เลือกจาก val)")

    max_acc = df[~df.index.str.startswith("Baseline")]["Accuracy"].max()
    if max_acc > SUSPICIOUS_ACCURACY:
        print(f"\n  !! เตือน: accuracy {max_acc:.4f} สูงเกิน "
              f"{SUSPICIOUS_ACCURACY} ผิดปกติสำหรับงานนี้")
        print("     ให้กลับไปตรวจว่ามี data leakage หรือไม่")

    return {"table": df, "preds": eval_preds, "y_test": y_eval,
            "test_index": X_eval.index, "best_model": best, "stage": eval_split,
            "ann_losses": ann_losses, "selected_features": selected_features,
            "feature_ranking": feature_ranking,
            "feature_selection_method": feature_selection_method,
            "validation_predictions": validation_predictions}


def run_one_ticker(ticker, dev=False,
                   feature_selection_method=FEATURE_SELECTION_METHOD):
    print("\n\n" + "#" * 78)
    print(f"#  หุ้น: {ticker}  (งาน C)")
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

    res = run_task_c(X, targets, dev=dev,
                     feature_selection_method=feature_selection_method)
    return {"ticker": ticker, "diag": diag, "c": res}


def save_results(all_results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for r in all_results:
        t = r["ticker"].replace(".", "_").replace("^", "")
        method = r["c"]["feature_selection_method"]
        suffix = "" if method == "none" else f"_{method}"
        path = os.path.join(OUTPUT_DIR, f"{t}_taskC_updown{suffix}.csv")
        r["c"]["table"].to_csv(path, encoding="utf-8-sig")
    print(f"\n[main] บันทึกตารางผลลัพธ์ไว้ที่โฟลเดอร์ '{OUTPUT_DIR}/'")


def save_validation_predictions(all_results):
    """บันทึก prediction จาก dev เป็น 3 CSV แยกตามโมเดล รวมทุกหุ้น.

    ใช้เฉพาะ validation prediction ที่สร้างใน run_one_ticker จึงไม่อ่านหรือ
    บันทึกข้อมูลจาก test set. Timestamp เดียวกันทำให้รู้ว่า 3 ไฟล์มาจาก run เดียวกัน.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    by_model = {}

    for result in all_results:
        predictions = result["c"].get("validation_predictions") or {}
        for model_name, table in predictions.items():
            frame = table.reset_index()
            frame.insert(0, "Ticker", result["ticker"])
            frame.insert(1, "Model", model_name)
            by_model.setdefault(model_name, []).append(frame)

    paths = []
    for model_name, frames in by_model.items():
        safe_name = (model_name.lower().replace(" ", "_")
                     .replace("(", "").replace(")", ""))
        path = os.path.join(
            OUTPUT_DIR, f"{safe_name}_validation_predictions_{run_at}.csv"
        )
        pd.concat(frames, ignore_index=True).to_csv(path, index=False,
                                                     encoding="utf-8-sig")
        paths.append(path)

    if paths:
        print("\n[main] บันทึก prediction ของ validation set แล้ว:")
        for path in paths:
            print(f"       {path}")
    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="รันงาน C (ขึ้น/ลง) — ปกติจะแตะ test ครั้งเดียวตอนจบ"
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="โหมดพัฒนา: เทรน+ประเมินบน train/val เท่านั้น "
             "ไม่แตะ test; บันทึก prediction validation แยกตามโมเดล",
    )
    parser.add_argument(
        "--feature-selection", choices=sorted(VALID_METHODS),
        default=FEATURE_SELECTION_METHOD,
        help="เลือก feature จาก train set: none=ใช้ครบ 33 (ค่าเดิม), "
             "l1=L1 Logistic Regression, top_k=mutual information Top-K",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 78)
    print("  งาน C : ทำนายขึ้น/ลง (Classification)")
    if args.dev:
        print("  *** โหมด dev: ไม่แตะ test; บันทึก prediction validation ***")
    print("=" * 78)

    all_results = []
    for ticker in TICKERS:
        try:
            all_results.append(run_one_ticker(
                ticker, dev=args.dev,
                feature_selection_method=args.feature_selection,
            ))
        except Exception as e:
            print(f"\n!! {ticker} รันไม่ผ่าน: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    if all_results and not args.dev:
        save_results(all_results)
    elif all_results and args.dev:
        save_validation_predictions(all_results)

    print("\nเสร็จสิ้น")
    return all_results


if __name__ == "__main__":
    main()
