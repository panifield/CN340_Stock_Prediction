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
  2. สร้าง feature (shift แล้ว) + target (y_flip + y_parity)
  3. ตรวจ leak เชิงโครงสร้าง
  4. วิเคราะห์ข้อมูลก่อนเทรน
  5. Walk-forward CV บน train+val -> เลือกโมเดลที่ดีที่สุด (ไม่แตะ test)
  6. ตรวจ overfit (train vs val เดี่ยว)
  7. [เฉพาะไม่ใช่ --dev] refit โมเดลที่ชนะบน train+val แล้วยิง test ครั้งเดียว
  8. บันทึกผลลง csv
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
from splits import chronological_split, walk_forward_splits
from models import get_classifiers
from baselines import get_classification_baselines
from evaluate import (
    classification_metrics, results_table, print_table, print_confusion,
)
from eval_stats import verdict, full_report, permutation_test
import rounding


def _prepare(X, y, extra=None):
    """จัด X และ y ให้ index ตรงกัน แล้วตัดแถวที่มี NaN ออก"""
    idx = X.index.intersection(y.dropna().index)
    X = X.loc[idx]
    y = y.loc[idx]

    # vol_ratio_prev = log(volume/MA) ถ้า volume=0 บางวันจะได้ -inf ซึ่ง
    # SimpleImputer จัดการไม่ได้ (ต่างจาก NaN) — เดิมไม่เคยเจอเพราะ
    # data_cache/ แช่แข็งช่วงที่ไม่โดนวันแบบนี้พอดี แต่ live data (ช่วง
    # เวลาเปลี่ยนไปเรื่อยๆ) เจอแน่นอนสักวัน แปลงเป็น NaN ไว้กันล่วงหน้า
    X = X.replace([np.inf, -np.inf], np.nan)

    # ตัดแถวที่ feature เป็น NaN เกินครึ่ง (ช่วงต้นที่ rolling ยังไม่ครบ)
    ok = X.isna().mean(axis=1) < 0.5
    X, y = X[ok], y[ok]

    if extra is not None:
        extra = extra.loc[X.index]
        return X, y, extra
    return X, y


def _print_logreg_coefs(pipeline, feature_names, top_n=8):
    """
    พิมพ์ coefficient ของ Logistic Regression (ไม่ใช่เพื่อชนะ accuracy
    แต่ไว้ดูว่า feature ไหนมีผลจริงต่อการพลิก parity)
    """
    coefs = pipeline.named_steps["model"].coef_[0]
    s = pd.Series(coefs, index=feature_names).sort_values(
        key=lambda x: x.abs(), ascending=False)
    print(f"\n    Logistic Regression coefficients (top {top_n} โดย |ค่า|):")
    for feat, c in s.head(top_n).items():
        print(f"        {feat:35s} {c:+.4f}")


def _run_permutation_check(name, model, X_train, yflip_train, X_eval, yflip_eval,
                           split_name="", n_permutations=100):
    """
    Permutation test บน y_flip (ตัวที่โมเดลเทรนจริง) ของโมเดลที่เลือก
    มาแล้วเท่านั้น (ถูกเลือกจาก walk-forward OOF ไม่ใช่จาก test เอง)
    ถ้าผลจริงตกในช่วง null distribution = ยืนยันว่าไม่มี signal
    เกินกว่าที่ walk-forward บอกไว้แล้ว + ไม่มี leak แอบแฝง
    """
    print(f"\n  --- Permutation test บน y_flip ({name}, {split_name}) ---")

    def fit_fn(Xtr, ytr, Xva, _model=model):
        _model.fit(Xtr, ytr)
        return _model.predict(Xva)

    res = permutation_test(fit_fn, X_train, yflip_train, X_eval, yflip_eval,
                           n_permutations=n_permutations, verbose=False)
    print(f"    observed={res['observed_acc']:.4f}  "
          f"null={res['null_mean']:.4f}±{res['null_std']:.4f}  "
          f"p={res['p_value']:.4f}  -> {res['interpretation']}")


def _reconstruct_parity(prev_parity, flip_pred, flip_proba=None):
    """
    แปลงคำทาย "พลิกหรือไม่" (y_flip) กลับเป็น parity เต็มบาทจริง
    parity_hat = (prev_parity + flip_pred) mod 2   <- deterministic เป๊ะ
    prev_parity มาจากราคาปิดจริงของเมื่อวาน ไม่ใช่ค่าที่ทาย จึงไม่ leak
    """
    prev_parity = np.asarray(prev_parity, dtype=float)
    flip_pred = np.asarray(flip_pred, dtype=float)
    parity_pred = (prev_parity + flip_pred) % 2

    proba1 = None
    if flip_proba is not None:
        flip_proba = np.asarray(flip_proba, dtype=float)
        # prev=0: parity=1 ก็ต่อเมื่อพลิก -> proba(parity=1)=proba(flip=1)
        # prev=1: parity=1 ก็ต่อเมื่อไม่พลิก -> proba(parity=1)=1-proba(flip=1)
        proba1 = np.where(prev_parity == 0, flip_proba, 1 - flip_proba)

    return parity_pred, proba1


def _walk_forward_eval(X_pool, yflip_pool, extra_, n_splits=5, min_train=250,
                       verbose=True):
    """
    Walk-forward CV บน train+val pool แล้วรวม out-of-fold prediction ของ
    ทุก fold เป็นชุดเดียว (n ~ ขนาด pool แทนที่จะเป็น val เดี่ยว 365 วัน)

    เหตุผล: val เดี่ยว 365 วันมี noise สูงมาก (CI กว้างเป็นสิบ percentage
    point) พอรวม out-of-fold ของหลาย fold เข้าด้วยกัน n ใหญ่ขึ้นหลายเท่า
    -> CI แคบลง เชื่อถือได้กว่าตอนใช้เลือกโมเดล/เทียบ baseline มาก
    ไม่แตะ test เลยตลอดขั้นตอนนี้
    """
    folds = list(walk_forward_splits(X_pool, n_splits=n_splits,
                                     min_train=min_train))
    model_names = list(get_classifiers().keys())
    baseline_names = ["Baseline: Majority", "Baseline: Persistence",
                      "Baseline: Markov(1)"]

    oof_true, oof_pred = [], {n: [] for n in model_names + baseline_names}
    oof_proba = {n: [] for n in model_names}

    for fold_i, (train_idx, test_idx) in enumerate(folds, 1):
        X_tr, X_te = X_pool.iloc[train_idx], X_pool.iloc[test_idx]
        yflip_tr = yflip_pool.iloc[train_idx]
        prev_parity_te = extra_.loc[X_te.index, "prev_parity"].fillna(0)
        parity_te = extra_.loc[X_te.index, "y_parity"]
        parity_tr = extra_.loc[X_tr.index, "y_parity"]
        oof_true.append(parity_te)

        fold_accs = []
        for name, model in get_classifiers().items():
            model.fit(X_tr, yflip_tr)
            flip_pred = model.predict(X_te)
            try:
                flip_proba = model.predict_proba(X_te)[:, 1]
            except Exception:
                flip_proba = None
            parity_pred, parity_proba = _reconstruct_parity(
                prev_parity_te, flip_pred, flip_proba)
            oof_pred[name].append(pd.Series(parity_pred, index=X_te.index))
            if parity_proba is not None:
                oof_proba[name].append(pd.Series(parity_proba, index=X_te.index))
            fold_accs.append(f"{name}={np.mean(parity_pred == parity_te.values):.3f}")

        base = get_classification_baselines(
            parity_tr, parity_te, prev_values=prev_parity_te.fillna(0))
        for bname, p in base.items():
            oof_pred[bname].append(pd.Series(p, index=X_te.index))

        if verbose:
            print(f"    fold {fold_i}/{len(folds)} "
                  f"({X_te.index[0].date()}~{X_te.index[-1].date()}, "
                  f"n={len(X_te)}): " + ", ".join(fold_accs))

    y_oof = pd.concat(oof_true)
    preds_oof = {n: pd.concat(lst) for n, lst in oof_pred.items()}
    probas_oof = {n: pd.concat(lst) for n, lst in oof_proba.items() if lst}
    return y_oof, preds_oof, probas_oof


def run_task_a(X, targets, verbose=True, dev=False):
    print("\n" + "=" * 78)
    print("  งาน A : ทำนายราคาปิดปัดเป็นจำนวนเต็มแล้ว เป็นเลขคู่หรือคี่")
    print("  (เทรนบน y_flip = พลิก parity หรือไม่ แล้ว reconstruct กลับ)")
    if dev:
        print("  (โหมด dev — ยังไม่แตะ test)")
    print("=" * 78)

    y_flip = targets["y_flip"]
    extra = targets[["prev_parity", "y_parity"]]
    X_, yflip_, extra_ = _prepare(X, y_flip, extra)

    parts = chronological_split(X_, yflip_, verbose=verbose, name="งาน A")
    X_train, yflip_train = parts["train"]
    X_val, yflip_val = parts["val"]
    dev_pool_X = pd.concat([X_train, X_val])
    dev_pool_yflip = pd.concat([yflip_train, yflip_val])

    # ---------- Walk-forward CV บน train+val pool (เลือกโมเดล ไม่แตะ test) ----------
    print(f"\n    Walk-forward CV บน train+val pool ({len(dev_pool_X)} แถว) ...")
    y_oof, preds_oof, probas_oof = _walk_forward_eval(
        dev_pool_X, dev_pool_yflip, extra_, verbose=verbose)

    wf_results = {name: classification_metrics(y_oof, pred, probas_oof.get(name))
                 for name, pred in preds_oof.items()}
    wf_df = results_table(wf_results, sort_by="Accuracy", ascending=False)
    print_table(wf_df, f"งาน A : Walk-forward Out-of-Fold (n={len(y_oof)})")

    model_names = ["ANN (MLP)", "LightGBM", "Logistic Regression"]
    best = max(model_names, key=lambda n: wf_results[n]["Accuracy"])
    best_baseline_name = max(
        (n for n in wf_df.index if n.startswith("Baseline")),
        key=lambda n: wf_df.loc[n, "Accuracy"],
    )
    print("\n" + verdict(wf_df.loc[best, "Accuracy"],
                         wf_df.loc[best_baseline_name, "Accuracy"],
                         n=len(y_oof)))
    full_report(y_oof, preds_oof, probabilities=probas_oof,
               baseline_name=best_baseline_name)
    print(f"\n    เลือกโมเดลที่ดีที่สุดจาก walk-forward OOF: {best} "
          f"(Accuracy={wf_results[best]['Accuracy']:.4f}, n={len(y_oof)})")

    # ---------- ตรวจ overfit ต่อโมเดล (train เดียว vs val เดียว) ----------
    print("\n  --- ตรวจ overfit (train vs val เดี่ยว, ใช้แค่วินิจฉัย) ---")
    prev_parity_train = extra_.loc[X_train.index, "prev_parity"].fillna(0)
    parity_train = extra_.loc[X_train.index, "y_parity"]
    prev_parity_val = extra_.loc[X_val.index, "prev_parity"].fillna(0)
    parity_val = extra_.loc[X_val.index, "y_parity"]

    for name, model in get_classifiers().items():
        model.fit(X_train, yflip_train)
        train_pred, _ = _reconstruct_parity(
            prev_parity_train, model.predict(X_train))
        val_pred, _ = _reconstruct_parity(
            prev_parity_val, model.predict(X_val))
        train_acc = float(np.mean(train_pred == parity_train.values))
        val_acc = float(np.mean(val_pred == parity_val.values))
        print(f"    {name:22s} train={train_acc:.4f}  val={val_acc:.4f}  "
              f"gap={train_acc - val_acc:+.4f}")
        if name == "Logistic Regression":
            _print_logreg_coefs(model, X_train.columns)

    if dev:
        print("\n  (โหมด dev — จบแค่นี้ ไม่แตะ test ไม่บันทึกผล)")
        return {"table": wf_df, "preds": preds_oof, "y_test": y_oof,
                "test_index": y_oof.index, "best_model": best,
                "stage": "walk_forward_oof"}

    # ---------- ยิงนัดเดียวบน test: refit โมเดลที่ชนะบน train+val ทั้งก้อน ----------
    print(f"\n  === รัน test ครั้งเดียว: refit {best} บน train+val "
          f"({len(dev_pool_X)} แถว) ===")
    X_test, yflip_test = parts["test"]
    final_model = get_classifiers()[best]
    final_model.fit(dev_pool_X, dev_pool_yflip)

    prev_parity_test = extra_.loc[X_test.index, "prev_parity"].fillna(0)
    parity_test = extra_.loc[X_test.index, "y_parity"]
    flip_test_pred = final_model.predict(X_test)
    try:
        flip_test_proba = final_model.predict_proba(X_test)[:, 1]
    except Exception:
        flip_test_proba = None
    parity_test_pred, parity_test_proba = _reconstruct_parity(
        prev_parity_test, flip_test_pred, flip_test_proba)

    parity_pool = extra_.loc[dev_pool_X.index, "y_parity"]
    base_preds_test = get_classification_baselines(
        parity_pool, parity_test, prev_values=prev_parity_test)

    test_results = {best: classification_metrics(
        parity_test, parity_test_pred, parity_test_proba)}
    for name, p in base_preds_test.items():
        test_results[name] = classification_metrics(parity_test, p)

    test_df = results_table(test_results, sort_by="Accuracy", ascending=False)
    print_table(test_df, "งาน A : ผลลัพธ์บน Test Set (ยิงนัดเดียว)")

    best_baseline_test = max(
        (n for n in test_df.index if n.startswith("Baseline")),
        key=lambda n: test_df.loc[n, "Accuracy"],
    )
    print("\n" + verdict(test_df.loc[best, "Accuracy"],
                         test_df.loc[best_baseline_test, "Accuracy"],
                         n=len(parity_test)))

    all_preds_test = {best: parity_test_pred, **base_preds_test}
    probas_test = {best: parity_test_proba} if parity_test_proba is not None else {}
    full_report(parity_test, all_preds_test, probabilities=probas_test,
               baseline_name=best_baseline_test)

    _run_permutation_check(best, final_model, dev_pool_X, dev_pool_yflip,
                          X_test, yflip_test, split_name="Test Set")

    print_confusion(parity_test, parity_test_pred, labels=("คู่", "คี่"),
                    title=f"({best}, refit บน train+val)")

    return {"table": test_df, "preds": {best: parity_test_pred},
            "y_test": parity_test, "test_index": X_test.index,
            "best_model": best, "stage": "test"}


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
