"""
overfit_curves.py  — A6
=======================
กราฟ train loss vs validation loss ต่อ epoch/iteration

ทำไมต้องมี
----------
ตอนนี้เรามีแค่ "ตัวเลข gap ณ จุดสุดท้าย"
    ANN        train=0.6373  val=0.5041  gap=+0.1332
    LightGBM   train=0.6778  val=0.4795  gap=+0.1984

ซึ่งบอกว่า *มี* overfitting แต่ไม่บอกว่า
  - มันเริ่มตอนไหน (epoch ที่เท่าไหร่)
  - แยกเร็วแค่ไหน
  - ถ้า early stop จะช่วยไหม

กราฟ loss ต่อ epoch ตอบได้ทั้งสามข้อ และเป็นรูปที่สื่อสารได้ดีที่สุด
สำหรับ overfitting (ตามที่เรียนใน Lecture 3)

สิ่งที่คาดว่าจะเห็นในงานนี้
  train loss ลงเรื่อย ๆ (โมเดลจำข้อมูลได้)
  val loss แบนหรือโค้งขึ้นตั้งแต่ต้น ไม่เคยลงเลย
  -> ยืนยันว่าไม่มี signal ให้เรียนรู้ มีแต่ noise ให้จำ

เกณฑ์อ้างอิง: log(2) = 0.6931 คือ loss ของการทายความน่าจะเป็น 0.5 ทุกครั้ง
ถ้า val loss ไม่เคยลงต่ำกว่าเส้นนี้ = โมเดลไม่เคยเรียนรู้อะไรที่ใช้ได้จริง

*** ไม่แตะ test set ***
เทรนบน train split ประเมินบน val split ซึ่งทั้งคู่อยู่ใน dev

วิธีใช้
    python overfit_curves.py
    python overfit_curves.py --epochs 300 --models ann,lgbm
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from data_adapter import load_investing_csv
from tick_utils import resolve_data_path
from run_permutation_fixed import build_features

warnings.filterwarnings("ignore")

LOG2 = float(np.log(2))
SEED = 42

# สัดส่วน train/val ภายใน dev — ต้องตรงกับ config.py
try:
    from config import TRAIN_RATIO, VAL_RATIO
except ImportError:
    TRAIN_RATIO, VAL_RATIO = 0.70, 0.15


def split_train_val(data: pd.DataFrame):
    """
    แบ่ง train / val ตามเวลา (ไม่แตะ test)
    ใช้สัดส่วนเดียวกับ splits.chronological_split()
    """
    n = len(data)
    i_train = int(n * TRAIN_RATIO)
    i_val = int(n * (TRAIN_RATIO + VAL_RATIO))

    train, val = data.iloc[:i_train], data.iloc[i_train:i_val]
    assert train.index[-1] < val.index[0], "train ทับ val"
    assert i_val < n, "ไม่ได้กันส่วน test ไว้"
    return train, val, data.iloc[i_val:]


def xy(frame: pd.DataFrame):
    cols = [c for c in frame.columns
            if c not in {"y_parity", "y_flip", "prev_parity"}]
    return frame[cols].to_numpy(float), frame["y_flip"].to_numpy(int)


def logloss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------
def curve_ann(Xtr, ytr, Xva, yva, epochs: int, hidden=(16,), alpha=0.5):
    """
    เทรน MLP ทีละ epoch ด้วย warm_start แล้วบันทึก loss ทั้งสองฝั่ง

    ใช้ warm_start=True + max_iter=1 วนเอง แทนการเรียก fit ครั้งเดียว
    เพื่อให้เก็บ validation loss ได้ทุก epoch (MLPClassifier เก็บให้แค่
    training loss ใน loss_curve_)
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    Xtr_s = sc.transform(imp.transform(Xtr))
    Xva_s = sc.transform(imp.transform(Xva))

    model = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                          alpha=alpha, learning_rate_init=1e-3,
                          max_iter=1, warm_start=True, early_stopping=False,
                          random_state=SEED)

    rows = []
    for epoch in range(1, epochs + 1):
        model.fit(Xtr_s, ytr)
        rows.append({
            "epoch": epoch,
            "train_loss": logloss(ytr, model.predict_proba(Xtr_s)[:, 1]),
            "val_loss": logloss(yva, model.predict_proba(Xva_s)[:, 1]),
            "train_acc": float(model.score(Xtr_s, ytr)),
            "val_acc": float(model.score(Xva_s, yva)),
        })
    return pd.DataFrame(rows)


def curve_lgbm(Xtr, ytr, Xva, yva, n_estimators: int):
    """
    LightGBM: ใช้ eval_set + record_evaluation เก็บ logloss ทุก iteration
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("  [lgbm] ไม่ได้ติดตั้ง lightgbm — ข้าม")
        return None

    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr_i, Xva_i = imp.transform(Xtr), imp.transform(Xva)

    history = {}
    model = lgb.LGBMClassifier(
        num_leaves=7, max_depth=3, learning_rate=0.02,
        n_estimators=n_estimators, min_child_samples=100, reg_lambda=10,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1,
    )
    model.fit(Xtr_i, ytr,
              eval_set=[(Xtr_i, ytr), (Xva_i, yva)],
              eval_names=["train", "val"],
              eval_metric="binary_logloss",
              callbacks=[lgb.record_evaluation(history),
                         lgb.log_evaluation(0)])

    return pd.DataFrame({
        "epoch": np.arange(1, len(history["train"]["binary_logloss"]) + 1),
        "train_loss": history["train"]["binary_logloss"],
        "val_loss": history["val"]["binary_logloss"],
    })


# ---------------------------------------------------------------
def summarise(curve: pd.DataFrame, ticker: str, model: str) -> dict:
    """หาจุดที่ val loss ต่ำสุด และดูว่าเคยลงต่ำกว่า log(2) ไหม"""
    best = curve.loc[curve["val_loss"].idxmin()]
    final = curve.iloc[-1]
    below = curve["val_loss"] < LOG2

    return {
        "ticker": ticker,
        "model": model,
        "epochs": int(final["epoch"]),
        "best_val_epoch": int(best["epoch"]),
        "best_val_loss": round(float(best["val_loss"]), 4),
        "final_train_loss": round(float(final["train_loss"]), 4),
        "final_val_loss": round(float(final["val_loss"]), 4),
        "final_gap": round(float(final["val_loss"] - final["train_loss"]), 4),
        "log2_reference": round(LOG2, 4),
        "val_ever_below_log2": "YES" if below.any() else "no",
        "epochs_below_log2": int(below.sum()),
    }


def plot(curves: dict, outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [(k, v) for k, v in curves.items() if v is not None and len(v)]
    if not items:
        return
    fig, axes = plt.subplots(1, len(items), figsize=(6 * len(items), 4.8),
                             squeeze=False)

    for ax, ((ticker, model), curve) in zip(axes[0], items):
        ax.plot(curve["epoch"], curve["train_loss"], lw=2,
                color="#4c78a8", label="Train loss")
        ax.plot(curve["epoch"], curve["val_loss"], lw=2,
                color="#e45756", label="Validation loss")
        ax.axhline(LOG2, color="black", ls=":", lw=1.5,
                   label=f"log(2) = {LOG2:.4f}")
        best = curve.loc[curve["val_loss"].idxmin()]
        ax.axvline(best["epoch"], color="grey", ls="--", lw=1,
                   label=f"Best val @ epoch {int(best['epoch'])}")
        ax.set_xlabel("Epoch / boosting iteration")
        ax.set_ylabel("Binary log loss")
        ax.set_title(f"{ticker} — {model.upper()}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "A6_overfit_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] {path}")


# ---------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*",
                    default=["KBANK_10Y_Cleaned.csv", "ADVANC_10Y_Cleaned.csv"])
    ap.add_argument("--models", default="ann,lgbm")
    ap.add_argument("--epochs", type=int, default=300,
                    help="จำนวน epoch ของ ANN / boosting rounds ของ LightGBM")
    ap.add_argument("--round-mode", default="gte", choices=["gt", "gte"])
    ap.add_argument("--date-format", default="%m/%d/%Y")
    ap.add_argument("--outdir", default="results/dev")
    args = ap.parse_args()

    kinds = [k.strip() for k in args.models.split(",")]
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print("  A6 — Train vs Validation loss curves   (dev เท่านั้น)")
    print(f"  เกณฑ์อ้างอิง log(2) = {LOG2:.4f} "
          f"(loss ของการทายความน่าจะเป็น 0.5 ทุกครั้ง)")
    print("=" * 78)

    curves, summaries = {}, []

    for raw in args.files:
        path = resolve_data_path(raw)
        name = os.path.basename(path).split("_")[0]
        frame = load_investing_csv(path, date_format=args.date_format,
                                   verbose=False)
        data = build_features(frame, round_mode=args.round_mode)
        train, val, test = split_train_val(data)

        print(f"\n[{name}] train {len(train)} แถว "
              f"({train.index[0].date()} -> {train.index[-1].date()})")
        print(f"         val   {len(val)} แถว "
              f"({val.index[0].date()} -> {val.index[-1].date()})")
        print(f"         test  {len(test)} แถว กันไว้ ไม่แตะ")

        Xtr, ytr = xy(train)
        Xva, yva = xy(val)

        for kind in kinds:
            print(f"  [{kind}] เทรน {args.epochs} รอบ ...",
                  end=" ", flush=True)
            if kind == "ann":
                curve = curve_ann(Xtr, ytr, Xva, yva, args.epochs)
            else:
                curve = curve_lgbm(Xtr, ytr, Xva, yva, args.epochs)
            print("เสร็จ" if curve is not None else "")

            if curve is not None:
                curves[(name, kind)] = curve
                summaries.append(summarise(curve, name, kind))
                curve.to_csv(
                    os.path.join(args.outdir, f"A6_curve_{name}_{kind}.csv"),
                    index=False)

    table = pd.DataFrame(summaries)
    print(f"\n{'='*78}")
    print("  สรุป")
    print(f"{'='*78}")
    print(table.to_string(index=False))
    table.to_csv(os.path.join(args.outdir, "A6_overfit_summary.csv"),
                 index=False, encoding="utf-8-sig")

    print(f"\n  การตีความ")
    for r in table.itertuples():
        print(f"    [{r.ticker}/{r.model}]")
        print(f"      train loss สุดท้าย {r.final_train_loss:.4f} | "
              f"val loss สุดท้าย {r.final_val_loss:.4f} | "
              f"ห่างกัน {r.final_gap:+.4f}")
        if r.val_ever_below_log2 == "no":
            print(f"      val loss ไม่เคยลงต่ำกว่า log(2) เลยสักรอบ")
            print(f"      -> โมเดลไม่เคยเรียนรู้อะไรที่ใช้ได้จริง "
                  f"มีแต่การจำ noise ในชุดฝึก")
        else:
            print(f"      val loss ต่ำกว่า log(2) ได้ {r.epochs_below_log2} "
                  f"จาก {r.epochs} รอบ (ต่ำสุด {r.best_val_loss:.4f} "
                  f"ที่รอบ {r.best_val_epoch})")
            if r.best_val_loss > LOG2 - 0.005:
                print(f"      -> ต่ำกว่าเพียงเล็กน้อย ยังถือว่าไม่มี signal")

    plot(curves, args.outdir)
    print(f"\n  [csv] {args.outdir}/")
    print("  *** ยังไม่มีตัวเลขไหนมาจาก test set ***")
