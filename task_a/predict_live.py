"""
predict_live.py — ทำนายคู่/คี่ของราคาปิด "วันข้างหน้า" จริง (ไม่ใช่ backtest)
=============================================================================

    python predict_live.py --target-date 2026-09-29
    python predict_live.py --target-date 2026-09-29 --dry-run

*** ไฟล์นี้ไม่ใช่การเปิด test set ***
main.py = historical evaluation (walk-forward CV + test แช่แข็งที่ยิงไปแล้ว)
ไฟล์นี้ = prospective prediction — เทรนด้วยข้อมูลที่รู้ผลแล้วทั้งหมด (รวม
ถึงช่วงที่เคยเป็น "test" ของ main.py ด้วย เพราะตอนนี้รู้ผลแล้วจริงๆ)
เพื่อทำนายวันที่ยังไม่เกิดขึ้นเท่านั้น ห้ามเอาผลจากไฟล์นี้ไปอ้างเป็นผล
test set ของ main.py เด็ดขาด

*** ทำไมต้องมีไฟล์นี้แยกจาก main.py ***
main.py ใช้ data_cache/ ที่แช่แข็งไว้ที่ START_DATE-END_DATE ของ config.py
(2016-08-26 -> 2026-08-28) โดยตั้งใจ เพื่อให้ผล CV/test ที่รายงานไปแล้ว
reproduce ได้ตรงเป๊ะทุกครั้งที่รัน ไม่ขยับไปตามวันที่ปัจจุบัน — ไฟล์นี้
จึงต้อง**ดึงข้อมูลสดของตัวเองแยกต่างหาก** ไม่แตะ data_cache/ เดิมเลย

*** ห้ามให้สคริปต์เดาวันทำการถัดไปเอง ***
ปฏิทินวันหยุดไทยซับซ้อนเกินกว่าจะเดา -> บังคับระบุ --target-date เสมอ
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from config import TICKERS, START_DATE, RANDOM_STATE
from data_loader import clean
from features import build_features, build_raw_features
from targets import build_targets
from models import get_classifiers
from main import _prepare, _walk_forward_eval, _reconstruct_parity

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results"
LOG_PATH = OUTPUT_DIR / "prediction_log.csv"
DRYRUN_LOG_PATH = OUTPUT_DIR / "dryrun" / "prediction_log_dryrun.csv"
BANGKOK = timezone(timedelta(hours=7))

KEY = ["ticker", "target_date"]
LOG_COLUMNS = [
    "generated_at", "ticker", "target_date", "data_cutoff", "train_rows",
    "model", "prev_parity", "predicted_flip_proba", "predicted_parity",
    "predicted_label", "is_dry_run",
]


# ---------------------------------------------------------------
# ดึงข้อมูลสด — แยกจาก data_loader.load_stock() / data_cache/ เดิม
# โดยตั้งใจ (ดู docstring ด้านบน)
# ---------------------------------------------------------------

def fetch_live(ticker):
    end_exclusive = (pd.Timestamp.now(tz=BANGKOK).normalize()
                     + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=START_DATE, end=end_exclusive,
                     auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"ดึงข้อมูลสดของ {ticker} ไม่ได้")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return clean(df, verbose=False)


def build_live_feature(df):
    """
    Feature ของ "วันนี้" (แถวสุดท้ายของ df) ที่ยังไม่ shift — คือ feature
    ที่ถูกต้องสำหรับทำนาย "พรุ่งนี้" (เพราะ pipeline ทั้งหมด shift(1) ก่อน
    เทรน แถวของวัน t ใช้ raw indicator ของวัน t-1 เสมอ)
    """
    raw = build_raw_features(df)
    last = raw.iloc[[-1]].copy()
    last.columns = [f"{c}_prev" for c in last.columns]
    last.index = [df.index[-1]]
    return last, df.index[-1]


def select_and_fit_best(X_all, yflip_all, extra_all, verbose=True):
    """
    เลือกโมเดลด้วย walk-forward CV บนข้อมูลทั้งหมดที่มี (ไม่มี held-out
    test แยก เพราะนี่คือการ deploy ไม่ใช่การประเมิน) แล้ว refit ตัวที่
    ชนะบนข้อมูลทั้งหมดอีกที ก่อนเอาไปทำนายจริง
    """
    y_oof, preds_oof, probas_oof = _walk_forward_eval(
        X_all, yflip_all, extra_all, verbose=verbose)

    model_names = ["ANN (MLP)", "LightGBM", "Logistic Regression"]
    accs = {}
    for name in model_names:
        pred = preds_oof[name]
        yt = y_oof.loc[pred.index]
        accs[name] = float(np.mean(pred.values == yt.values))
    best = max(accs, key=accs.get)
    if verbose:
        print(f"    เลือกจาก walk-forward OOF (n={len(y_oof)}): "
              + ", ".join(f"{k}={v:.4f}" for k, v in accs.items())
              + f"  -> ใช้ {best}")

    final_model = get_classifiers()[best]
    final_model.fit(X_all, yflip_all)
    return best, final_model


def predict_one_ticker(ticker, target_date, verbose=True):
    print("\n" + "#" * 78)
    print(f"#  หุ้น: {ticker}  ->  ทำนาย {pd.Timestamp(target_date).date()}")
    print("#" * 78)

    df = fetch_live(ticker)
    X = build_features(df, verbose=False)
    targets = build_targets(df, verbose=False)

    X_all, yflip_all, extra_all = _prepare(
        X, targets["y_flip"], targets[["prev_parity", "y_parity"]])

    data_cutoff = df.index[-1]
    print(f"[live] ข้อมูลล่าสุด (data_cutoff) = {data_cutoff.date()}  "
          f"({len(X_all)} แถวใช้เทรนได้)")

    expected_target = data_cutoff + pd.offsets.BDay(1)
    if pd.Timestamp(target_date).date() <= data_cutoff.date():
        raise RuntimeError(
            f"--target-date ({target_date}) ไม่ได้อยู่หลัง data_cutoff "
            f"({data_cutoff.date()}) เลย — ข้อมูลอาจเก่าไป หรือพิมพ์วันที่ผิด"
        )
    if pd.Timestamp(target_date).date() != expected_target.date():
        print(f"    !! เตือน: คาดว่า target ถัดไปคือ {expected_target.date()} "
              f"(วันทำการถัดจาก data_cutoff) แต่ --target-date = {target_date} "
              f"— ถ้าตั้งใจข้ามวันหยุดยาวก็โอเค ถ้าไม่ตั้งใจ ให้เช็คอีกที")

    best, final_model = select_and_fit_best(X_all, yflip_all, extra_all,
                                            verbose=verbose)

    X_live, live_date = build_live_feature(df)
    assert live_date == data_cutoff

    prev_parity = float(targets["y_parity"].loc[data_cutoff])
    flip_pred = final_model.predict(X_live)
    try:
        flip_proba = final_model.predict_proba(X_live)[:, 1]
    except Exception:
        flip_proba = [np.nan]
    parity_pred, _ = _reconstruct_parity(
        np.array([prev_parity]), flip_pred, np.array(flip_proba))

    label = "คี่" if parity_pred[0] == 1 else "คู่"
    print(f"    โมเดล: {best}")
    print(f"    parity ของวันนี้ ({data_cutoff.date()}) = "
          f"{'คี่' if prev_parity == 1 else 'คู่'}")
    print(f"    P(พลิก parity พรุ่งนี้) = {flip_proba[0]:.4f}")
    print(f"    ทำนาย {pd.Timestamp(target_date).date()}: parity = {label}")

    return {
        "ticker": ticker,
        "target_date": pd.Timestamp(target_date).date().isoformat(),
        "data_cutoff": data_cutoff.date().isoformat(),
        "train_rows": len(X_all),
        "model": best,
        "prev_parity": prev_parity,
        "predicted_flip_proba": round(float(flip_proba[0]), 6),
        "predicted_parity": float(parity_pred[0]),
        "predicted_label": label,
    }


# ---------------------------------------------------------------
# Log (append-only) + guard กันทำนายซ้ำ
# ---------------------------------------------------------------

def check_no_duplicate(new_rows, log_path=LOG_PATH):
    if not log_path.exists():
        return
    old = pd.read_csv(log_path)
    if old.empty:
        return
    for k in KEY:
        old[k] = old[k].astype(str)
    probe = new_rows[KEY].astype(str)
    merged = probe.merge(old[KEY].drop_duplicates(), on=KEY, how="inner")
    if len(merged):
        raise RuntimeError(
            "มีคำทำนายสำหรับ (ticker, target_date) นี้อยู่แล้วใน log:\n"
            f"{merged.drop_duplicates().to_string(index=False)}\n"
            "ปฏิเสธการเขียนซ้ำ — ห้ามทำนายวันเดียวกันซ้ำสองรอบ"
        )


def append_log(rows, log_path, columns=LOG_COLUMNS):
    rows = rows[columns]
    if log_path.exists():
        header = pd.read_csv(log_path, nrows=0).columns.tolist()
        if header != columns:
            raise RuntimeError(
                f"schema ของ {log_path.name} ไม่ตรงกับที่โค้ดกำหนดตอนนี้\n"
                f"  ในไฟล์: {header}\n  ในโค้ด : {columns}\n"
                "  ห้าม append ผสม schema"
            )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    print(f"\n[log] เขียนต่อท้าย {len(rows)} แถว -> {log_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="ทำนายคู่/คี่ของวันข้างหน้าจริง (ไม่ใช่ backtest)"
    )
    p.add_argument("--target-date", required=True,
                  help="วันที่จะทำนาย YYYY-MM-DD (ต้องเป็นวันทำการตลาด)")
    p.add_argument("--dry-run", action="store_true",
                  help="ทดสอบ: ข้าม duplicate guard, เขียนลง "
                       "results/dryrun/prediction_log_dryrun.csv แทน")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 78)
    print("  Task A : Live Prediction (คู่/คี่ ของวันข้างหน้า)")
    print(f"  target_date = {args.target_date}")
    if args.dry_run:
        print("  *** DRY RUN — ไม่ใช่คำทำนายอย่างเป็นทางการ ***")
    print("=" * 78)

    generated_at = datetime.now(BANGKOK).isoformat(timespec="seconds")

    rows = [predict_one_ticker(t, args.target_date) for t in TICKERS]
    out = pd.DataFrame(rows)
    out["generated_at"] = generated_at
    out["is_dry_run"] = args.dry_run
    out = out[LOG_COLUMNS]

    log_path = DRYRUN_LOG_PATH if args.dry_run else LOG_PATH
    if not args.dry_run:
        check_no_duplicate(out)
    append_log(out, log_path)

    print("\n" + "=" * 78)
    print(out[["ticker", "target_date", "model", "predicted_label",
              "predicted_flip_proba"]].to_string(index=False))
    print("=" * 78)
    if args.dry_run:
        print("\n*** DRY RUN — ไม่แตะ production log ***")
    else:
        print("\nอย่าลืม commit + push log:")
        print("  git add task_a/results/prediction_log.csv")
        print(f"  git commit -m 'prediction log: {args.target_date}' && git push")
    return out


if __name__ == "__main__":
    main()
