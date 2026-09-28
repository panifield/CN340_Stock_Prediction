"""
predict_live_1600.py — Task 2: ทำนายทิศทาง Close 16:00 จาก snapshot ณ cutoff จริง
================================================================================

    python predict_live_1600.py --target-date 2026-09-29
    python predict_live_1600.py --target-date 2026-09-29 --dry-run

*** ต้องรันหลัง cutoff (ค่า default 15:00 น. เวลาไทย ดู config.INTRADAY_CUTOFF_HOUR)
ของวันซื้อขายเท่านั้น *** ก่อนหน้านั้นแท่ง cutoff ของวันนี้ยังไม่มีข้อมูลครบ
สคริปต์จะปฏิเสธการทำนายอย่างเป็นทางการถ้าเรียกก่อนเวลา (ใช้ --dry-run เพื่อ
ทดสอบนอกเวลาได้) — รูปแบบเดียวกับ task_a2/predict_live.py

*** ไม่ใช่การเปิด test set ***
intraday_task2.py = historical evaluation (chronological train/val/test ที่ยิงไปแล้ว)
ไฟล์นี้ = prospective prediction ของวันที่ยังไม่ปิดตลาด

*** ทำไมต้องดึงข้อมูลสดเอง ***
intraday_task2.py อ่านจาก data_cache/{ticker}_1h_730d.csv ที่ค้างวันที่ (ไม่
auto-update) ไฟล์นี้จึงดึงแท่ง 1h สดจาก yfinance ทุกครั้งแยกต่างหาก ไม่แตะ
data_cache/ เดิมเลย (เหมือน task_a/task_a2/task_c predict_live.py next_day)

*** ทำไมไม่เรียก intraday_task2.build_task2_dataset() ตรงๆ กับข้อมูลทั้งก้อน ***
ฟังก์ชันนั้นต้องการทั้งแท่ง cutoff และแท่ง close_hour (16:00) ของทุกวัน — แต่
"วันนี้" ยังไม่มีแท่ง 16:00 (คือสิ่งที่กำลังทำนาย) จึงต้องแยก: ใช้
build_task2_dataset() กับข้อมูล**ก่อนวันนี้**เท่านั้นสำหรับเทรน (วันเหล่านั้น
ปิดตลาดไปแล้วจริง มีทั้งแท่ง cutoff และ close ครบ) แล้วคำนวณ feature ของ
วันนี้เองแยกต่างหากด้วยสูตรเดียวกันเป๊ะ (ไม่มี target เพราะยังไม่เกิด)

*** ห้ามให้สคริปต์เดาวันที่เอง ***
บังคับระบุ --target-date เสมอ (ต้องเป็น "วันนี้" ที่กำลังทำนาย ไม่ใช่วันถัดไป
— Task 2 ทำนายราคาปิดของวันเดียวกัน ไม่ใช่วันข้างหน้าแบบ Task 1)
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone

warnings.filterwarnings("ignore")

from config import TICKERS, INTRADAY_CUTOFF_HOUR, INTRADAY_CLOSE_HOUR
from models import get_classifiers
from splits import walk_forward_splits
from intraday_task2 import build_task2_dataset, _prepare

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results"
LOG_PATH = OUTPUT_DIR / "prediction_log_1600.csv"
DRYRUN_LOG_PATH = OUTPUT_DIR / "dryrun" / "prediction_log_1600_dryrun.csv"
BANGKOK = timezone(timedelta(hours=7))

FEATURE_COLUMNS = [
    "cutoff_hour_used", "return_open_to_cutoff", "high_from_open",
    "low_from_open", "range_to_cutoff", "cutoff_bar_return",
    "cutoff_bar_range", "prev_day_return", "overnight_gap",
    "volume_progress_vs_20d", "day_of_week",
]

KEY = ["ticker", "target_date"]
LOG_COLUMNS = [
    "generated_at", "ticker", "target_date", "cutoff_hour_used", "cutoff_price",
    "train_rows", "model", "predicted_up_proba", "predicted_up_to_close",
    "predicted_label", "is_dry_run",
]


# ---------------------------------------------------------------
# ดึงข้อมูลสด — แยกจาก data_cache/{ticker}_1h_730d.csv เดิม โดยตั้งใจ
# ---------------------------------------------------------------

def fetch_live_1h(ticker):
    df = yf.download(ticker, interval="1h", period="730d",
                     auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"ดึงแท่ง 1h ของ {ticker} ไม่ได้")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    if "Datetime" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Datetime"})
    idx = pd.DatetimeIndex(df["Datetime"])
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df["Datetime"] = idx.tz_convert("Asia/Bangkok")
    df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.sort_values("Datetime").drop_duplicates("Datetime")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[df["Close"] > 0]
    df["Date"] = df["Datetime"].dt.normalize()
    df["Hour"] = df["Datetime"].dt.hour
    return df


def _daily_aggregates(hourly, before_date):
    """actual_close (จากแท่ง close_hour) และ daily_volume (รวมทั้งวัน) ของทุกวันก่อน before_date"""
    hist = hourly[hourly["Date"] < before_date]
    daily_volume = hist.groupby("Date")["Volume"].sum().sort_index()
    close_bars = hist[hist["Hour"] == INTRADAY_CLOSE_HOUR]
    actual_close = close_bars.groupby("Date")["Close"].last().sort_index()
    return actual_close, daily_volume


def _today_feature_row(hourly, today, cutoff_hour=INTRADAY_CUTOFF_HOUR):
    """feature ของวันนี้ถึง cutoff เท่านั้น — สูตรเดียวกับ intraday_task2.build_task2_dataset()"""
    day = hourly[hourly["Date"] == today].sort_values("Datetime")
    available = day.loc[day["Hour"] <= cutoff_hour]
    if available.empty:
        raise RuntimeError(f"ยังไม่มีแท่งของวันนี้ ({today.date()}) เลย")

    cutoff = available.iloc[-1]
    open_price = float(available.iloc[0]["Open"])
    cutoff_close = float(cutoff["Close"])
    high_so_far = float(available["High"].max())
    low_so_far = float(available["Low"].min())
    cum_volume = float(available["Volume"].fillna(0).sum())
    cutoff_open = float(cutoff["Open"])

    return {
        "cutoff_price": cutoff_close,
        "session_open": open_price,
        "cutoff_hour_used": int(cutoff["Hour"]),
        "return_open_to_cutoff": cutoff_close / open_price - 1,
        "high_from_open": high_so_far / open_price - 1,
        "low_from_open": low_so_far / open_price - 1,
        "range_to_cutoff": (high_so_far - low_so_far) / cutoff_close,
        "cutoff_bar_return": cutoff_close / cutoff_open - 1,
        "cutoff_bar_range": (float(cutoff["High"]) - float(cutoff["Low"])) / cutoff_close,
        "cumulative_volume": cum_volume,
        "day_of_week": today.dayofweek,
    }


def select_and_fit_best(X_all, y_all, verbose=True):
    """
    เลือกโมเดลด้วย walk-forward CV บนข้อมูลทั้งหมดที่มี (ไม่มี held-out
    val/test แยก เพราะนี่คือการ deploy ไม่ใช่การประเมิน) แล้ว refit ตัวที่
    ชนะบนข้อมูลทั้งหมดอีกที ก่อนเอาไปทำนายจริง (เหมือน task_c/predict_live.py)
    """
    models = get_classifiers()
    oof_correct = {name: 0 for name in models}
    oof_total = 0

    for train_idx, test_idx in walk_forward_splits(X_all, n_splits=5, min_train=150):
        X_tr, y_tr = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_te, y_te = X_all.iloc[test_idx], y_all.iloc[test_idx]
        for name, model in models.items():
            fold_model = clone(model)
            fold_model.fit(X_tr, y_tr)
            pred = fold_model.predict(X_te)
            oof_correct[name] += int((pred == y_te.to_numpy()).sum())
        oof_total += len(test_idx)

    if oof_total == 0:
        raise RuntimeError(
            "ข้อมูลไม่พอสำหรับ walk-forward CV (n < min_train ของ "
            "splits.walk_forward_splits) — ตรวจว่าดึงข้อมูลสดมาครบไหม"
        )

    accs = {name: oof_correct[name] / oof_total for name in models}
    best = max(accs, key=accs.get)
    if verbose:
        print(f"    เลือกจาก walk-forward OOF (n={oof_total}): "
              + ", ".join(f"{k}={v:.4f}" for k, v in accs.items())
              + f"  -> ใช้ {best}")

    final_model = models[best]
    final_model.fit(X_all, y_all)
    return best, final_model


def predict_one_ticker(ticker, target_date, allow_incomplete, verbose=True):
    print("\n" + "#" * 78)
    print(f"#  หุ้น: {ticker}  ->  ทำนายราคาปิด {pd.Timestamp(target_date).date()}")
    print("#" * 78)

    hourly = fetch_live_1h(ticker)
    # ต้อง tz-aware ตรงกับ hourly["Date"] (Asia/Bangkok) ไม่งั้นเทียบกันไม่ได้
    # (== จะเงียบๆ คืน False ทุกแถว, < จะ raise TypeError ตรงๆ)
    today = pd.Timestamp(target_date, tz=BANGKOK).normalize()

    today_bars = hourly[hourly["Date"] == today]
    has_cutoff_bar = (today_bars["Hour"] == INTRADAY_CUTOFF_HOUR).any()
    now_bkk = pd.Timestamp.now(tz=BANGKOK)
    if not has_cutoff_bar:
        msg = (f"ยังไม่มีแท่ง {INTRADAY_CUTOFF_HOUR}:00 ของ {today.date()} "
              f"(cutoff ของ Task 2) — ตอนนี้เวลา {now_bkk.strftime('%H:%M')} น. "
              f"ต้องรันหลัง ~{INTRADAY_CUTOFF_HOUR}:00 น. ของวันซื้อขาย")
        if allow_incomplete:
            print(f"    !! {msg} (ข้ามไปเพราะ --dry-run)")
        else:
            raise RuntimeError(msg)

    hist_hourly = hourly[hourly["Date"] < today]
    X_hist, y_hist, _ = build_task2_dataset(hist_hourly)
    X_hist, y_hist, _ = _prepare(X_hist, y_hist, _)
    if len(X_hist) < 100:
        raise RuntimeError(f"ประวัติมีแค่ {len(X_hist)} วัน น้อยเกินจะเทรน")

    print(f"[live] ประวัติสำหรับเทรน: {len(X_hist)} วัน "
          f"({X_hist.index[0].date()} -> {X_hist.index[-1].date()})")

    best, final_model = select_and_fit_best(X_hist, y_hist, verbose=verbose)

    row = _today_feature_row(hourly, today)
    actual_close_hist, daily_volume_hist = _daily_aggregates(hourly, today)
    if len(actual_close_hist) < 21 or len(daily_volume_hist) < 21:
        raise RuntimeError(
            "ประวัติราคาปิด/ปริมาณซื้อขายไม่พอคำนวณ feature ของวันนี้ "
            "(ต้องการอย่างน้อย 21 วันก่อนหน้า)"
        )
    prev_close = float(actual_close_hist.iloc[-1])
    prev_prev_close = float(actual_close_hist.iloc[-2])
    row["prev_day_return"] = prev_close / prev_prev_close - 1
    row["overnight_gap"] = row["session_open"] / prev_close - 1
    expected_volume = float(daily_volume_hist.iloc[-20:].mean())
    row["volume_progress_vs_20d"] = row["cumulative_volume"] / expected_volume

    X_live = pd.DataFrame([row], index=[today])[FEATURE_COLUMNS]

    up_pred = final_model.predict(X_live)
    try:
        up_proba = final_model.predict_proba(X_live)[:, 1]
    except Exception:
        up_proba = [np.nan]

    label = "ขึ้น" if up_pred[0] == 1 else "ลง/นิ่ง"

    print(f"    โมเดล: {best}")
    print(f"    ราคา ณ cutoff {row['cutoff_hour_used']}:00 = {row['cutoff_price']:.2f}")
    print(f"    P(ขึ้นถึง close {INTRADAY_CLOSE_HOUR}:00) = {up_proba[0]:.4f}")
    print(f"    ทำนายราคาปิดวันนี้ ({today.date()}): {label}")

    return {
        "ticker": ticker,
        "target_date": today.date().isoformat(),
        "cutoff_hour_used": row["cutoff_hour_used"],
        "cutoff_price": round(row["cutoff_price"], 4),
        "train_rows": len(X_hist),
        "model": best,
        "predicted_up_proba": round(float(up_proba[0]), 6),
        "predicted_up_to_close": float(up_pred[0]),
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
        description="Task 2: ทำนายทิศทาง Close 16:00 จาก snapshot ณ cutoff จริง (ไม่ใช่ backtest)"
    )
    p.add_argument("--target-date", required=True,
                  help="วันที่กำลังทำนาย YYYY-MM-DD (ต้องเป็นวันนี้ตามปฏิทินตลาด)")
    p.add_argument("--dry-run", action="store_true",
                  help="ทดสอบ: อนุญาตรันก่อนถึง cutoff (feature ไม่ครบ), "
                       "ข้าม duplicate guard, เขียนลง dryrun log แทน")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 78)
    print("  Task C : Live Prediction (Task 2 -- ทิศทาง Close 16:00 จาก cutoff)")
    print(f"  target_date = {args.target_date}")
    if args.dry_run:
        print("  *** DRY RUN — ไม่ใช่คำทำนายอย่างเป็นทางการ ***")
    print("=" * 78)

    generated_at = datetime.now(BANGKOK).isoformat(timespec="seconds")

    rows = [predict_one_ticker(t, args.target_date, args.dry_run) for t in TICKERS]
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
              "predicted_up_proba"]].to_string(index=False))
    print("=" * 78)
    if args.dry_run:
        print("\n*** DRY RUN — ไม่แตะ production log ***")
    else:
        print("\nอย่าลืม commit + push log:")
        print("  git add task_c/results/prediction_log_1600.csv")
        print(f"  git commit -m '1600 prediction log: {args.target_date}' && git push")
    return out


if __name__ == "__main__":
    main()
