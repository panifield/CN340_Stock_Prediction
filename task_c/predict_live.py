"""
predict_live.py — ทำนายทิศทางขึ้น/ลงของราคาปิด "วันข้างหน้า" จริง (ไม่ใช่ backtest)
====================================================================================

    python predict_live.py --target-date 2026-09-29
    python predict_live.py --target-date 2026-09-29 --dry-run

*** ไฟล์นี้ไม่ใช่การเปิด test set ***
main.py = historical evaluation (chronological train/val/test split ที่ยิงไปแล้ว)
ไฟล์นี้ = prospective prediction — เทรนด้วยข้อมูลที่รู้ผลแล้วทั้งหมด (รวม
ถึงช่วงที่เคยเป็น "val"/"test" ของ main.py ด้วย เพราะตอนนี้รู้ผลแล้วจริงๆ)
เพื่อทำนายวันที่ยังไม่เกิดขึ้นเท่านั้น ห้ามเอาผลจากไฟล์นี้ไปอ้างเป็นผล
test set ของ main.py เด็ดขาด

*** ทำไมต้องมีไฟล์นี้แยกจาก main.py ***
main.py ใช้ data_cache/ ที่แช่แข็งไว้ที่ START_DATE-END_DATE ของ config.py
(2016-08-26 -> 2026-08-28) โดยตั้งใจ เพื่อให้ผล val/test ที่รายงานไปแล้ว
reproduce ได้ตรงเป๊ะทุกครั้งที่รัน ไม่ขยับไปตามวันที่ปัจจุบัน — ไฟล์นี้
จึงต้อง**ดึงข้อมูลสดของตัวเองแยกต่างหาก** ไม่แตะ data_cache/ เดิมเลย
(รูปแบบเดียวกับ task_a/predict_live.py และ task_a2/predict_live.py)

*** เกี่ยวกับ Task 2 (intraday_task2.py, t1600) ***
ไฟล์นี้ทำนายเฉพาะงาน C "next_day" (ทิศทางราคาปิดของวันทำการถัดไป) เท่านั้น
ไม่เกี่ยวกับ intraday_task2.py ที่ cutoff/target ยังมีปัญหาอยู่ (ดู
KNOWN_ISSUES.md) — ห้ามเอาสคริปต์นี้ไปอ้างว่าแก้ปัญหา Task 2 ด้วย

*** วิธีเลือกโมเดล ***
main.py เลือกโมเดลที่ดีที่สุดจาก val set เดียว (chronological_split)
แต่ไฟล์นี้ไม่มี held-out val/test แยกให้ใช้ เพราะข้อมูลทั้งหมดถูกใช้เทรน
เพื่อ deploy จริง — จึงใช้ walk-forward CV (splits.walk_forward_splits)
ประเมิน out-of-fold accuracy ของทั้ง 3 โมเดลแทน แล้ว refit ตัวที่ชนะบน
ข้อมูลทั้งหมดอีกที ก่อนเอาไปทำนายจริง

*** ห้ามให้สคริปต์เดาวันทำการถัดไปเอง ***
ปฏิทินวันหยุดไทยซับซ้อนเกินกว่าจะเดา -> บังคับระบุ --target-date เสมอ
"""

from __future__ import annotations

import argparse
import subprocess
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone

warnings.filterwarnings("ignore")

from config import TICKERS, START_DATE
from data_loader import clean
from features import build_features, build_raw_features
from targets import build_targets
from models import get_classifiers
from splits import walk_forward_splits
from main import _prepare

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results"
LOG_PATH = OUTPUT_DIR / "prediction_log.csv"
DRYRUN_LOG_PATH = OUTPUT_DIR / "dryrun" / "prediction_log_dryrun.csv"
BANGKOK = timezone(timedelta(hours=7))

KEY = ["ticker", "target_date", "model"]   # ต้องมี model ด้วย เพราะตอนนี้ทำนาย 3 โมเดล/หุ้น (2026-09-30)
LOG_COLUMNS = [
    "generated_at", "ticker", "target_date", "data_cutoff", "train_rows",
    "model", "last_close", "last_direction", "predicted_up_proba",
    "predicted_updown", "predicted_label", "code_commit", "worktree_dirty",
    "is_dry_run",
]


def get_code_commit(cwd=BASE_DIR):
    """git commit hash ของโค้ดที่รันอยู่ -- ดู task_a/predict_live.py สำหรับเหตุผล"""
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def worktree_dirty(cwd=BASE_DIR):
    """True ถ้ามีไฟล์ใน task_c/ ต่างจาก HEAD ตอนรัน"""
    r = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=cwd,
                       capture_output=True, text=True)
    return bool(r.stdout.strip()) if r.returncode == 0 else None


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

    # yfinance บางครั้งคืนแถวของวันล่าสุดมาแล้ว แต่ Close ยังเป็น NaN (ยังไม่
    # backfill แม้ตลาดปิดไปแล้ว) -- clean() ด้านล่างเรียกด้วย verbose=False จะ
    # ตัดแถวนี้ทิ้งเงียบๆ ทำให้ data_cutoff ถอยไปหลายวันโดยไม่มีใครรู้ตัว เตือน
    # ให้ชัดตรงนี้ก่อน
    n_trailing_nan = 0
    for v in df["Close"].iloc[::-1]:
        if pd.isna(v):
            n_trailing_nan += 1
        else:
            break
    if n_trailing_nan:
        dropped = df.index[-n_trailing_nan:]
        print(f"[live] !! yfinance ยังไม่มีราคาปิดของ {n_trailing_nan} วันล่าสุด "
              f"({dropped[0].date()} -> {dropped[-1].date()}, Close=NaN แถวเหล่านี้ "
              f"มักเกิดจาก backfill ช้าหลังตลาดปิด) -> data_cutoff จะถอยไปใช้วันก่อนหน้า"
              f"ที่มีราคาปิดจริง")

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


def fit_all_models(X_all, y_all, verbose=True):
    """
    เทรนทั้ง 3 โมเดลบนข้อมูลทั้งหมดที่มี (ไม่มี held-out val/test แยก เพราะ
    นี่คือการ deploy ไม่ใช่การประเมิน) -- เปลี่ยน 2026-09-30: เดิมเลือกแค่
    โมเดลที่ชนะ walk-forward OOF ตัวเดียวมาทำนาย ตอนนี้ทำนายด้วยทั้ง 3 โมเดล
    แล้ว log ทุกตัว (เหมือน task_b/predict_live.py ที่ไม่เคยเลือกผู้ชนะอยู่
    แล้ว และ task_a/task_a2/task_c t1600 ที่เพิ่งแก้ตาม) ยัง print
    walk-forward OOF accuracy ไว้ให้เห็นว่าตัวไหนแม่นกว่ากันในอดีต แค่ไม่ใช้
    ตัดสินว่าจะทำนายด้วยตัวไหน
    """
    models = get_classifiers()
    oof_correct = {name: 0 for name in models}
    oof_total = 0

    for train_idx, test_idx in walk_forward_splits(X_all, n_splits=5, min_train=250):
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
    if verbose:
        print(f"    walk-forward OOF (n={oof_total}): "
              + ", ".join(f"{k}={v:.4f}" for k, v in accs.items()))

    for model in models.values():
        model.fit(X_all, y_all)
    return models


def predict_one_ticker(ticker, target_date, verbose=True):
    print("\n" + "#" * 78)
    print(f"#  หุ้น: {ticker}  ->  ทำนาย {pd.Timestamp(target_date).date()}")
    print("#" * 78)

    df = fetch_live(ticker)
    X = build_features(df, verbose=False)
    targets = build_targets(df, verbose=False)

    X_all, y_all = _prepare(X, targets["y_updown"])

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

    models = fit_all_models(X_all, y_all, verbose=verbose)

    X_live, live_date = build_live_feature(df)
    assert live_date == data_cutoff

    close = df["Close"]
    last_close = float(close.iloc[-1])
    if len(close) > 1 and close.iloc[-1] != close.iloc[-2]:
        last_direction = "ขึ้น" if close.iloc[-1] > close.iloc[-2] else "ลง"
    else:
        last_direction = "นิ่ง"

    print(f"    ราคาปิดล่าสุด ({data_cutoff.date()}) = {last_close:.2f} "
          f"(ทิศทางของวันนั้น: {last_direction})")

    rows = []
    for name, model in models.items():
        updown_pred = model.predict(X_live)
        try:
            up_proba = model.predict_proba(X_live)[:, 1]
        except Exception:
            up_proba = [np.nan]
        label = "ขึ้น" if updown_pred[0] == 1 else "ลง/นิ่ง"
        print(f"    {name:20s} P(ขึ้น พรุ่งนี้)={up_proba[0]:.4f} "
              f"-> ทำนาย {pd.Timestamp(target_date).date()}: {label}")
        rows.append({
            "ticker": ticker,
            "target_date": pd.Timestamp(target_date).date().isoformat(),
            "data_cutoff": data_cutoff.date().isoformat(),
            "train_rows": len(X_all),
            "model": name,
            "last_close": round(last_close, 4),
            "last_direction": last_direction,
            "predicted_up_proba": round(float(up_proba[0]), 6),
            "predicted_updown": float(updown_pred[0]),
            "predicted_label": label,
        })
    return rows


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


MARKET_OPEN_HOUR = 10          # SET เปิดเช้า ~10:00
MARKET_CLOSE_SAFE_HOUR = 18   # เดียวกับ task_b/daily_next_day.py: "รันหลังตลาดปิด (หลัง 18:00 +07:00)"


def check_market_closed(now=None):
    """
    กันรัน official ระหว่างตลาดอาจเปิดอยู่ หรือเพิ่งปิดแต่ข้อมูลยังไม่ settle —
    fetch_live() ดึงข้อมูลสดทุกครั้งไม่มีอะไรการันตีว่า "วันนี้" นิ่งแล้ว (ดู
    [live] !! คำเตือน NaN Close ด้านบน)

    ช่วงอันตรายคือ 10:00-18:00 (ตลาดอาจเปิดอยู่ หรือเพิ่งปิด 16:30 ไม่ถึง 2 ชม.
    ข้อมูลมักยังไม่ backfill) นอกช่วงนี้ (18:00 ถึงก่อน 10:00 ของวันถัดไป) ถือว่า
    ปลอดภัย รวมถึงรันข้ามเที่ยงคืนตอนเช้ามืด (เช่น automation/register_next_day.ps1
    ตั้งไว้ 08:30 -- เดิมเช็คแค่ `now.hour < 18` ซึ่งบล็อก 08:30 ผิดพลาดเพราะ 8 < 18
    ทั้งที่จริงห่างจากตลาดปิดมากกว่า 18:00 เดิมด้วยซ้ำ แก้ 2026-10-01)
    """
    now = now or pd.Timestamp.now(tz=BANGKOK)
    if MARKET_OPEN_HOUR <= now.hour < MARKET_CLOSE_SAFE_HOUR:
        raise RuntimeError(
            f"ตอนนี้ {now.strftime('%H:%M')} น. (เวลาไทย) -- official prediction "
            f"ควรรันหลัง {MARKET_CLOSE_SAFE_HOUR}:00 น. หรือก่อน {MARKET_OPEN_HOUR}:00 น."
            "ของวันถัดไปเท่านั้น เพื่อให้ราคาปิดของวันนี้นิ่งและ backfill ใน yfinance"
            "ทันแล้ว ใช้ --dry-run ถ้าต้องการ"
            "ทดสอบนอกเวลานี้"
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="ทำนายทิศทางขึ้น/ลงของวันข้างหน้าจริง (ไม่ใช่ backtest)"
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
    print("  Task C : Live Prediction (ทิศทางขึ้น/ลงของวันข้างหน้า)")
    print(f"  target_date = {args.target_date}")
    if args.dry_run:
        print("  *** DRY RUN — ไม่ใช่คำทำนายอย่างเป็นทางการ ***")
    print("=" * 78)

    if not args.dry_run:
        check_market_closed()

    generated_at = datetime.now(BANGKOK).isoformat(timespec="seconds")

    rows = []
    for t in TICKERS:
        rows.extend(predict_one_ticker(t, args.target_date))
    out = pd.DataFrame(rows)
    out["generated_at"] = generated_at
    out["code_commit"] = get_code_commit()
    out["worktree_dirty"] = worktree_dirty()
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
        print("  git add task_c/results/prediction_log.csv")
        print(f"  git commit -m 'prediction log: {args.target_date}' && git push")
    return out


if __name__ == "__main__":
    main()
