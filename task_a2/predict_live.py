"""
predict_live.py — ทำนายคู่/คี่ของราคาปิด "วันนี้" จาก snapshot ณ 16:00 จริง
==============================================================================

    python predict_live.py --target-date 2026-09-28
    python predict_live.py --target-date 2026-09-28 --dry-run

*** ต้องรันหลัง 16:00 น. (เวลาไทย) ของวันซื้อขายเท่านั้น ***
ก่อนหน้านั้นแท่ง "15:00" (= ราคา ณ 16:00 ตามที่ features_a2.py นิยาม)
ของวันนี้ยังไม่มีข้อมูลครบ สคริปต์จะปฏิเสธการทำนายอย่างเป็นทางการถ้า
เรียกก่อนเวลา (ใช้ --dry-run เพื่อทดสอบนอกเวลาได้)

*** ไม่ใช่การเปิด test set ***
main_a2.py = historical evaluation (walk-forward CV + test ที่ยิงไปแล้ว)
ไฟล์นี้ = prospective prediction ของวันที่ยังไม่ปิดตลาด

*** ทำไมต้องดึงข้อมูลสดเอง ไม่ใช้ data/*_1h_730d.csv เดิม ***
ไฟล์ 1h เดิมค้างที่วันสุดท้ายตอนดึงครั้งก่อน (ไม่ auto-update) และ
features_a2.build_daily_context_features() เดิมก็เรียก task_a/load_stock()
ซึ่งอ่านจาก data_cache/ ที่แช่แข็งไว้ที่ END_DATE ของ config.py เหมือนกัน
ไฟล์นี้จึงดึงทั้งสองแหล่งสดใหม่ทุกครั้ง แยกจาก path ที่ backtest ใช้

*** ห้ามให้สคริปต์เดาวันที่เอง ***
บังคับระบุ --target-date เสมอ (ต้องตรงกับวันที่ของราคาปิดที่กำลังทำนาย)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
TASK_A = HERE / ".." / "task_a"
sys.path.insert(0, str(TASK_A))

from config import RANDOM_STATE  # noqa: E402
from rounding import to_int_baht, parity as parity_fn  # noqa: E402
from main import _prepare, _reconstruct_parity  # noqa: E402

# ใช้ fetch_live ตัวเดียวกับ task_a/predict_live.py เป๊ะ (ไม่ก็อปโค้ด
# ดึงราคาปิดรายวันสดซ้ำอีกที่ — เดิมมีสองชุดที่ทำเรื่องเดียวกัน เสี่ยง
# แก้ไม่ครบทั้งคู่ตอนมีบั๊ก ให้เหลือแหล่งความจริงเดียว)
# ใช้ importlib โหลดจาก path ตรงๆ แทน "from predict_live import ..."
# เพราะไฟล์นี้เองก็ชื่อ predict_live.py เหมือนกัน — ถ้าพึ่ง sys.path
# ล้วนๆ จะเปราะบาง (ผลลัพธ์ขึ้นกับลำดับ sys.path ตอนรัน)
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "task_a_predict_live", str(TASK_A / "predict_live.py"))
_task_a_predict_live = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_task_a_predict_live)
fetch_live_daily = _task_a_predict_live.fetch_live

import features_a2  # noqa: E402
from main_a2 import (  # noqa: E402
    get_classifiers, walk_forward_eval, baseline_persistence_1600,
)

OUTPUT_DIR = HERE / "results"
LOG_PATH = OUTPUT_DIR / "prediction_log.csv"
DRYRUN_LOG_PATH = OUTPUT_DIR / "dryrun" / "prediction_log_dryrun.csv"
BANGKOK = timezone(timedelta(hours=7))

TICKERS = {"KBANK.BK": "KBANK", "ADVANC.BK": "ADVANC"}

KEY = ["ticker", "target_date", "model"]   # ต้องมี model ด้วย เพราะตอนนี้ทำนาย 3 โมเดล/หุ้น (2026-09-30)
LOG_COLUMNS = [
    "generated_at", "ticker", "target_date", "bar_1600_time", "train_rows",
    "model", "price_1600", "parity_1600", "predicted_flip_proba",
    "predicted_parity", "predicted_label", "code_commit", "worktree_dirty",
    "is_dry_run",
]


def get_code_commit(cwd=HERE):
    """git commit hash ของโค้ดที่รันอยู่ -- ดู task_a/predict_live.py สำหรับเหตุผล"""
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def worktree_dirty(cwd=HERE):
    """True ถ้ามีไฟล์ใน task_a2/ ต่างจาก HEAD ตอนรัน"""
    r = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=cwd,
                       capture_output=True, text=True)
    return bool(r.stdout.strip()) if r.returncode == 0 else None


# ---------------------------------------------------------------
# ดึงข้อมูลสด — แยกจาก data/*_1h_730d.csv และ task_a/data_cache/ เดิม
# ---------------------------------------------------------------

def fetch_live_1h(yf_ticker):
    df = yf.download(yf_ticker, interval="1h", period="730d",
                     auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"ดึง 1h ของ {yf_ticker} ไม่ได้")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert("Asia/Bangkok")
    return df



def check_has_1500_bar(today_bars, today, now_bkk, allow_incomplete):
    """
    เช็คว่ามีแท่ง 15:00 ของวันนี้แล้วหรือยัง (= ราคา ณ 16:00 ตาม
    features_a2.py) -- แยกเป็นฟังก์ชันเดี่ยวเพื่อ unit test ได้โดยไม่ต้อง
    ดึงข้อมูลสดจริง (2026-09-29)
    """
    has_1500_bar = (pd.Timestamp("15:00").time() in
                    [t.time() for t in today_bars.index])
    if not has_1500_bar:
        msg = (f"ยังไม่มีแท่ง 15:00 ของ {today} (= ราคา ณ 16:00) — "
              f"ตอนนี้เวลา {now_bkk.strftime('%H:%M')} น. "
              f"ต้องรันหลัง ~16:00 น. ของวันซื้อขาย")
        if allow_incomplete:
            print(f"    !! {msg} (ข้ามไปเพราะ --dry-run)")
        else:
            raise RuntimeError(msg)


def check_time_window(now_bkk, allow_incomplete):
    """
    official ต้องรันในช่วง 16:00-16:30 น. เท่านั้น (เหมือน
    task_b/predict_1600.py) -- เพิ่ม 2026-09-29: เดิมเช็คแค่ขอบล่าง
    (check_has_1500_bar) ไม่มีขอบบน รันดึกแค่ไหนก็ผ่านได้ ไม่สอดคล้องกับ
    "official = ทำนายจริงตอนใกล้ตลาดปิด"
    """
    if allow_incomplete:
        return
    deadline = pd.Timestamp(year=now_bkk.year, month=now_bkk.month,
                            day=now_bkk.day, hour=16, minute=30, tz=BANGKOK)
    start_window = deadline.replace(hour=16, minute=0)
    if not (start_window <= now_bkk < deadline):
        raise RuntimeError(
            f"official ต้องรันระหว่าง 16:00-16:30 น. (เหมือน "
            f"task_b/predict_1600.py) ตอนนี้ {now_bkk.strftime('%H:%M:%S')} "
            "น. -- ใช้ --dry-run ถ้าต้องการทดสอบนอกเวลานี้"
        )


def predict_one_ticker(yf_ticker, short_name, target_date, allow_incomplete,
                      verbose=True):
    print("\n" + "#" * 78)
    print(f"#  หุ้น: {yf_ticker}  ->  ทำนาย {pd.Timestamp(target_date).date()}")
    print("#" * 78)

    df_1h = fetch_live_1h(yf_ticker)
    df_daily = fetch_live_daily(yf_ticker)

    today = pd.Timestamp(target_date).date()
    today_bars = df_1h[df_1h.index.date == today]
    now_bkk = pd.Timestamp.now(tz=BANGKOK)

    check_has_1500_bar(today_bars, today, now_bkk, allow_incomplete)
    check_time_window(now_bkk, allow_incomplete)

    intraday = features_a2.build_intraday_features(
        yf_ticker, short_name, df_1h=df_1h)
    daily_ctx = features_a2.build_daily_context_features(
        yf_ticker, df=df_daily)

    close_daily = df_daily["Close"].copy()
    close_daily.index = pd.to_datetime(close_daily.index)

    common = intraday.index.intersection(daily_ctx.index)
    common_hist = common[common.date < np.datetime64(today)] \
        if hasattr(common, "date") else common
    # ประวัติสำหรับเทรน: ทุกวันก่อนวันนี้ที่มีทั้ง feature และราคาปิดจริงแล้ว
    hist_dates = [d for d in common if d.date() < today and d in close_daily.index]
    if len(hist_dates) < 100:
        raise RuntimeError(f"ประวัติมีแค่ {len(hist_dates)} วัน น้อยเกินจะเทรน")

    X_hist = pd.concat([intraday.loc[hist_dates].drop(columns=["price_1600"]),
                       daily_ctx.loc[hist_dates]], axis=1)
    X_hist = X_hist.replace([np.inf, -np.inf], np.nan)

    close_hist = close_daily.loc[hist_dates]
    y_parity_hist = parity_fn(to_int_baht(close_hist))
    parity_1600_hist = parity_fn(to_int_baht(intraday.loc[hist_dates, "price_1600"]))
    yflip_hist = pd.Series((y_parity_hist.values + parity_1600_hist.values) % 2,
                          index=hist_dates)
    parity1600_hist_s = pd.Series(parity_1600_hist.values, index=hist_dates)
    yparity_hist_s = pd.Series(y_parity_hist.values, index=hist_dates)
    dist_hist = X_hist["dist_to_boundary_1600"]

    print(f"[live] ประวัติสำหรับเทรน: {len(hist_dates)} วัน "
          f"({min(hist_dates).date()} -> {max(hist_dates).date()})")

    y_oof, preds_oof, probas_oof = walk_forward_eval(
        X_hist, yflip_hist, parity1600_hist_s, yparity_hist_s, dist_hist,
        verbose=verbose)
    model_names = ["ANN (MLP)", "LightGBM", "Logistic Regression"]
    accs = {}
    for name in model_names:
        pred = preds_oof[name]
        yt = y_oof.loc[pred.index]
        accs[name] = float(np.mean(pred.values == yt.values))
    # เปลี่ยน 2026-09-30: เดิมเลือกแค่โมเดลที่ชนะ walk-forward OOF ตัวเดียวมาทำนาย
    # ตอนนี้ทำนายด้วยทั้ง 3 โมเดลแล้ว log ทุกตัว (เหมือน task_b/predict_1600.py
    # ที่ไม่เลือกผู้ชนะ แต่ให้คนอ่าน log เห็นทุกโมเดล) -- OOF accuracy ยัง print
    # ไว้ให้เห็นว่าตัวไหนแม่นกว่ากันในอดีต แค่ไม่ใช้ตัดสินว่าจะทำนายด้วยตัวไหน
    print(f"    walk-forward OOF (n={len(y_oof)}): "
          + ", ".join(f"{k}={v:.4f}" for k, v in accs.items()))

    if today not in intraday.index.date:
        raise RuntimeError(f"ไม่มี feature ของวันนี้ ({today}) เลย")
    today_ts = [d for d in intraday.index if d.date() == today][0]
    X_live = pd.concat([
        intraday.loc[[today_ts]].drop(columns=["price_1600"]),
        daily_ctx.reindex([today_ts]),
    ], axis=1)
    X_live = X_live.reindex(columns=X_hist.columns)
    X_live = X_live.replace([np.inf, -np.inf], np.nan)

    price_1600 = float(intraday.loc[today_ts, "price_1600"])
    parity_1600 = float(parity_fn(to_int_baht(pd.Series([price_1600]))).iloc[0])

    bar_time = [t.strftime("%H:%M") for t in today_bars.index
               if t.time() == pd.Timestamp("15:00").time()]
    bar_time = bar_time[0] if bar_time else None

    print(f"    ราคา ณ 16:00 = {price_1600:.2f}  "
          f"(parity = {'คี่' if parity_1600 == 1 else 'คู่'})")

    rows = []
    for name in model_names:
        model = get_classifiers()[name]
        model.fit(X_hist, yflip_hist)
        flip_pred = model.predict(X_live)
        try:
            flip_proba = model.predict_proba(X_live)[:, 1]
        except Exception:
            flip_proba = np.array([np.nan])
        parity_pred, _ = _reconstruct_parity(
            np.array([parity_1600]), flip_pred, flip_proba)
        label = "คี่" if parity_pred[0] == 1 else "คู่"
        print(f"    {name:20s} P(พลิก parity ตอนปิด)={flip_proba[0]:.4f} "
              f"-> parity = {label}")
        rows.append({
            "ticker": yf_ticker,
            "target_date": today.isoformat(),
            "bar_1600_time": bar_time,
            "train_rows": len(hist_dates),
            "model": name,
            "price_1600": price_1600,
            "parity_1600": parity_1600,
            "predicted_flip_proba": round(float(flip_proba[0]), 6),
            "predicted_parity": float(parity_pred[0]),
            "predicted_label": label,
        })
    return rows


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
            "ปฏิเสธการเขียนซ้ำ"
        )


def append_log(rows, log_path, columns=LOG_COLUMNS):
    rows = rows[columns]
    if log_path.exists():
        header = pd.read_csv(log_path, nrows=0).columns.tolist()
        if header != columns:
            raise RuntimeError(
                f"schema ของ {log_path.name} ไม่ตรงกับที่โค้ดกำหนดตอนนี้\n"
                f"  ในไฟล์: {header}\n  ในโค้ด : {columns}"
            )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    print(f"\n[log] เขียนต่อท้าย {len(rows)} แถว -> {log_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="ทำนายคู่/คี่ของราคาปิดวันนี้ จาก snapshot ณ 16:00"
    )
    p.add_argument("--target-date", required=True,
                  help="วันที่กำลังทำนาย YYYY-MM-DD (ต้องเป็นวันนี้ตามปฏิทินตลาด)")
    p.add_argument("--dry-run", action="store_true",
                  help="ทดสอบ: อนุญาตรันก่อน 16:00 (feature ไม่ครบ), "
                       "ข้าม duplicate guard, เขียนลง dryrun log แทน")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 78)
    print("  Task A2 : Live Prediction (คู่/คี่ ณ 16:00 -> ราคาปิด)")
    print(f"  target_date = {args.target_date}")
    if args.dry_run:
        print("  *** DRY RUN — ไม่ใช่คำทำนายอย่างเป็นทางการ ***")
    print("=" * 78)

    generated_at = datetime.now(BANGKOK).isoformat(timespec="seconds")

    rows = []
    for t, short in TICKERS.items():
        rows.extend(predict_one_ticker(t, short, args.target_date, args.dry_run))
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
              "predicted_flip_proba"]].to_string(index=False))
    print("=" * 78)
    if args.dry_run:
        print("\n*** DRY RUN — ไม่แตะ production log ***")
    else:
        print("\nอย่าลืม commit + push log:")
        print("  git add task_a2/results/prediction_log.csv")
        print(f"  git commit -m 'A2 prediction log: {args.target_date}' && git push")
    return out


if __name__ == "__main__":
    main()
