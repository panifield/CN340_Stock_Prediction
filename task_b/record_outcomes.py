"""
record_outcomes.py — จับคู่คำทำนายใน prediction log กับผลจริง (live test)
=======================================================================

    python record_outcomes.py
    python record_outcomes.py --log results/dryrun/prediction_log_dryrun.csv \\
                              --out results/dryrun/outcomes_dryrun.csv --include-dry-run

ทำอะไร:
  อ่าน prediction log -> หาแถวที่ target_date มีราคาปิดจริงใน raw_data/ แล้ว
  -> เขียนผลจริงต่อท้าย results/outcomes.csv (append-only · ไม่แก้ prediction log)

กติกา:
  - รองรับเฉพาะ prediction_type = next_day (same_day_1600 ยังไม่เปิด official)
  - ผลจริงมาจากแหล่งเดียวกับที่โมเดลเทรน: raw_data/ (Investing.com) -- ห้ามใช้ Yahoo
  - ค่าเริ่มต้นนับเฉพาะแถว official (is_dry_run = False) · dry-run ต้องสั่ง --include-dry-run
  - key เดิมที่บันทึกแล้วจะไม่ถูกเขียนซ้ำหรือเขียนทับ
  - prev_close ใน log ต้องตรงกับราคาปิดของ data_cutoff ใน raw_data ปัจจุบัน
    ถ้าไม่ตรง (ข้อมูลถูก revise) ยังบันทึก แต่ติดธง prev_close_match = False
"""

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, BASE_DIR, TICKERS
from data_loader import load_stock, raw_data_path, dataset_fingerprint

BANGKOK = timezone(timedelta(hours=7))
LOG_PATH = OUTPUT_DIR / "prediction_log.csv"
OUTCOMES_PATH = OUTPUT_DIR / "outcomes.csv"
KEY = ["prediction_type", "ticker", "target_date", "model"]
SUPPORTED_TYPES = ["next_day"]

OUTCOME_COLUMNS = [
    "recorded_at", "prediction_type", "ticker", "target_date", "data_cutoff", "model",
    "config_tag", "code_commit", "generated_at", "is_dry_run",
    "prev_close", "predicted_return", "predicted_close",
    "actual_close", "actual_return", "error_return", "abs_error", "abs_error_baht",
    "naive_abs_error", "beat_naive", "prev_close_match",
    "outcome_source", "outcome_sha256",
]


def _is_true(s):
    return s.astype(str).str.lower().isin(["true", "1"])


def compute_outcomes(log, closes, source, sha, recorded_at, include_dry_run=False):
    """
    log    : DataFrame ของ prediction log
    closes : {ticker: Series ราคาปิดจริง index = วันที่}
    คืน DataFrame ผลจริงของแถวที่ target_date มีราคาปิดแล้ว (คอลัมน์ OUTCOME_COLUMNS)

    error_return    = predicted_return − actual_return
    naive_abs_error = |actual_return|              (Naive ทาย return = 0)
    beat_naive      = win / lose / tie             (|error| เทียบ |error ของ Naive|)
    """
    log = log.copy()
    if not include_dry_run:
        log = log[~_is_true(log["is_dry_run"])]
    log = log[log["prediction_type"].isin(SUPPORTED_TYPES)]
    rows = []
    for _, r in log.iterrows():
        c = closes.get(r["ticker"])
        if c is None:
            continue
        t, cut = pd.Timestamp(r["target_date"]), pd.Timestamp(r["data_cutoff"])
        if t not in c.index:
            continue                          # ผลจริงยังไม่ออก / ยังไม่อัปเดต raw_data
        actual_close = float(c.loc[t])
        prev = float(r["prev_close"])
        actual_ret = actual_close / prev - 1
        pred_ret = float(r["predicted_return"])
        err = pred_ret - actual_ret
        naive = abs(actual_ret)
        beat = "tie" if np.isclose(abs(err), naive, rtol=0, atol=1e-12) else (
            "win" if abs(err) < naive else "lose")
        rows.append({
            "recorded_at": recorded_at,
            **{k: r[k] for k in ["prediction_type", "ticker", "target_date", "data_cutoff",
                                 "model", "config_tag", "code_commit", "generated_at",
                                 "is_dry_run"]},
            "prev_close": prev, "predicted_return": pred_ret,
            "predicted_close": float(r["predicted_close"]),
            "actual_close": actual_close, "actual_return": actual_ret,
            "error_return": err, "abs_error": abs(err), "abs_error_baht": abs(err) * prev,
            "naive_abs_error": naive, "beat_naive": beat,
            "prev_close_match": bool(cut in c.index and np.isclose(c.loc[cut], prev,
                                                                   rtol=0, atol=1e-9)),
            "outcome_source": source.get(r["ticker"], ""), "outcome_sha256": sha.get(r["ticker"], ""),
        })
    return pd.DataFrame(rows, columns=OUTCOME_COLUMNS)


def append_new(outcomes, path):
    """เขียนต่อท้ายเฉพาะ key ที่ยังไม่มี · คืนจำนวนแถวที่เพิ่ม"""
    if outcomes.empty:
        return 0
    if path.exists():
        old = pd.read_csv(path)
        if old.columns.tolist() != OUTCOME_COLUMNS:
            raise RuntimeError(f"schema ของ {path.name} ไม่ตรง OUTCOME_COLUMNS")
        seen = set(map(tuple, old[KEY].astype(str).to_numpy()))
        new = outcomes[[tuple(k) not in seen for k in outcomes[KEY].astype(str).to_numpy()]]
    else:
        new = outcomes
    if new.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(new)


def main(argv=None):
    p = argparse.ArgumentParser(description="บันทึกผลจริงของคำทำนาย (append-only)")
    p.add_argument("--log", default=str(LOG_PATH))
    p.add_argument("--out", default=str(OUTCOMES_PATH))
    p.add_argument("--include-dry-run", action="store_true",
                   help="นับแถว dry-run ด้วย (ใช้ทดสอบเท่านั้น -- ควรเขียนลงไฟล์แยก)")
    args = p.parse_args(argv)
    log_path, out_path = BASE_DIR / args.log, BASE_DIR / args.out
    if args.include_dry_run and out_path.resolve() == OUTCOMES_PATH.resolve():
        raise ValueError("ห้ามเขียนผลของ dry-run ลง results/outcomes.csv -- ใช้ --out ไฟล์อื่น")

    log = pd.read_csv(log_path)
    skipped = log[~log["prediction_type"].isin(SUPPORTED_TYPES)]
    if len(skipped):
        print(f"[outcomes] ข้าม {len(skipped)} แถวที่ prediction_type ยังไม่รองรับ "
              f"{sorted(skipped['prediction_type'].unique())}")
    closes, source, sha = {}, {}, {}
    for t in TICKERS:
        closes[t] = load_stock(t, verbose=False)["Close"]
        source[t] = f"raw_data/{raw_data_path(t).name} (investing.com)"
        sha[t] = dataset_fingerprint(raw_data_path(t))
    now = datetime.now(BANGKOK).isoformat(timespec="seconds")
    out = compute_outcomes(log, closes, source, sha, now, args.include_dry_run)
    n = append_new(out, out_path)
    pending = log[(log["prediction_type"].isin(SUPPORTED_TYPES))
                  & (args.include_dry_run | ~_is_true(log["is_dry_run"]))]
    print(f"[outcomes] มีผลจริงแล้ว {len(out)} แถว · เพิ่มใหม่ {n} แถว -> {out_path}"
          f" · ยังรอผลจริง {len(pending) - len(out)} แถว")
    if len(out) and not out["prev_close_match"].all():
        print("  !! มีแถวที่ prev_close ไม่ตรง raw_data ปัจจุบัน (ข้อมูลอาจถูก revise) -- ตรวจด้วยตา")
    return n


if __name__ == "__main__":
    main()
