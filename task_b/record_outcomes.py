"""
record_outcomes.py — จับคู่คำทำนายใน prediction log กับผลจริง (live test)
=======================================================================

    python record_outcomes.py
    python record_outcomes.py --log results/dryrun/prediction_log_dryrun.csv \\
                              --out results/dryrun/outcomes_dryrun.csv --include-dry-run

ทำอะไร:
  อ่าน prediction log -> หาแถวที่ผลจริงออกแล้ว
  -> เขียนผลจริงต่อท้าย results/outcomes.csv (append-only · ไม่แก้ prediction log)

กติกา:
  - ผลจริงมาจากแหล่งเดียวกับที่โมเดลนั้นเทรน:
      next_day      : ราคาปิดใน raw_data/ (Investing.com)
      same_day_1600 : close_bar16 จาก Yahoo (live snapshot ที่ดาวน์โหลดหลัง 17:00 ของวันนั้น
                      ใน raw_data_intraday_live/ หรือไฟล์ล็อก raw_data_intraday/)
                      · actual_return = close_bar16 / close_bar15 − 1 (prev_close = close_bar15)
  - ค่าเริ่มต้นนับเฉพาะแถว official (is_dry_run = False) · dry-run ต้องสั่ง --include-dry-run
  - key เดิมที่บันทึกแล้วจะไม่ถูกเขียนซ้ำหรือเขียนทับ
  - prev_close ใน log ต้องตรงกับค่าในแหล่งผลจริง (next_day: ราคาปิดวัน data_cutoff ·
    same_day_1600: close_bar15 ของวันนั้น) ถ้าไม่ตรง (ข้อมูลถูก revise) ยังบันทึก
    แต่ติดธง prev_close_match = False
  - log dry-run ของ predict_1600 (schema ต่างกัน) ถูกแปลงคอลัมน์ให้ตรงก่อนอัตโนมัติ
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
SUPPORTED_TYPES = ["next_day", "same_day_1600"]

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


def normalize_log(log):
    """แปลง log dry-run ของ predict_1600 (close_bar15 ฯลฯ) ให้มีคอลัมน์แบบ prediction log"""
    if "prev_close" in log.columns:
        return log
    if "close_bar15" not in log.columns:
        raise ValueError("ไม่รู้จัก schema ของ log")
    log = log.copy()
    log["prev_close"] = log["close_bar15"]
    log["predicted_close"] = log["predicted_close_bar16"]
    log["data_cutoff"] = log["feature_window_end"]
    log["config_tag"] = "phase1d-modeA"
    return log


def compute_outcomes(log, closes, source, sha, recorded_at, include_dry_run=False,
                     intraday=None):
    """
    log      : DataFrame ของ prediction log
    closes   : {ticker: Series ราคาปิดจริง index = วันที่}   (next_day)
    intraday : callable(ticker, target) -> (C16, C15, source, sha) หรือ None   (same_day_1600)
               ถ้าไม่ให้มา แถว same_day_1600 จะถูกข้าม
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
        t = pd.Timestamp(r["target_date"])
        prev = float(r["prev_close"])
        if r["prediction_type"] == "same_day_1600":
            got = intraday(r["ticker"], t) if intraday is not None else None
            if got is None:
                continue                      # ยังไม่มี snapshot หลังตลาดปิด
            actual_close, c15, src, dsha = got
            prev_match = bool(np.isclose(c15, prev, rtol=0, atol=1e-9))
        else:
            c = closes.get(r["ticker"])
            if c is None:
                continue
            cut = pd.Timestamp(r["data_cutoff"])
            if t not in c.index:
                continue                      # ผลจริงยังไม่ออก / ยังไม่อัปเดต raw_data
            actual_close = float(c.loc[t])
            prev_match = bool(cut in c.index and np.isclose(c.loc[cut], prev, rtol=0, atol=1e-9))
            src, dsha = source.get(r["ticker"], ""), sha.get(r["ticker"], "")
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
            "prev_close_match": prev_match,
            "outcome_source": src, "outcome_sha256": dsha,
        })
    return pd.DataFrame(rows, columns=OUTCOME_COLUMNS)


def intraday_outcome_lookup():
    """close_bar16 จริงจาก Yahoo: snapshot หลัง 17:00 ของวันนั้น หรือไฟล์ล็อก (โหลดครั้งเดียว)"""
    from intraday_1600 import load_ticker_bars
    from live_1600 import find_snapshots, outcome_close_bar16
    cache = {}

    def lookup(ticker, target):
        if ticker not in cache:
            cache[ticker] = (find_snapshots(ticker), load_ticker_bars(ticker))
        snaps, locked = cache[ticker]
        return outcome_close_bar16(ticker, target, snaps, locked)
    return lookup


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

    log = normalize_log(pd.read_csv(log_path))
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
    out = compute_outcomes(log, closes, source, sha, now, args.include_dry_run,
                           intraday=intraday_outcome_lookup())
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
