"""
backfill/common.py — ของใช้ร่วมของงาน backfill (ไม่แตะ production log / สคริปต์ของ task ใดเลย)
================================================================================================

backfill = ทำนาย "ย้อนหลัง" ให้วันที่ระบบอัตโนมัติพลาด โดยตัดข้อมูลไว้ ณ เวลาตัดของวันนั้น
(next_day: ราคาปิดถึงวันทำการก่อนหน้า · t1600: แท่งชั่วโมงถึง 15:00 ของวันนั้น) แล้วรันโค้ด/โมเดล
ชุดเดียวกับที่ระบบจริงใช้ ผลลง backfill/results/ เท่านั้น -- ห้ามนำไปปนกับ prediction_log.csv จริง

ข้อจำกัดที่ต้องรู้ (ดู backfill/README.md):
  - generated_at = เวลาที่รัน backfill จริง ไม่ใช่เวลาของวันนั้น -> ไม่ใช่หลักฐานการทำนายสด
  - ข้อมูลที่ใช้คือข้อมูลที่ Yahoo คืนตอนนี้ (นิ่งแล้ว) อาจต่างจากที่เห็นตอน 16:02 ของวันนั้นราว 1 tick
    โดยเฉพาะแท่ง 15:00 ของกลุ่ม t1600 (ดู task_b/results/dryrun/availability_1600.csv)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BACKFILL_DIR = ROOT / "backfill"
RESULTS_DIR = Path(os.environ.get("BACKFILL_RESULTS_DIR") or BACKFILL_DIR / "results")   # override ได้สำหรับรันตรวจความถูกต้อง
DATA_DIR = Path(os.environ.get("BACKFILL_DATA_DIR") or BACKFILL_DIR / "data")   # override ได้สำหรับ leak_test.py
YF_CACHE_DIR = DATA_DIR / "yf_cache"
BANGKOK = timezone(timedelta(hours=7))
HOLIDAYS_PATH = ROOT / "task_b" / "set_holidays.txt"
START = pd.Timestamp("2026-09-28")          # วันแรกที่ต้องการให้ log ครบ

N_TICKERS = 2
N_MODELS = 3
ROWS_PER_DATE = N_TICKERS * N_MODELS

# ชื่อ task ภายในงาน backfill -> (โฟลเดอร์ task, โหมด, ไฟล์ log จริง, filter prediction_type)
TASKS = {
    "task_a":  dict(dir="task_a",  mode="next_day", live="task_a/results/prediction_log.csv", ptype=None),
    "task_c":  dict(dir="task_c",  mode="next_day", live="task_c/results/prediction_log.csv", ptype=None),
    "task_b":  dict(dir="task_b",  mode="next_day", live="task_b/results/prediction_log.csv", ptype="next_day"),
    "task_a2": dict(dir="task_a2", mode="t1600",    live="task_a2/results/prediction_log.csv", ptype=None),
    "task_b_1600": dict(dir="task_b", mode="t1600", live="task_b/results/prediction_log.csv", ptype="same_day_1600"),
    "task_c_1600": dict(dir="task_c", mode="t1600", live="task_c/results/prediction_log_1600.csv", ptype=None),
}
EXTRA_COLUMNS = ["is_backfill", "backfill_reason"]


def out_path(task):
    t = TASKS[task]
    return RESULTS_DIR / f"{task}_{t['mode']}.csv"


# ---------------------------------------------------------------
# ปฏิทินวันทำการ (ใช้ไฟล์วันหยุดเดียวกับ automation/run_next_day.py)
# ---------------------------------------------------------------

def read_holidays(path=HOLIDAYS_PATH):
    out = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("confirmed_through:"):
            out.add(pd.Timestamp(line))
    return out


def trading_days(start=START, end=None, holidays=None):
    end = end or pd.Timestamp.now(tz=BANGKOK).normalize().tz_localize(None)
    holidays = read_holidays() if holidays is None else holidays
    return [d for d in pd.date_range(start, end) if d.dayofweek < 5 and d not in holidays]


# ---------------------------------------------------------------
# log จริง (อ่านอย่างเดียว) -> จำนวนแถว official ต่อ target_date
# ---------------------------------------------------------------

def live_counts(task):
    t = TASKS[task]
    d = pd.read_csv(ROOT / t["live"])
    d = d[~d["is_dry_run"].astype(str).str.lower().isin(["true", "1"])]
    if t["ptype"]:
        d = d[d["prediction_type"] == t["ptype"]]
    return d.groupby("target_date").size().to_dict()


def backfill_counts(task):
    p = out_path(task)
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    return d.groupby("target_date").size().to_dict()


def find_gaps(task, days):
    """[(target_date_iso, reason)] -- reason = missing (ไม่มีเลย) | partial_models (มีไม่ครบ 6 แถว)"""
    live = live_counts(task)
    gaps = []
    for d in days:
        iso = d.date().isoformat()
        n = live.get(iso, 0)
        if n == 0:
            gaps.append((iso, "missing"))
        elif n < ROWS_PER_DATE:
            gaps.append((iso, "partial_models"))
    return gaps


# ---------------------------------------------------------------
# แคชการดาวน์โหลดจาก yfinance -- ให้ทุก run ใช้ข้อมูลชุดเดียวกัน + เก็บเป็นหลักฐานว่าใช้ข้อมูลอะไร
# ---------------------------------------------------------------

def install_yf_cache():
    import yfinance as yf
    if getattr(yf.download, "_backfill_cached", False):
        return
    orig = yf.download
    YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def cached(*args, **kw):
        key = json.dumps([[str(a) for a in args], sorted((k, str(v)) for k, v in kw.items())])
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        tk = str(args[0] if args else kw.get("tickers", "x")).replace(".", "_")
        path = YF_CACHE_DIR / f"{tk}_{kw.get('interval', '1d')}_{h}.pkl"
        if path.exists():
            return pd.read_pickle(path)
        if os.environ.get("BACKFILL_OFFLINE"):
            raise RuntimeError(f"BACKFILL_OFFLINE: ไม่พบแคชสำหรับ {key} -- ห้ามดาวน์โหลดใหม่ตอนทดสอบ")
        df = orig(*args, **kw)
        if df is not None and len(df):
            pd.to_pickle(df, path)
            (YF_CACHE_DIR / "index.jsonl").open("a", encoding="utf-8").write(
                json.dumps({"file": path.name, "args": key,
                            "fetched_at": datetime.now(BANGKOK).isoformat(timespec="seconds"),
                            "rows": len(df)}, ensure_ascii=False) + "\n")
        return df

    cached._backfill_cached = True
    yf.download = cached


# ---------------------------------------------------------------
# เขียนผล (append-only + กันซ้ำ + ตรวจ schema)
# ---------------------------------------------------------------

def git_head():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def write_rows(task, rows: pd.DataFrame, reason: str):
    rows = rows.copy()
    rows["is_backfill"] = True
    rows["backfill_reason"] = reason
    path = out_path(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        if list(old.columns) != list(rows.columns):
            raise RuntimeError(f"schema ของ {path.name} ไม่ตรง\n  ไฟล์: {list(old.columns)}\n  ใหม่: {list(rows.columns)}")
        key = ["ticker", "target_date", "model"]
        dup = rows[key].astype(str).merge(old[key].astype(str).drop_duplicates(), on=key)
        if len(dup):
            raise RuntimeError(f"มี backfill ของ key นี้แล้วใน {path.name}:\n{dup.to_string(index=False)}")
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"[backfill] เขียน {len(rows)} แถว -> {path}")


def now_iso():
    return datetime.now(BANGKOK).isoformat(timespec="seconds")
