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
ROWS_PER_DATE = N_TICKERS * N_MODELS       # ค่าเริ่มต้น: 2 หุ้น × 3 โมเดล

# จำนวนแถวที่ถือว่า "ครบ" ของแต่ละชุด แยกตามช่วงวัน -- [(วันที่เริ่มใช้, แถว/วัน)] เรียงเก่า -> ใหม่
# task_a / task_a2 เพิ่มแถว "Baseline: Persistence" เป็นแถวที่ 4 ต่อหุ้น ตั้งแต่ 2026-10-20 (commit dd20373)
# -- ไม่ใช่โมเดล ML ตัวใหม่ (ยังเป็น ANN/LightGBM/Logistic 3 ตัวเดิม) แต่เป็นกฎตายตัว "ทายว่าไม่พลิก"
# ที่ log ไว้ทุกวันเพื่อเทียบ · evaluate_live.py ไม่นับแถวนี้ในยอด "รวม ML ทุกโมเดล" · ถ้าใช้เกณฑ์ 6 แถว
# เหมือนเดิมจะจับไม่ได้เวลาขาดไป 1 โมเดลจาก 4 (เหลือ 6 แถว) แต่ถ้าใช้ 8 ทั้งช่วงก็จะหาว่าวันยุค 3 โมเดลขาดหมด
EXPECTED_ROWS = {
    "task_a":  [("2026-09-28", 6), ("2026-10-20", 8)],
    "task_a2": [("2026-09-28", 6), ("2026-10-20", 8)],
}


def expected_rows(task, day):
    """แถว/วันที่ชุดนี้ควรมี ณ วันที่ day (ชุดที่ไม่ได้ระบุใน EXPECTED_ROWS = ROWS_PER_DATE ตลอด)"""
    sched = EXPECTED_ROWS.get(task) or [(str(START.date()), ROWS_PER_DATE)]
    day = pd.Timestamp(day)
    n = sched[0][1]
    for start, k in sched:
        if day >= pd.Timestamp(start):
            n = k
    return n

# ชื่อ task ภายในงาน backfill -> (โฟลเดอร์ task, โหมด, ไฟล์ log จริง, filter prediction_type)
# start / end = ช่วงวันที่ชุดนั้น "ควรมี" log (ไม่ใส่ = START ถึงวันนี้) -- ใช้กับชุดที่เลิกรันหรือเริ่มกลางทาง
# รอบ 16:00 ของ task_b เปลี่ยนจากรอบ 1 (same_day_1600 · predict_1600.py · target = close ของแท่ง 16:00)
# เป็นรอบ 2 (same_day_close · predict_close.py · target = ราคาปิดทางการ) ตั้งแต่ 2026-10-19
# (PHASE1D_PLAN_R2.md) -- target/โมเดลคนละอย่าง จึงเป็นสองชุดแยกกัน ไม่ใช่ชุดเดียวที่ต่อกัน
TASKS = {
    "task_a":  dict(dir="task_a",  mode="next_day", live="task_a/results/prediction_log.csv", ptype=None),
    "task_c":  dict(dir="task_c",  mode="next_day", live="task_c/results/prediction_log.csv", ptype=None),
    "task_b":  dict(dir="task_b",  mode="next_day", live="task_b/results/prediction_log.csv", ptype="next_day"),
    "task_a2": dict(dir="task_a2", mode="t1600",    live="task_a2/results/prediction_log.csv", ptype=None),
    "task_b_1600": dict(dir="task_b", mode="t1600", live="task_b/results/prediction_log.csv",
                        ptype="same_day_1600", end="2026-10-15"),
    "task_b_close": dict(dir="task_b", mode="t1600", live="task_b/results/prediction_log.csv",
                         ptype="same_day_close", start="2026-10-19"),
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


def task_days(task, days):
    """วันที่ชุดนี้ควรมี log (ตัดช่วงก่อน start / หลัง end ของชุดที่เลิกรันหรือเริ่มกลางทางออก)"""
    t = TASKS[task]
    start = pd.Timestamp(t.get("start") or START)
    end = pd.Timestamp(t["end"]) if t.get("end") else None
    return [d for d in days if d >= start and (end is None or d <= end)]


def find_gaps(task, days):
    """[(target_date_iso, reason)] -- reason = missing (ไม่มีเลย) | partial_models (มีไม่ครบตาม expected_rows)"""
    live = live_counts(task)
    gaps = []
    for d in task_days(task, days):
        iso = d.date().isoformat()
        n = live.get(iso, 0)
        if n == 0:
            gaps.append((iso, "missing"))
        elif n < expected_rows(task, d):
            gaps.append((iso, "partial_models"))
    return gaps


# ---------------------------------------------------------------
# แคชการดาวน์โหลดจาก yfinance -- ให้ทุก run ใช้ข้อมูลชุดเดียวกัน + เก็บเป็นหลักฐานว่าใช้ข้อมูลอะไร
# ---------------------------------------------------------------

def last_local_date(df):
    """วันสุดท้ายที่มีข้อมูลใน df (index เป็นเวลา -> แปลงเป็นวันที่ตามเวลากรุงเทพก่อน)"""
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(BANGKOK)
    return idx.max().date()


def covers(df, through):
    return df is not None and len(df) and last_local_date(df) >= pd.Timestamp(through).date()


def install_yf_cache():
    """
    ครอบ yf.download ด้วยแคชไฟล์ เพื่อให้ทุก run ใช้ข้อมูลชุดเดียวกัน + เก็บเป็นหลักฐานว่าใช้ข้อมูลอะไร

    BACKFILL_CACHE_THROUGH (ตั้งโดย run_one.py = target_date) แก้ปัญหาแคชค้าง: คำขอที่ใช้หน้าต่าง
    *สัมพัทธ์* (period="60d"/"730d" ของแท่ง 1h) มี cache key เดิมตลอด ถ้าไม่ตรวจอะไรเลย การ backfill
    วันใหม่จะได้ไฟล์เก่าที่ยังไม่มีวันนั้นแบบเงียบ ๆ (อาการ: "ไม่มีแท่ง 15:00") แคชจึงถือว่าใช้ได้เฉพาะ
    เมื่อข้อมูลครอบคลุมถึงวัน target · ถ้าไม่ครอบคลุม จะดาวน์โหลดใหม่เก็บเป็นไฟล์แยก `_thru<วันที่>`
    (ไม่ทับไฟล์เดิม -> ผล backfill ที่ทำไว้แล้วรันซ้ำได้ค่าเดิม) และเลือกไฟล์เก่าที่สุดที่ครอบคลุมพอ
    """
    import yfinance as yf
    if getattr(yf.download, "_backfill_cached", False):
        return
    orig = yf.download
    YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def cached(*args, **kw):
        key = json.dumps([[str(a) for a in args], sorted((k, str(v)) for k, v in kw.items())])
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        tk = str(args[0] if args else kw.get("tickers", "x")).replace(".", "_")
        stem = f"{tk}_{kw.get('interval', '1d')}_{h}"
        path = YF_CACHE_DIR / f"{stem}.pkl"
        # เช็คความครอบคลุมเฉพาะคำขอที่ใช้หน้าต่าง *สัมพัทธ์* (ไม่ระบุ end) เพราะคำขอที่ระบุ end
        # มี cache key ต่างกันตามวันอยู่แล้ว และโดยนิยามคืนข้อมูลถึงก่อน end (เวลาตัดของ next_day)
        through = os.environ.get("BACKFILL_CACHE_THROUGH") if "end" not in kw else None
        for p in [path] + sorted(YF_CACHE_DIR.glob(f"{stem}_thru*.pkl")):
            if not p.exists():
                continue
            df = pd.read_pickle(p)
            if not through or covers(df, through):
                return df
        if os.environ.get("BACKFILL_OFFLINE"):
            raise RuntimeError(f"BACKFILL_OFFLINE: ไม่พบแคชสำหรับ {key} -- ห้ามดาวน์โหลดใหม่ตอนทดสอบ")
        df = orig(*args, **kw)
        if df is not None and len(df):
            if through and not covers(df, through):
                raise RuntimeError(f"Yahoo คืนข้อมูลถึง {last_local_date(df)} ไม่ถึง {through} "
                                   f"ที่ต้องใช้ -- ทำ backfill วันนี้ไม่ได้ ({key})")
            out = path if not path.exists() else YF_CACHE_DIR / f"{stem}_thru{through}.pkl"
            pd.to_pickle(df, out)
            (YF_CACHE_DIR / "index.jsonl").open("a", encoding="utf-8").write(
                json.dumps({"file": out.name, "args": key,
                            "fetched_at": datetime.now(BANGKOK).isoformat(timespec="seconds"),
                            "rows": len(df), "last_date": str(last_local_date(df)),
                            "needed_through": through or ""}, ensure_ascii=False) + "\n")
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
