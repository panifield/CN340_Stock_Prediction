"""
daily_next_day.py — รันขั้นตอนประจำวันของโมเดลแบบที่ 1 (next_day) ครบในคำสั่งเดียว
===================================================================================

    python daily_next_day.py --dry-run            # ทุกขั้น แต่ทำนายแบบ dry-run · ไม่ commit
    python daily_next_day.py --commit --push      # official (Task Scheduler / GitHub Actions)

รันหลังตลาดปิด (หลัง 18:00 +07:00) ของวันทำการ t−1 · ขั้นตอนเดียวกับ INTEGRATION.md ข้อ 7:

  0. tools/check_env.py              เวอร์ชัน library ตรง pin (INTEGRATION.md ข้อ 5)
  1. tools/fetch_yahoo_daily.py      ตัวเชื่อม: ดึง Yahoo + ตรวจ OHLC ช่วงทับกับ raw_data/ เป๊ะ
  2. tools/append_investing.py       append แบบ byte-safe (--source yahoo) + check_raw_update
  3. main.py --dev                   regression: results/*_val.csv ต้องไม่เปลี่ยน
  4. commit ① ข้อมูล                  (--commit)
  5. record_outcomes.py + report_live.py
  6. predict_live.py                 target = วันทำการถัดไปจาก set_holidays.txt · --expected-cutoff = วันสุดท้ายของข้อมูล
  7. commit ② prediction log + push  (--commit / --push)

ไม่มีวันใหม่ (วันหยุด) -> ข้ามขั้น 2-4 · ถ้าคำทำนายของ target นั้นมีใน log แล้ว -> ข้ามขั้น 6 (รันซ้ำได้)
หยุดทันทีเมื่อขั้นใดล้มเหลว · สคริปต์นี้ไม่แตะ historical test (main.py รันด้วย --dev เท่านั้น)
"""

import argparse
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config import BASE_DIR, OUTPUT_DIR, RAW_DATA_DIR, RAW_DATA_SUFFIX, TICKERS

TOOLS = BASE_DIR / "tools"
HOLIDAYS_PATH = BASE_DIR / "set_holidays.txt"
LOG_PATH = OUTPUT_DIR / "prediction_log.csv"
VAL_GLOB = "results/*_val.csv"
EXIT_NO_NEW = 3            # ตรงกับ fetch_yahoo_daily.EXIT_NO_NEW
PY = sys.executable
BANGKOK = timezone(timedelta(hours=7))


class StepError(RuntimeError):
    pass


def run(step, args, ok_codes=(0,)):
    print(f"\n{'=' * 78}\n[{step}] {' '.join(str(a) for a in args)}\n{'=' * 78}", flush=True)
    rc = subprocess.run(args, cwd=BASE_DIR).returncode
    if rc not in ok_codes:
        raise StepError(f"ขั้น {step} ล้มเหลว (exit {rc}) -- หยุด")
    return rc


def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=BASE_DIR, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise StepError(f"git {' '.join(args)} ล้มเหลว:\n{r.stderr.strip()}")
    return r


# ---------------------------------------------------------------
# วันทำการ
# ---------------------------------------------------------------
def read_holidays(path=HOLIDAYS_PATH):
    """คืน (set ของวันหยุด, confirmed_through)"""
    holidays, through = set(), None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("confirmed_through:"):
            through = pd.Timestamp(line.split(":", 1)[1].strip())
        else:
            holidays.add(pd.Timestamp(line))
    if through is None:
        raise StepError(f"{Path(path).name}: ไม่มีบรรทัด confirmed_through:")
    return holidays, through


def next_trading_day(after, holidays, through):
    d = pd.Timestamp(after) + pd.Timedelta(days=1)
    while d.dayofweek > 4 or d in holidays:
        d += pd.Timedelta(days=1)
    if d > through:
        raise StepError(
            f"วันทำการถัดไปที่คำนวณได้ ({d.date()}) เกิน confirmed_through ({through.date()}) ใน "
            f"{HOLIDAYS_PATH.name}\n  -> เพิ่มวันหยุด SET จากประกาศทางการ แล้วเลื่อน confirmed_through")
    return d


def data_cutoff():
    """วันสุดท้ายของ raw_data/ -- ทุกหุ้นต้องเท่ากัน"""
    ends = {}
    for t in TICKERS:
        df = pd.read_csv(RAW_DATA_DIR / f"{t.replace('.BK', '')}{RAW_DATA_SUFFIX}", usecols=["Date"])
        ends[t] = pd.to_datetime(df["Date"], format="%m/%d/%Y").max()
    if len(set(ends.values())) != 1:
        raise StepError(f"วันสุดท้ายของ raw_data ไม่เท่ากัน: {ends}")
    return next(iter(ends.values()))


def already_predicted(target):
    if not LOG_PATH.exists():
        return False
    log = pd.read_csv(LOG_PATH, usecols=["prediction_type", "target_date", "is_dry_run"])
    official = log[(log["prediction_type"] == "next_day") & (~log["is_dry_run"].astype(str).eq("True"))]
    return (official["target_date"] == target.date().isoformat()).any()


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="ขั้นตอนประจำวันของ next_day ครบในคำสั่งเดียว")
    ap.add_argument("--dry-run", action="store_true", help="ทำนายแบบ dry-run (ไม่เขียน production log) · ห้ามใช้กับ --commit")
    ap.add_argument("--commit", action="store_true", help="commit ข้อมูลและ prediction log")
    ap.add_argument("--push", action="store_true", help="git push หลัง commit (ต้องใช้กับ --commit)")
    ap.add_argument("--holidays", type=Path, default=HOLIDAYS_PATH, help="ไฟล์วันหยุด SET (default set_holidays.txt)")
    args = ap.parse_args(argv)
    if args.dry_run and args.commit:
        ap.error("--dry-run ใช้กับ --commit ไม่ได้")
    if args.push and not args.commit:
        ap.error("--push ต้องใช้กับ --commit")

    try:
        run("0 env", [PY, TOOLS / "check_env.py"])
        holidays, through = read_holidays(args.holidays)     # ตรวจไฟล์วันหยุดก่อนเริ่มงานหนัก

        if not args.dry_run and git("status", "--porcelain", "--", ".").stdout.strip():
            raise StepError("task_b/ มีไฟล์ที่ยังไม่ commit -- official ต้องเริ่มจาก tree สะอาด")

        tag = datetime.now(BANGKOK).strftime("%Y%m%d")
        staging = BASE_DIR / ".staging" / f"yahoo_{tag}"
        rc = run("1 fetch", [PY, TOOLS / "fetch_yahoo_daily.py", "--out-dir", staging], ok_codes=(0, EXIT_NO_NEW))

        if rc == 0:
            files = sorted(staging.glob("*.csv"))
            run("2 append", [PY, TOOLS / "append_investing.py", *files, "--source", "yahoo", "--date-tag", tag])
            run("3 regression", [PY, "main.py", "--dev"])
            if git("diff", "--quiet", "--", VAL_GLOB, check=False).returncode != 0:
                raise StepError("results/*_val.csv เปลี่ยนหลัง append -- ข้อมูลเก่าถูกกระทบ ห้ามไปต่อ "
                                "(คืนไฟล์ด้วย git checkout -- raw_data results แล้วหาสาเหตุ)")
            print("[3 regression] ok: *_val.csv ไม่เปลี่ยน")
            if args.commit:
                git("add", "raw_data", "raw_data_sources")
                git("commit", "-m", f"data: task_b raw_data ถึง {data_cutoff().date()} (yahoo)")
        else:
            print("[1 fetch] ไม่มีวันใหม่ -- ข้ามขั้น 2-4")

        cutoff = data_cutoff()
        target = next_trading_day(cutoff, holidays, through)
        print(f"\ndata_cutoff = {cutoff.date()} -> target_date = {target.date()}")

        run("5 outcomes", [PY, "record_outcomes.py"])
        run("5 report", [PY, "report_live.py"])

        if not args.dry_run and already_predicted(target):
            print(f"[6 predict] มีคำทำนาย next_day ของ {target.date()} ใน log แล้ว -- ข้าม")
        else:
            cmd = [PY, "predict_live.py", "--target-date", target.date().isoformat(),
                   "--expected-cutoff", cutoff.date().isoformat()]
            run("6 predict", cmd + (["--dry-run"] if args.dry_run else []))

        if args.commit:
            git("add", "results")
            if git("diff", "--cached", "--quiet", check=False).returncode != 0:
                git("commit", "-m", f"prediction log: {target.date()}")
            if args.push:
                git("push")
                print("[7 push] ok")
    except StepError as e:
        print(f"\n!! {e}")
        return 1
    print("\nเสร็จ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
