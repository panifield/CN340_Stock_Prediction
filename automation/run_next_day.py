"""
automation/run_next_day.py — รันกลุ่ม next_day ของทุก task พร้อมกัน (task_a, task_b, task_c)
==============================================================================================

    python automation/run_next_day.py --dry-run
    python automation/run_next_day.py --commit --push

รันหลังตลาดปิดจริง (แนะนำหลัง 18:00 +07:00 ตามที่ task_b/daily_next_day.py
กำหนดไว้) ของวันทำการ — ตัวนี้ไม่มี logic โมเดลเอง แค่เรียกสคริปต์ของแต่ละ
task ตามลำดับ:

  1. task_a/predict_live.py       (คู่/คี่)
  2. task_c/predict_live.py       (ขึ้น/ลง)
  3. task_b/daily_next_day.py     (ราคา — มี orchestration ของตัวเองอยู่แล้ว
                                    ครบทั้ง fetch/append/regression-check/commit)

target_date (วันทำการถัดไป) คำนวณจาก task_b/set_holidays.txt เพราะเป็น
ปฏิทินวันหยุด SET เดียวที่มีในเรโป — ใช้ร่วมกับ task_a/task_c ได้เพราะหุ้น
ทั้ง 4 ตัวเทรดในตลาดเดียวกัน (SET)

task_a และ task_c มี check_market_closed() กันรันช่วง 10:00-18:00 เอง (ตลาดอาจ
เปิดอยู่ หรือเพิ่งปิดไม่ถึง 2 ชม. ข้อมูลยังไม่ settle) แก้ให้ครอบคลุมกรณีรันข้าม
เที่ยงคืนตอนเช้ามืดด้วยแล้ว (เดิมเช็คแค่ now.hour < 18 ซึ่งบล็อกตอนเช้าผิดพลาด)

*** task_a และ task_c ไม่ commit เองเหมือน task_b ***
predict_live.py ของสองตัวนี้แค่เขียนไฟล์แล้วพิมพ์เตือนให้ commit เอง —
orchestrator นี้จึงรับหน้าที่ commit/push ให้แทนเมื่อสั่ง --commit/--push

หยุดทันทีถ้า task ไหนล้มเหลว — ไม่ข้ามไป task ถัดไป (กันสับสนว่า task ไหนมี
prediction ของวันนั้นแล้วบ้าง)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ต้องตั้งก่อน import/subprocess อื่นใด -- รันผ่าน Task Scheduler บน Windows พบว่า
# "cmd.exe /c set PYTHONUTF8=1 && ... && python.exe ..." แบบ inline ทำให้ python.exe
# fatal error "invalid PYTHONUTF8 environment variable value" (เหตุผลไม่ชัดเจน อาจเป็น
# quirk เฉพาะเครื่อง) เปลี่ยนมาตั้งจากใน python เองแทน (ยืนยันแล้วว่า os.environ ตรงนี้
# propagate ไปถึง subprocess ลูกหลานได้ปกติแม้รันผ่าน Task Scheduler) -- จำเป็นเพราะ
# task_a/task_c/task_b พิมพ์ข้อความไทยออก stdout ซึ่งพังด้วย cp1252 (ค่า default ของ
# console บนเครื่องนี้) ถ้าไม่มีตัวนี้ (แก้ 2026-10-01)
os.environ.setdefault("PYTHONUTF8", "1")

BASE_DIR = Path(__file__).resolve().parent.parent          # now/
TASK_B_HOLIDAYS = BASE_DIR / "task_b" / "set_holidays.txt"
PY = sys.executable
MARKET_OPEN_HOUR = 10   # เดียวกับ task_a/task_c: check_market_closed()


class StepError(RuntimeError):
    pass


def run(step, args, cwd):
    print(f"\n{'=' * 78}\n[{step}] (cwd={cwd.name}) {' '.join(str(a) for a in args)}\n{'=' * 78}",
          flush=True)
    rc = subprocess.run(args, cwd=cwd).returncode
    if rc != 0:
        raise StepError(f"ขั้น {step} ล้มเหลว (exit {rc}) -- หยุด ไม่ไปต่อ task ถัดไป")


def git(step, args, cwd):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise StepError(f"git {' '.join(args)} ({step}) ล้มเหลว:\n{r.stderr.strip()}")
    return r


def commit_log(step, cwd, log_rel_path, target, push):
    """commit log ของ task ที่ไม่ commit ให้ตัวเอง (task_a, task_c) — no-op ถ้าไม่มีอะไรใหม่ให้ commit"""
    git(step, ["add", log_rel_path], cwd)
    has_staged_changes = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=cwd
    ).returncode != 0
    if has_staged_changes:
        git(step, ["commit", "-m", f"{cwd.name} prediction log: {target}"], cwd)
        print(f"[{step}] committed {log_rel_path}")
        if push:
            git(step, ["push"], cwd)
            print(f"[{step}] pushed")
    else:
        print(f"[{step}] ไม่มีอะไรใหม่ให้ commit ({log_rel_path})")


# ---------------------------------------------------------------
# วันทำการ (ยืมปฏิทินวันหยุดของ task_b — ตลาดเดียวกัน)
# ---------------------------------------------------------------

def read_holidays(path):
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
        raise StepError(f"{path.name}: ไม่มีบรรทัด confirmed_through:")
    return holidays, through


def next_trading_day(after, holidays, through):
    d = pd.Timestamp(after) + pd.Timedelta(days=1)
    while d.dayofweek > 4 or d in holidays:
        d += pd.Timedelta(days=1)
    if d > through:
        raise StepError(
            f"วันทำการถัดไปที่คำนวณได้ ({d.date()}) เกิน confirmed_through ({through.date()}) ใน "
            f"{TASK_B_HOLIDAYS} -> เพิ่มวันหยุด SET จากประกาศทางการ แล้วเลื่อน confirmed_through"
        )
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description="เรียก predict_live ของกลุ่ม next_day ทุก task")
    ap.add_argument("--dry-run", action="store_true",
                    help="ทดสอบทุก task แบบ dry-run ไม่ commit อะไรเลย")
    ap.add_argument("--commit", action="store_true",
                    help="commit prediction log ของทุก task (ต้องไม่ใช้กับ --dry-run)")
    ap.add_argument("--push", action="store_true",
                    help="git push หลัง commit (ต้องใช้กับ --commit)")
    args = ap.parse_args(argv)
    if args.dry_run and args.commit:
        ap.error("--dry-run ใช้กับ --commit ไม่ได้")
    if args.push and not args.commit:
        ap.error("--push ต้องใช้กับ --commit")
    if not args.dry_run and not args.commit:
        ap.error("ต้องระบุ --dry-run หรือ --commit อย่างใดอย่างหนึ่ง")

    try:
        holidays, through = read_holidays(TASK_B_HOLIDAYS)
        now = pd.Timestamp.now(tz="Asia/Bangkok")
        today = now.normalize().tz_localize(None)
        # ถ้ารันตอนเช้ามืดก่อนตลาดเปิด (เช่น automation/register_next_day.ps1 ตั้งไว้
        # 08:30) ข้อมูลที่นิ่งล่าสุดคือของ "เมื่อวาน" (วันนี้ยังไม่เปิดตลาด) ต้องถอย
        # reference กลับ 1 วันปฏิทินก่อนหา next_trading_day มิฉะนั้นจะได้ target เกิน
        # ไป 1 วันทำการเสมอ (พบจริงตอนทดสอบ 2026-10-01: วันนี้=10-01 หลุดไปคำนวณ
        # target=10-02 ทั้งที่ data_cutoff จริงยังเป็น 09-29 เพราะยังไม่ได้ fetch ของ
        # 09-30 เลยด้วยซ้ำ -- ควรได้ target=09-30 ไม่ใช่ 10-02 แก้ 2026-10-01)
        reference = today - pd.Timedelta(days=1) if now.hour < MARKET_OPEN_HOUR else today
        target = next_trading_day(reference, holidays, through)
        print(f"วันนี้ = {today.date()} -> target_date (วันทำการถัดไป) = {target.date()}")
        target_iso = target.date().isoformat()

        # 1) task_a
        cmd = [PY, "predict_live.py", "--target-date", target_iso]
        run("task_a", cmd + (["--dry-run"] if args.dry_run else []), BASE_DIR / "task_a")
        if args.commit:
            commit_log("task_a commit", BASE_DIR / "task_a",
                      "results/prediction_log.csv", target_iso, args.push)

        # 2) task_c
        cmd = [PY, "predict_live.py", "--target-date", target_iso]
        run("task_c", cmd + (["--dry-run"] if args.dry_run else []), BASE_DIR / "task_c")
        if args.commit:
            commit_log("task_c commit", BASE_DIR / "task_c",
                      "results/prediction_log.csv", target_iso, args.push)

        # 3) task_b -- มี orchestration ของตัวเองครบอยู่แล้ว (fetch/append/regression/commit)
        cmd = [PY, "daily_next_day.py"]
        cmd += ["--dry-run"] if args.dry_run else ["--commit"] + (["--push"] if args.push else [])
        run("task_b", cmd, BASE_DIR / "task_b")

    except StepError as e:
        print(f"\n!! {e}")
        return 1
    print("\nเสร็จ — รันครบทุก task ของกลุ่ม next_day แล้ว")
    return 0


if __name__ == "__main__":
    sys.exit(main())
