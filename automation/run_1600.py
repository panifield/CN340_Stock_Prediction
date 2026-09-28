"""
automation/run_1600.py — รันกลุ่ม t1600 ของทุก task พร้อมกัน (task_a2, task_b, task_c)
=========================================================================================

    python automation/run_1600.py --dry-run
    python automation/run_1600.py --commit --push

ต้องรันในช่วง ~16:00–16:30 น. ไทย (วันทำการ) — ไม่มี logic โมเดลเอง แค่เรียก
สคริปต์ของแต่ละ task ตามลำดับ:

  1. task_a2/predict_live.py         (คู่/คี่ ณ 16:00)
  2. task_b/daily_1600.py --phase predict   (ราคา — wrapper ทางการของ task_b
                                        เอง: fetch -> predict -> availability
                                        check -> commit — เข้มงวดสุด)
  3. task_c/predict_live_1600.py     (ขึ้น/ลง ณ 16:00)

*** ความเข้มงวดของแต่ละ task ไม่เท่ากัน (สำคัญ อ่านก่อนใช้งานจริง) ***
task_b เช็คช่วงเวลา 16:00–16:30 ทั้งขอบบนขอบล่างเอง (ถ้า orchestrator นี้
รันช้าเกิน 16:30 ก่อนจะถึงขั้น task_b มันจะ fail ทันทีและ "หยุด ไม่ไปต่อ" —
แปลว่า task_a2/task_c ที่รันไปก่อนหน้าอาจเขียน log ไปแล้วแต่ task_b ไม่ได้
เขียน ต้องรู้ไว้ก่อน) ส่วน task_a2/task_c เช็คแค่ขอบล่าง (มีแท่ง cutoff
หรือยัง) ไม่มีขอบบนของเวลา — ดู KNOWN_ISSUES.md หัวข้อ "ความเข้มงวดของ t1600
guard ต่างกัน"

*** task_a2 และ task_c ไม่ commit เองเหมือน task_b/daily_1600.py ***
task_b/daily_1600.py commit ให้ตัวเองอยู่แล้วผ่าน --commit/--push (ส่งต่อ
flag ตรงๆ) แต่ task_a2/task_c ไม่มี wrapper แบบนี้ -- orchestrator นี้จึง
รับหน้าที่ commit/push แทนเฉพาะสองตัวนั้น

*** ยังไม่ครอบคลุม phase "outcome" ของ task_b ***
daily_1600.py --phase outcome (บันทึกผลจริงหลัง 17:00 เทียบกับที่ทำนายไว้)
เป็นคนละช่วงเวลากับ predict (16:00-16:30) ตั้งใจไม่รวมไว้ใน orchestrator นี้
ต้องตั้ง schedule แยกต่างหาก (ดู task_b/automation/register_tasks.ps1) --
task_a2/task_c ไม่มี phase "outcome" แบบนี้เลย (KNOWN_ISSUES.md)

หยุดทันทีถ้า task ไหนล้มเหลว
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent          # now/
PY = sys.executable


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
    """commit log ของ task ที่ไม่ commit ให้ตัวเอง — no-op ถ้าไม่มีอะไรใหม่ให้ commit"""
    git(step, ["add", log_rel_path], cwd)
    has_staged_changes = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=cwd
    ).returncode != 0
    if has_staged_changes:
        git(step, ["commit", "-m", f"{cwd.name} 1600 prediction log: {target}"], cwd)
        print(f"[{step}] committed {log_rel_path}")
        if push:
            git(step, ["push"], cwd)
            print(f"[{step}] pushed")
    else:
        print(f"[{step}] ไม่มีอะไรใหม่ให้ commit ({log_rel_path})")


def main(argv=None):
    ap = argparse.ArgumentParser(description="เรียก predict ของกลุ่ม t1600 ทุก task")
    ap.add_argument("--dry-run", action="store_true",
                    help="ทดสอบทุก task แบบ dry-run ไม่ commit อะไรเลย")
    ap.add_argument("--commit", action="store_true",
                    help="รันแบบ official ของทุก task + commit prediction log")
    ap.add_argument("--push", action="store_true",
                    help="git push หลัง commit (ต้องใช้กับ --commit)")
    args = ap.parse_args(argv)
    if args.dry_run and args.commit:
        ap.error("--dry-run ใช้กับ --commit ไม่ได้")
    if args.push and not args.commit:
        ap.error("--push ต้องใช้กับ --commit")
    if not args.dry_run and not args.commit:
        ap.error("ต้องระบุ --dry-run หรือ --commit อย่างใดอย่างหนึ่ง")

    today_iso = pd.Timestamp.now(tz="Asia/Bangkok").normalize().date().isoformat()
    print(f"target_date (วันนี้) = {today_iso}")

    try:
        # 1) task_a2
        cmd = [PY, "predict_live.py", "--target-date", today_iso]
        run("task_a2", cmd + (["--dry-run"] if args.dry_run else []), BASE_DIR / "task_a2")
        if args.commit:
            commit_log("task_a2 commit", BASE_DIR / "task_a2",
                      "results/prediction_log.csv", today_iso, args.push)

        # 2) task_b -- ใช้ daily_1600.py (wrapper ทางการของ task_b เอง: fetch ->
        # predict -> tools/check_snapshot_1600.py availability check -> commit) แทนที่จะ
        # เรียก fetch_yahoo_intraday.py + predict_1600.py ตรงๆ เอง (แบบเดิม) เพราะแบบเดิม
        # ข้ามขั้น availability check ไปเฉยๆ และมี commit logic ซ้ำซ้อนกับที่ daily_1600.py
        # ทำเองอยู่แล้ว -- แก้ตาม pattern เดียวกับที่ run_next_day.py เรียก daily_next_day.py
        # (2026-09-29)
        cmd = [PY, "daily_1600.py", "--phase", "predict"]
        if args.commit:
            cmd += ["--official", "--commit"] + (["--push"] if args.push else [])
        run("task_b", cmd, BASE_DIR / "task_b")

        # 3) task_c
        cmd = [PY, "predict_live_1600.py", "--target-date", today_iso]
        run("task_c", cmd + (["--dry-run"] if args.dry_run else []), BASE_DIR / "task_c")
        if args.commit:
            commit_log("task_c commit", BASE_DIR / "task_c",
                      "results/prediction_log_1600.csv", today_iso, args.push)

    except StepError as e:
        print(f"\n!! {e}")
        return 1
    print("\nเสร็จ — รันครบทุก task ของกลุ่ม t1600 แล้ว")
    return 0


if __name__ == "__main__":
    sys.exit(main())
