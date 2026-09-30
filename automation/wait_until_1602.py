"""รอจนถึง 16:02 เวลาไทย แล้วเรียก run_1600 หนึ่งครั้ง.

ใช้เมื่ออยากเปิด terminal ทิ้งไว้เอง แทนการใช้ launchd:

    venv/bin/python automation/wait_until_1602.py
    venv/bin/python automation/wait_until_1602.py --dry-run

หากเริ่มก่อน 16:02 ในวันทำการ จะรอถึง 16:02 ของวันนั้น; หากเริ่มหลังเวลา
ดังกล่าวหรือวันหยุดสุดสัปดาห์ จะรอถึงวันจันทร์-ศุกร์ถัดไป เวลา 16:02.
ไม่ควรเปิดพร้อมกับ launchd job ``com.cn340.stockprediction.run1600`` เพราะ
ทั้งสองตัวจะเรียก run_1600 ซ้ำกัน.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, time as clock_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


BANGKOK = ZoneInfo("Asia/Bangkok")
RUN_TIME = clock_time(hour=16, minute=2)
ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "automation" / "run_1600_with_retry.sh"


def next_run_time(now: datetime) -> datetime:
    """คืนเวลา 16:02 ของวันทำการถัดไปตามเวลาไทย."""
    target = now.replace(hour=RUN_TIME.hour, minute=RUN_TIME.minute,
                         second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    while target.weekday() >= 5:  # Saturday=5, Sunday=6
        target += timedelta(days=1)
    return target


def wait_until(target: datetime) -> None:
    """รอเป็นช่วงสั้น ๆ เพื่อให้ Ctrl-C หยุดได้ทันทีพอสมควร."""
    while True:
        remaining = (target - datetime.now(BANGKOK)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 60))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="รอถึง 16:02 เวลาไทย แล้วเรียก automation/run_1600.py"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="ส่ง --dry-run ให้ run_1600 (ไม่ commit/push)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not LAUNCHER.exists():
        print(f"ไม่พบ launcher: {LAUNCHER}", file=sys.stderr)
        return 1

    target = next_run_time(datetime.now(BANGKOK))
    print("กำลังรอ run_1600...")
    print(f"จะเริ่ม: {target.strftime('%A %d %B %Y %H:%M %Z')}")
    print("กด Ctrl-C เพื่อยกเลิก; เครื่องต้องไม่ sleep")
    try:
        wait_until(target)
    except KeyboardInterrupt:
        print("\nยกเลิกการรอแล้ว")
        return 130

    command = ["/bin/bash", str(LAUNCHER)]
    if args.dry_run:
        command.append("--dry-run")
    print(f"ถึงเวลา 16:02 — เริ่ม {' '.join(command)}")
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
