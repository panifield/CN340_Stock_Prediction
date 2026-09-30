#!/bin/bash
# ติดตั้ง/ถอนงาน launchd ของ macOS สำหรับ automation/run_1600_with_retry.sh
# เรียกด้วย: bash automation/install_1600_launchd_macos.sh
# ถอนด้วย:  bash automation/install_1600_launchd_macos.sh --uninstall

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.cn340.stockprediction.run1600"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$PLIST_DIR/$LABEL.plist"
USER_ID="$(id -u)"

if [[ "${1:-}" == "--uninstall" ]]; then
  launchctl bootout "gui/$USER_ID" "$PLIST_PATH" 2>/dev/null || true
  rm -f "$PLIST_PATH"
  echo "ถอน launchd job แล้ว: $LABEL"
  exit 0
fi

if [[ "$#" -ne 0 ]]; then
  echo "usage: bash automation/install_1600_launchd_macos.sh [--uninstall]"
  exit 64
fi

mkdir -p "$PLIST_DIR" "$ROOT/automation/.staging/logs"

ROOT="$ROOT" PLIST_PATH="$PLIST_PATH" python3 - <<'PY'
import os
import plistlib

root = os.environ["ROOT"]
plist_path = os.environ["PLIST_PATH"]
launch_agent = {
    "Label": "com.cn340.stockprediction.run1600",
    "ProgramArguments": ["/bin/bash", f"{root}/automation/run_1600_with_retry.sh"],
    "WorkingDirectory": root,
    "StartCalendarInterval": [
        {"Weekday": weekday, "Hour": 16, "Minute": 2}
        for weekday in range(1, 6)  # Monday–Friday; Sunday=0
    ],
    "RunAtLoad": False,
    "ProcessType": "Background",
    "StandardOutPath": f"{root}/automation/.staging/logs/launchd_1600.out.log",
    "StandardErrorPath": f"{root}/automation/.staging/logs/launchd_1600.err.log",
}
with open(plist_path, "wb") as file:
    plistlib.dump(launch_agent, file, sort_keys=False)
PY

# อัปเดตงานเดิมได้โดยไม่ error หากยังไม่ได้ติดตั้งมาก่อน
launchctl bootout "gui/$USER_ID" "$PLIST_PATH" 2>/dev/null || true
launchctl bootstrap "gui/$USER_ID" "$PLIST_PATH"
launchctl enable "gui/$USER_ID/$LABEL"

echo "ติดตั้ง launchd job แล้ว: $LABEL"
echo "schedule: จันทร์-ศุกร์ 16:02 ตาม timezone ของ macOS"
echo "official command: automation/run_1600_with_retry.sh -> --commit --push"
echo "log: $ROOT/automation/.staging/logs/run_1600_YYYYMMDD.log"
echo "หมายเหตุ: เครื่องต้องล็อกอินและตื่นอยู่; launchd จะไม่ปลุกเครื่องจาก sleep"
