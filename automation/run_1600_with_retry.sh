#!/bin/bash
# macOS launcher: รัน orchestrator 16:00 หนึ่งครั้ง และ retry อีกหนึ่งครั้งเมื่อ fail.
# launchd เรียกไฟล์นี้เวลา 16:02 วันจันทร์-ศุกร์; ห้ามใช้เพื่อ bypass time guard
# ของ task_b ซึ่งยังบังคับช่วง 16:00-16:30 เวลาไทยตามเดิม.

set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/automation/.staging/logs"
mkdir -p "$LOG_DIR"

# ใช้วันตามไทยทั้งในชื่อ log และตอนเช็กว่ามีเวลาพอ retry หรือไม่
RUN_DATE="$(TZ=Asia/Bangkok date +%Y%m%d)"
LOG_FILE="$LOG_DIR/run_1600_${RUN_DATE}.log"
exec >> "$LOG_FILE" 2>&1

echo ""
echo "===== $(TZ=Asia/Bangkok date '+%F %T %Z') : run_1600 launcher started ====="

if [[ -x "$ROOT/venv/bin/python" ]]; then
  PYTHON="$ROOT/venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi

# Scheduled job ต้องใช้ official mode. การเรียกด้วยมือรองรับ --dry-run เท่านั้น
# เพื่อทดสอบตัว launcher โดยไม่ commit/push.
if [[ "${1:-}" == "--dry-run" ]]; then
  MODE_ARGS=(--dry-run)
elif [[ "$#" -eq 0 ]]; then
  MODE_ARGS=(--commit --push)
else
  echo "usage: $0 [--dry-run]"
  exit 64
fi

run_attempt() {
  local attempt="$1"
  echo "[launcher] attempt ${attempt}/2: ${PYTHON} automation/run_1600.py ${MODE_ARGS[*]}"
  (
    cd "$ROOT" || exit 1
    PYTHONUTF8=1 "$PYTHON" automation/run_1600.py "${MODE_ARGS[@]}"
  )
  return $?
}

# ไม่เริ่ม retry หากจะชนเส้นตาย 16:30 ของ task_b; ต้องเริ่มไม่เกิน 16:25
# เพื่อเหลือเวลาอย่างน้อย 5 นาทีให้ task_b ตรวจ availability/เขียนผล.
retry_window_open() {
  local hour minute
  hour="$(TZ=Asia/Bangkok date +%H)"
  minute="$(TZ=Asia/Bangkok date +%M)"
  hour=$((10#$hour))
  minute=$((10#$minute))
  [[ "$hour" -lt 16 || ( "$hour" -eq 16 && "$minute" -le 25 ) ]]
}

if run_attempt 1; then
  echo "[launcher] attempt 1 สำเร็จ"
  exit 0
fi

echo "[launcher] attempt 1 ล้มเหลว"
if ! retry_window_open; then
  echo "[launcher] ไม่ retry: เวลาไทยเกิน 16:25 แล้ว เพื่อไม่ชน deadline 16:30"
  exit 1
fi

echo "[launcher] รอ 180 วินาที แล้ว retry 1 ครั้ง"
sleep 180

if ! retry_window_open; then
  echo "[launcher] ไม่ retry: หลังรอแล้วเวลาไทยเกิน 16:25"
  exit 1
fi

if run_attempt 2; then
  echo "[launcher] attempt 2 สำเร็จ"
  exit 0
fi

echo "[launcher] attempt 2 ล้มเหลว — โปรดตรวจ ${LOG_FILE}"
exit 1
