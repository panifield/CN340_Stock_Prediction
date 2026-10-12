#!/usr/bin/env bash
# deploy/railway_run.sh -- สคริปต์เริ่มงานของ Railway cron
# =========================================================================================
#   railway_run.sh auto        # service เดียวรับทั้งสองงาน: เลือกตามเวลาไทย (ดูด้านล่าง) <- ใช้ตัวนี้บน Railway
#   railway_run.sh next_day    # งานเช้า: automation/run_next_day.py  (task_a -> task_c -> task_b)
#   railway_run.sh t1600       # งาน 16:02: automation/run_1600.py    (task_a2 -> task_b -> task_c)
#   railway_run.sh smoke       # ทดสอบสภาพแวดล้อมอย่างเดียว (ไม่ทำนาย ไม่ commit)
#
# ลำดับ: เช็ค env -> clone repo ล่าสุด -> เช็ควัน/เวลา -> precheck (ทำเสร็จแล้วไหม / Yahoo พร้อมไหม) -> รัน orchestrator -> คืน exit code
# (cron ต้อง "ทำงานเสร็จแล้วจบ" -- ห้ามค้าง ไม่งั้นรอบถัดไปจะถูกข้าม)
#
# ตัวแปร environment (ตั้งใน Railway -> service -> Variables)
#   DRY_RUN=1        ค่าเริ่มต้น = ทดสอบ (--dry-run ไม่ commit/push) · ต้องตั้ง DRY_RUN=0 ถึงจะบันทึกจริง
#   GITHUB_TOKEN     token ที่มีสิทธิ์เขียน repo (ต้องมีเมื่อ DRY_RUN=0 และตอน smoke ที่อยากทดสอบสิทธิ์ push)
#   SMOKE_ONLY=1     ทุกโหมดรันแค่ smoke test แล้วจบ (ใช้ตอนทดสอบรอบแรก)
#   GIT_REPO         ค่าเริ่มต้น panifield/CN340_Stock_Prediction · GIT_BRANCH ค่าเริ่มต้น main
#   GIT_USER_NAME / GIT_USER_EMAIL   ตัวตนของ commit (ค่าเริ่มต้น railway-bot)
#   SKIP_TIME_CHECK=1 / SKIP_HOLIDAY_CHECK=1   ข้ามการเช็ควัน/เวลา (ใช้ตอนทดสอบเท่านั้น)
#   AUTO_FORCE_MODE=next_day|t1600  บังคับโหมดของ auto โดยไม่ดูเวลา (ทดสอบ)
#   NOT_READY_FINAL_HHMM  ค่าเริ่มต้น 950 -- next_day รอบที่เริ่มก่อนเวลานี้ ถ้า Yahoo ยังไม่พร้อมจะ "จบเงียบ (exit 0)" รอ cron รอบถัดไป
#                         รอบตั้งแต่เวลานี้เป็นต้นไปถ้ายังไม่พร้อมจะล้มเหลวจริง (exit 4)
#
# exit code: 0 สำเร็จ/ข้ามอย่างตั้งใจ · 1 clone หรือ task ล้มเหลว · 2 ตั้งค่าผิด · 3 เริ่มผิดเวลา · 4 ข้อมูลไม่พร้อมจนรอบสุดท้าย
#            5 ทำไปแล้วบางส่วน (ต้องมีคนดู) · 6 precheck ผิดพลาด
#   FAKE_NOW_BKK="YYYY-MM-DD HH:MM"   จำลองเวลาไทย (ใช้ทดสอบ logic วัน/เวลา)
set -euo pipefail

MODE="${1:-${RUN_MODE:-smoke}}"
DRY_RUN="${DRY_RUN:-1}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
GIT_REPO="${GIT_REPO:-panifield/CN340_Stock_Prediction}"
GIT_URL="${GIT_URL:-https://github.com/${GIT_REPO}.git}"
GIT_BRANCH="${GIT_BRANCH:-main}"
GIT_USER_NAME="${GIT_USER_NAME:-railway-bot}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-railway-bot@users.noreply.github.com}"
WORKDIR="${WORKDIR:-/app}"
PY="${PYTHON:-python}"
TMP="${TMPDIR:-/tmp}"

# เวลาไทย = UTC+7 (ไม่มี DST) -- คำนวณจาก UTC ตรง ๆ ไม่พึ่งฐานข้อมูล timezone ของระบบ (ซึ่งบางสภาพแวดล้อมไม่มี)
log() { echo "[railway_run $(date -u -d '+7 hours' '+%F %T') +07] $*"; }

# เวลาไทยปัจจุบัน (หรือเวลาจำลอง) รูปแบบ "YYYY-MM-DD HH:MM"
now_bkk() {
  if [ -n "${FAKE_NOW_BKK:-}" ]; then echo "$FAKE_NOW_BKK"; else date -u -d '+7 hours' '+%F %H:%M'; fi
}

setup_auth() {
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    # token ไม่ถูกใส่ใน URL / ไม่ถูกพิมพ์ลง log -- git ถามผ่าน askpass ซึ่งอ่านจาก env ตอนรัน
    cat > "$TMP/git-askpass.sh" <<'EOS'
#!/bin/sh
case "$1" in
  Username*) echo "x-access-token" ;;
  *) echo "$GITHUB_TOKEN" ;;
esac
EOS
    chmod +x "$TMP/git-askpass.sh"
    export GIT_ASKPASS="$TMP/git-askpass.sh" GIT_TERMINAL_PROMPT=0
  fi
}

clone_repo() {
  local dest="$WORKDIR/repo" i
  mkdir -p "$WORKDIR"
  for i in 1 2 3; do
    rm -rf "$dest"
    if git clone --quiet --depth 1 --branch "$GIT_BRANCH" "$GIT_URL" "$dest"; then
      git -C "$dest" config user.name "$GIT_USER_NAME"
      git -C "$dest" config user.email "$GIT_USER_EMAIL"
      return 0
    fi
    log "clone ล้มเหลว (ครั้งที่ $i/3) -- รอ 5 วินาทีแล้วลองใหม่"
    sleep 5
  done
  return 1
}

smoke() {
  log "== smoke test =="
  "$PY" --version
  "$PY" - <<'PY'
import sys
import numpy, pandas, sklearn, xgboost, lightgbm, yfinance as yf
print("versions: numpy", numpy.__version__, "| pandas", pandas.__version__, "| sklearn", sklearn.__version__,
      "| xgboost", xgboost.__version__, "| lightgbm", lightgbm.__version__, "| yfinance", yf.__version__)
bad = []
for tk in ("KBANK.BK", "ADVANC.BK"):
    for label, kw in (("1d", dict(period="10d", interval="1d")), ("1h", dict(period="5d", interval="1h"))):
        try:
            df = yf.download(tk, auto_adjust=False, progress=False, threads=False, **kw)
        except Exception as e:                      # noqa: BLE001 -- ต้องการรายงานทุกแบบของความล้มเหลว
            print(f"FAIL {tk} {label}: {type(e).__name__}: {e}")
            bad.append((tk, label))
            continue
        if df is None or len(df) == 0:
            print(f"FAIL {tk} {label}: Yahoo ไม่คืนข้อมูล (อาจถูกจำกัด IP ของ cloud)")
            bad.append((tk, label))
            continue
        if hasattr(df.columns, "get_level_values"):
            df.columns = df.columns.get_level_values(0)
        last = df.iloc[-1]
        note = "  (!) Close ล่าสุดเป็น NaN -- Yahoo ยังไม่ backfill (พบได้ปกติตอนเช้า)" if last["Close"] != last["Close"] else ""
        print(f"ok   {tk} {label}: {len(df)} แถว, ล่าสุด {df.index[-1]} close={last['Close']}{note}")
sys.exit(1 if bad else 0)
PY
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    log "ทดสอบสิทธิ์ GitHub (clone + push --dry-run ไม่เปลี่ยนอะไรบน remote)"
    clone_repo
    git -C "$WORKDIR/repo" push --dry-run origin "HEAD:$GIT_BRANCH"
    log "สิทธิ์ push ผ่าน"
  else
    log "ไม่มี GITHUB_TOKEN -- ข้ามการทดสอบสิทธิ์ GitHub"
  fi
  log "== smoke test ผ่าน =="
}

# ------------------------------------------------------------------ main
case "$MODE" in
  next_day|t1600|smoke|auto) ;;
  *) echo "usage: railway_run.sh {auto|next_day|t1600|smoke}" >&2; exit 2 ;;
esac

# เวลาไทย (หรือเวลาจำลอง) -- คำนวณก่อน clone เพราะโหมด auto ต้องตัดสินใจจากเวลาก่อน
NOW="$(now_bkk)"
TODAY="${NOW%% *}"
HHMM="$(echo "${NOW##* }" | tr -d ':')"
HHMM=$((10#$HHMM))
NOT_READY_FINAL_HHMM="${NOT_READY_FINAL_HHMM:-950}"   # next_day: รอบที่เริ่มตั้งแต่เวลานี้ (09:50) ถ้ายังไม่พร้อมถือว่าล้มเหลวจริง

log "mode=$MODE DRY_RUN=$DRY_RUN SMOKE_ONLY=$SMOKE_ONLY repo=$GIT_REPO@$GIT_BRANCH"
setup_auth

if [ "$MODE" = "smoke" ] || [ "$SMOKE_ONLY" = "1" ]; then
  smoke
  exit 0
fi

if [ "$DRY_RUN" = "0" ]; then
  ARGS=(--commit --push)
  if [ -z "${GITHUB_TOKEN:-}" ]; then
    log "ผิดพลาด: DRY_RUN=0 ต้องมี GITHUB_TOKEN (ต้อง push log กลับ GitHub ไม่งั้น log หายเมื่อ container ดับ)"
    exit 2
  fi
else
  ARGS=(--dry-run)
fi

# โหมด auto: service เดียวรับทั้งสองงาน cron ตั้งให้ยิงหลายรอบต่อวัน ให้เวลาไทยเป็นตัวเลือกงาน
#   08:00-09:59 -> next_day · 16:00-16:29 -> t1600 · เวลาอื่น -> จบทันที (ไม่ clone ไม่ทำอะไร)
# ผลข้างเคียงที่ดี: deploy/redeploy ตอนไหนก็ไม่เผลอรันงานจริงผิดเวลา
# AUTO_FORCE_MODE=next_day|t1600 บังคับโหมดโดยไม่ดูเวลา (ใช้ทดสอบ เช่น กด Run now ตอนกลางคืน) -- ตัวเช็คเวลาด้านล่างยังทำงานเมื่อ DRY_RUN=0
if [ "$MODE" = "auto" ]; then
  if [ -n "${AUTO_FORCE_MODE:-}" ]; then
    MODE="$AUTO_FORCE_MODE"
  elif [ "$HHMM" -ge 800 ] && [ "$HHMM" -lt 1000 ]; then
    MODE="next_day"
  elif [ "$HHMM" -ge 1600 ] && [ "$HHMM" -lt 1630 ]; then
    MODE="t1600"
  else
    log "auto: $NOW ไม่อยู่ในช่วงงาน (08:00-09:59 / 16:00-16:29) -- ไม่ทำอะไร (exit 0)"
    exit 0
  fi
  log "auto -> mode=$MODE"
fi
case "$MODE" in
  next_day) SCRIPT="automation/run_next_day.py" ;;
  t1600)    SCRIPT="automation/run_1600.py" ;;
  *) log "ผิดพลาด: AUTO_FORCE_MODE ต้องเป็น next_day หรือ t1600 (ได้ $MODE)"; exit 2 ;;
esac

clone_repo || { log "ผิดพลาด: clone ไม่สำเร็จ"; exit 1; }
cd "$WORKDIR/repo"
log "HEAD = $(git rev-parse --short HEAD) : $(git log -1 --format=%s)"

if [ "$DRY_RUN" = "0" ]; then
  # 1) วันหยุด/เสาร์-อาทิตย์ -> ข้าม (ไม่ถือว่าล้มเหลว) กันทำนายซ้ำ key เดิมในวันทำการถัดไป
  if [ "${SKIP_HOLIDAY_CHECK:-0}" != "1" ]; then
    if ! "$PY" - "$TODAY" <<'PY'
import sys
import pandas as pd
today = pd.Timestamp(sys.argv[1])
hol = set()
for line in open("task_b/set_holidays.txt", encoding="utf-8").read().splitlines():
    line = line.split("#", 1)[0].strip()
    if line and not line.startswith("confirmed_through:"):
        hol.add(pd.Timestamp(line))
if today.dayofweek > 4 or today in hol:
    print(f"{today.date()} ไม่ใช่วันทำการของ SET")
    sys.exit(10)
PY
    then
      log "ข้ามการรัน: $TODAY ไม่ใช่วันทำการ (exit 0)"
      exit 0
    fi
  fi

  # 2) เวลา -- ตัวคุมรัน task หลายตัวต่อกันและ commit ทีละตัว ถ้าเริ่มผิดเวลา task แรกอาจเขียน log ไปก่อน
  #    แล้ว task หลังปฏิเสธ (เช่น task_b ปฏิเสธนอก 16:00-16:30) ได้ log ครึ่ง ๆ กลาง ๆ -> กันตั้งแต่ต้น
  if [ "${SKIP_TIME_CHECK:-0}" != "1" ]; then
    if [ "$MODE" = "t1600" ]; then
      if [ "$HHMM" -lt 1600 ] || [ "$HHMM" -ge 1630 ]; then
        log "ยกเลิก: t1600 ต้องเริ่มระหว่าง 16:00-16:29 (เวลาไทย) ตอนนี้ $NOW -- ไม่รันเพื่อกัน log ที่เขียนไม่ครบทุก task"
        exit 3
      fi
    else
      if [ "$HHMM" -ge 1000 ] && [ "$HHMM" -lt 1800 ]; then
        log "ยกเลิก: next_day ห้ามรัน 10:00-17:59 (ตลาดเปิด/ข้อมูลปิดตลาดยังไม่นิ่ง) ตอนนี้ $NOW"
        exit 3
      fi
    fi
  fi

  # 3) ตรวจก่อนรันจริง (deploy/precheck.py ไม่เขียนอะไร): วันนี้ทำเสร็จแล้วหรือยัง / ข้อมูล Yahoo พร้อมไหม
  #    cron ตั้งให้ลองหลายรอบต่อวัน -> รอบที่ไม่พร้อมต้อง "จบเงียบ ๆ โดยไม่เขียนอะไร" แล้วรอรอบถัดไป
  #    (ห้ามปล่อยให้ task_a/task_c เดินต่อด้วยข้อมูลเก่า แล้ว task_b หยุด -> log ครึ่งเดียว และรันซ้ำไม่ได้)
  set +e
  "$PY" deploy/precheck.py "$MODE" --now "$NOW"
  PRC=$?
  set -e
  case "$PRC" in
    0)  ;;
    10) log "ข้าม: วันนี้ทำเสร็จครบทุก task แล้ว (exit 0)"; exit 0 ;;
    11) if [ "$MODE" = "next_day" ] && [ "$HHMM" -lt "$NOT_READY_FINAL_HHMM" ]; then
          log "ข้อมูลยังไม่พร้อม -- ไม่เขียนอะไร รอ cron รอบถัดไป (exit 0)"; exit 0
        fi
        log "ล้มเหลว: ข้อมูลยังไม่พร้อมจนถึงรอบสุดท้ายของวัน ($NOW) -- วันนี้ไม่มี log ต้องมีคนตรวจ (exit 4)"; exit 4 ;;
    12) log "ล้มเหลว: วันนี้ทำไปแล้วบางส่วน ต้องมีคนตรวจ ห้ามรันซ้ำอัตโนมัติ (exit 5)"; exit 5 ;;
    *)  log "ล้มเหลว: precheck มีปัญหา (exit $PRC) -- ไม่รัน (exit 6)"; exit 6 ;;
  esac
else
  # dry-run: แสดงผลตรวจให้ดูเฉยๆ ไม่หยุดการทดสอบ (ไว้ดูว่าถ้าเป็นรอบจริงจะเกิดอะไร)
  set +e
  "$PY" deploy/precheck.py "$MODE" --now "$NOW"
  log "(dry-run) precheck exit=$? -- ไม่มีผลต่อการรัน"
  set -e
fi

log "เริ่ม: python $SCRIPT ${ARGS[*]}"
set +e
"$PY" -X utf8 "$SCRIPT" "${ARGS[@]}"
RC=$?
set -e
log "จบ exit=$RC · git log:"
git log --oneline -3 || true
exit "$RC"
