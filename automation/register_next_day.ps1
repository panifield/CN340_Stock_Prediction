# automation/register_next_day.ps1 -- ลงทะเบียนงาน next_day ของทั้ง 3 tasks (task_a, task_b, task_c)
# ใน Windows Task Scheduler
#
#   powershell -ExecutionPolicy Bypass -File automation\register_next_day.ps1                # ลงทะเบียน (official)
#   powershell -ExecutionPolicy Bypass -File automation\register_next_day.ps1 -DryRun        # ทดสอบ (dry-run)
#   powershell -ExecutionPolicy Bypass -File automation\register_next_day.ps1 -Unregister
#
# ใช้ Task Scheduler แทน GitHub Actions cron เพราะ schedule ของ GitHub ดีเลย์ได้หลายชม. (เจอจริง ~5.5 ชม.
# 2 รอบติด) และยังพบว่า yfinance เองก็ backfill ราคาปิดช้ากว่าที่คิด (ยังเป็น NaN แม้ผ่าน 18:00+7ชม.ไปแล้ว)
# เลยเลื่อนมารันตอนเช้าแทนตอนหัวค่ำ -- ยังมีเวลาห่างจากตลาดปิด (16:30) เยอะกว่าเดิม และ task_b/fetch
# เช็ค NaN เองอยู่แล้ว (ไม่เขียนไฟล์ถ้าข้อมูลไม่ครบ) จึง retry ด้วยมือได้ปลอดภัยถ้ารอบนี้ยังไม่พร้อม
#
# งาน (จันทร์-ศุกร์ เวลาเครื่อง ต้องตั้งเป็น +07:00):
#   next_day all tasks   08:30  automation\run_next_day.py --commit --push
# log ของแต่ละรอบอยู่ที่ automation\.staging\logs\ (ไม่ถูก commit)
# ข้อกำหนด: เครื่องเปิดอยู่ช่วงเวลานั้น (หรือ -StartWhenAvailable รันย้อนหลังให้เมื่อเปิดเครื่อง/login
#           เพราะ next_day ไม่มีเส้นตายแคบเหมือน t1600) · git push ได้โดยไม่ต้องพิมพ์รหัส ·
#           ไม่มีไฟล์ค้างไม่ commit ใน task_b (daily_next_day.py เช็คเอง)
#
# ตั้ง WakeToRun ไว้ด้วย (ปลุกเครื่องจาก Sleep มารันเองได้ ไม่ต้องตื่นมาเปิดเครื่องทัน) แต่ต้องมี 2
# อย่างนี้เพิ่ม (เป็นค่า Windows เอง ตั้งในสคริปต์ไม่ได้ ต้องทำมือครั้งเดียว):
#   1. เสียบชาร์จไว้ (DisallowStartIfOnBatteries=True -- ใช้แบตอย่างเดียวจะไม่รัน)
#   2. Control Panel -> Power Options -> Change plan settings -> Change advanced power
#      settings -> Sleep -> Allow wake timers -> Enable
# เครื่องต้องอยู่ในโหมด Sleep ไม่ใช่ Shutdown ถึงจะปลุกได้ (ล็อกหน้าจอได้ปกติ ไม่กระทบ)

param(
    [string]$Python = (Get-Command python).Source,
    [switch]$DryRun,
    [switch]$Unregister
)

$Root = Split-Path -Parent $PSScriptRoot
$Name = "run_next_day all tasks"

if ($Unregister) {
    try { Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction Stop } catch {}
    Write-Output "ลบงานแล้ว: $Name"
    return
}

$LogDir = Join-Path $PSScriptRoot ".staging\logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null

$ModeArg = "--commit --push"
if ($DryRun) { $ModeArg = "--dry-run" }

$log = Join-Path $LogDir "run_next_day.log"
$cmd = "/c set PYTHONUTF8=1 && echo ===== %DATE% %TIME% >> `"$log`" && `"$Python`" automation\run_next_day.py $ModeArg >> `"$log`" 2>&1"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $cmd -WorkingDirectory $Root
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At "08:30"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null

Write-Output "ลงทะเบียน '$Name' 08:30 -> python automation\run_next_day.py $ModeArg"
Write-Output "log: $log"
