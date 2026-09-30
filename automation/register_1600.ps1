# automation/register_1600.ps1 -- ลงทะเบียนงาน t1600 ของทั้ง 3 tasks (task_a2, task_b, task_c)
# ใน Windows Task Scheduler
#
#   powershell -ExecutionPolicy Bypass -File automation\register_1600.ps1                # ลงทะเบียน (official)
#   powershell -ExecutionPolicy Bypass -File automation\register_1600.ps1 -DryRun        # ทดสอบ (dry-run)
#   powershell -ExecutionPolicy Bypass -File automation\register_1600.ps1 -Unregister
#
# ใช้ Task Scheduler แทน GitHub Actions เพราะ t1600 มีเส้นตาย 16:00-16:30 แคบเกินกว่าจะทน cron ของ
# GitHub ที่ช้าได้เป็นชั่วโมง (เหตุผลเดียวกับที่ task_b/automation/register_tasks.ps1 ใช้อยู่แล้วสำหรับ
# task_b เอง -- ไฟล์นี้ครอบคลุมทั้ง 3 tasks ผ่าน automation/run_1600.py แทนที่จะลงแค่ task_b)
#
# งาน (จันทร์-ศุกร์ เวลาเครื่อง ต้องตั้งเป็น +07:00):
#   next_day all tasks   16:02  automation\run_1600.py --commit --push
# log ของแต่ละรอบอยู่ที่ automation\.staging\logs\ (ไม่ถูก commit)
# ข้อกำหนด: เครื่องเปิดอยู่ช่วงเวลานั้น (16:00-16:30) · git push ได้โดยไม่ต้องพิมพ์รหัส ·
#           ไม่มีไฟล์ค้างไม่ commit ใน task_b (predict_1600.py เช็คเอง)
#
# ตั้ง WakeToRun ไว้ด้วย (ปลุกเครื่องจาก Sleep มารันเองได้ ไม่ต้องเปิดเครื่องทัน) แต่ต้องมี 2 อย่างนี้
# เพิ่ม (เป็นค่า Windows เอง ตั้งในสคริปต์ไม่ได้ ต้องทำมือครั้งเดียว):
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
$Name = "run_1600 all tasks"

if ($Unregister) {
    try { Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction Stop } catch {}
    Write-Output "ลบงานแล้ว: $Name"
    return
}

$LogDir = Join-Path $PSScriptRoot ".staging\logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null

$ModeArg = "--commit --push"
if ($DryRun) { $ModeArg = "--dry-run" }

$log = Join-Path $LogDir "run_1600.log"
$cmd = "/c set PYTHONUTF8=1 && echo ===== %DATE% %TIME% >> `"$log`" && `"$Python`" automation\run_1600.py $ModeArg >> `"$log`" 2>&1"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $cmd -WorkingDirectory $Root
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At "16:02"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null

Write-Output "ลงทะเบียน '$Name' 16:02 -> python automation\run_1600.py $ModeArg"
Write-Output "log: $log"
