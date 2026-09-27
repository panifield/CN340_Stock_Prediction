# task_b/automation/register_tasks.ps1 -- ลงทะเบียนงานประจำวันของ task_b ใน Windows Task Scheduler
#
#   powershell -ExecutionPolicy Bypass -File task_b\automation\register_tasks.ps1                 # ลงทะเบียน (16:00 = official)
#   powershell -ExecutionPolicy Bypass -File task_b\automation\register_tasks.ps1 -DryRun1600     # 16:00 แบบ dry-run (ทดสอบ)
#   powershell -ExecutionPolicy Bypass -File task_b\automation\register_tasks.ps1 -Unregister
#
# งาน (จันทร์-ศุกร์ เวลาเครื่อง ต้องตั้งเป็น +07:00):
#   task_b next_day      18:30  daily_next_day.py --commit --push
#   task_b 1600 predict  16:02  daily_1600.py --phase predict --official --commit --push
#   task_b 1600 outcome  17:15  daily_1600.py --phase outcome --official --commit --push
# log ของแต่ละรอบอยู่ที่ task_b\.staging\logs\ (ไม่ถูก commit)
# ข้อกำหนด: เครื่องเปิดอยู่ช่วงเวลานั้น · git push ได้โดยไม่ต้องพิมพ์รหัส · ไม่มีไฟล์ค้างไม่ commit ใน task_b
# (โมเดล 16:00 มีเส้นตาย 16:30 -> ต้องใช้ scheduler ในเครื่อง ไม่ใช่ cron ของ GitHub ที่ช้าได้เป็นชั่วโมง)

param(
    [string]$Python = (Get-Command python).Source,
    [switch]$DryRun1600,
    [switch]$Unregister
)

$TaskB = Split-Path -Parent $PSScriptRoot
$Names = @("task_b next_day", "task_b 1600 predict", "task_b 1600 outcome")

if ($Unregister) {
    foreach ($n in $Names) { try { Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction Stop } catch {} }
    Write-Output "ลบงานแล้ว: $($Names -join ', ')"
    return
}

$LogDir = Join-Path $TaskB ".staging\logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$Official = " --official"
if ($DryRun1600) { $Official = "" }

$Jobs = @(
    @{ Name = $Names[0]; At = "18:30"; Args = "daily_next_day.py --commit --push"; Log = "next_day" },
    @{ Name = $Names[1]; At = "16:02"; Args = "daily_1600.py --phase predict$Official --commit --push"; Log = "1600_predict" },
    @{ Name = $Names[2]; At = "17:15"; Args = "daily_1600.py --phase outcome$Official --commit --push"; Log = "1600_outcome" }
)

foreach ($j in $Jobs) {
    $log = Join-Path $LogDir "$($j.Log).log"
    $cmd = "/c set PYTHONUTF8=1 && echo ===== %DATE% %TIME% >> `"$log`" && `"$Python`" $($j.Args) >> `"$log`" 2>&1"
    $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $cmd -WorkingDirectory $TaskB
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $j.At
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 45)
    Register-ScheduledTask -TaskName $j.Name -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
    Write-Output "ลงทะเบียน '$($j.Name)' $($j.At) -> python $($j.Args)"
}
Write-Output "log: $LogDir"
