@echo off
rem Manual run. Scheduled task SettradeDailyCollector runs pythonw.exe collect_settrade_daily.py --log ... directly (no console, so no Ctrl+C kill).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if not exist data\settrade_log mkdir data\settrade_log
"C:\Users\parin\AppData\Local\Programs\Python\Python312\python.exe" collect_settrade_daily.py >> data\settrade_log\scheduler_run.log 2>&1
