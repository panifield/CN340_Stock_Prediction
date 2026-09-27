@echo off
cd /d "C:\StudyUniversity\CN340\group_proj\task_a2"
set PYTHONIOENCODING=utf-8
"C:\Users\parin\AppData\Local\Programs\Python\Python312\python.exe" collect_settrade_daily.py >> data\settrade_log\scheduler_run.log 2>&1
