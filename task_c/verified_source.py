"""
โหลด task_a/verified_daily.py (ราคารายวันที่ตรวจแล้ว) จาก path ตรงๆ

ใช้ sys.path ไม่ได้ เพราะ task_c มี config.py / features.py / data_loader.py ชื่อซ้ำกับ task_a
"""

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parent.parent / "task_a" / "verified_daily.py"
_spec = importlib.util.spec_from_file_location("task_a_verified_daily", _path)
verified_daily = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verified_daily)
