"""
tools/check_env.py — ตรวจว่า library ที่ติดตั้งตรงกับเวอร์ชันที่ task_b pin ไว้

    python tools/check_env.py            # exit 0 = ตรงทุกตัว · exit 1 = มีตัวไม่ตรง/ไม่ได้ติดตั้ง

ผลของโมเดลขึ้นกับเวอร์ชัน (default ของ library เปลี่ยนได้) -> task_b/requirements.txt pin เป๊ะ
เมื่อ merge กับ task อื่นแล้วลง dependency ร่วมกัน ให้รันไฟล์นี้ก่อน main.py / predict_live.py
(predict_live.py / predict_1600.py แบบ official เรียกตรวจนี้เองและหยุดถ้าไม่ตรง)
ถ้าต้องเปลี่ยนเวอร์ชันจริง ให้ทำตาม INTEGRATION.md ข้อ 5-6 (regression check แล้วบันทึกผล)
"""

import re
import sys
from importlib import metadata
from pathlib import Path

REQ_PATH = Path(__file__).resolve().parent.parent / "requirements.txt"


def read_pins(path=REQ_PATH):
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([^\s;]+)", line)
        if not m:
            raise ValueError(f"{Path(path).name}: บรรทัดที่ไม่ใช่ pin แบบ == : {line!r}")
        pins[m.group(1)] = m.group(2)
    return pins


def check(pins=None, version=metadata.version):
    """คืน list ของปัญหา (ว่าง = ตรงทุกตัว)"""
    problems = []
    for name, want in (pins or read_pins()).items():
        try:
            got = version(name)
        except metadata.PackageNotFoundError:
            problems.append(f"{name}: ไม่ได้ติดตั้ง (ต้องการ {want})")
            continue
        if got != want:
            problems.append(f"{name}: ติดตั้ง {got} แต่ task_b pin {want}")
    return problems


def require_pinned_env():
    """ใช้ใน official path: raise ถ้าเวอร์ชันไม่ตรง"""
    problems = check()
    if problems:
        raise RuntimeError("เวอร์ชัน library ไม่ตรงกับ task_b/requirements.txt -- ห้ามบันทึก official prediction\n  "
                           + "\n  ".join(problems)
                           + "\n  แก้: pip install -r task_b/requirements.txt (ควรใช้ venv แยกของ task_b)")


def main():
    problems = check()
    if problems:
        print("!! เวอร์ชัน library ไม่ตรงกับ task_b/requirements.txt -- ผลอาจไม่ตรงกับที่บันทึกไว้")
        for p in problems:
            print(f"   {p}")
        print("   แก้: pip install -r task_b/requirements.txt  (ควรใช้ venv แยกของ task_b)")
        return 1
    print(f"ok: เวอร์ชันตรง task_b/requirements.txt ครบ {len(read_pins())} ตัว")
    return 0


if __name__ == "__main__":
    sys.exit(main())
