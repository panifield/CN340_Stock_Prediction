"""
log_schema.py — เพิ่มคอลัมน์ใหม่ให้ prediction_log.csv โดยไม่แตะค่าของแถวเก่า

กฎ: ห้ามแก้/ลบแถวเก่าใน log · เพิ่มคอลัมน์ได้ แต่แถวเก่าต้องเป็นค่าว่าง
จึงแก้แบบข้อความล้วน: ต่อ "," ท้ายบรรทัดเดิม (ไม่ผ่าน pandas ที่อาจเปลี่ยนรูปแบบตัวเลข)
และยอมเฉพาะกรณี header ในไฟล์ = คอลัมน์เดิมเป๊ะ (old_columns) เท่านั้น
append_log() ของแต่ละ task ยังปฏิเสธ schema ที่ไม่ตรงเหมือนเดิม

ห้าม import โมดูลอื่นในโปรเจกต์ (task_a2 โหลดผ่าน sys.path ของ task_a)
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def migrate_log_columns(log_path, old_columns, new_columns):
    """
    ถ้า header = old_columns -> เขียน header ใหม่ = old_columns + new_columns และเติม
    ค่าว่างท้ายทุกแถวเดิม · ถ้า header = old+new อยู่แล้ว -> ไม่ทำอะไร · อย่างอื่น -> raise
    คืน True ถ้าย้าย schema จริง
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return False
    raw = log_path.read_bytes()
    if not raw:
        return False
    text = raw.decode("utf-8-sig") if raw.startswith(b"\xef\xbb\xbf") else raw.decode("utf-8")
    bom = raw.startswith(b"\xef\xbb\xbf")
    eol = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(eol)
    header = next(csv.reader([lines[0]]))
    target = list(old_columns) + list(new_columns)
    if header == target:
        return False
    if header != list(old_columns):
        raise RuntimeError(
            f"schema ของ {log_path.name} ไม่ใช่ชุดเดิมที่รู้จัก ไม่ย้าย schema ให้\n"
            f"  ในไฟล์: {header}\n  คาดว่า: {list(old_columns)}")

    pad = "," * len(new_columns)
    out = [",".join(target)]
    for line in lines[1:]:
        out.append(line + pad if line != "" else line)   # บรรทัดว่างท้ายไฟล์คงไว้
    new_text = eol.join(out)
    log_path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + new_text.encode("utf-8"))
    print(f"[log] เพิ่มคอลัมน์ {list(new_columns)} ให้ {log_path.name} (แถวเดิมเป็นค่าว่าง)")
    return True


def old_part_sha256(log_path, n_old_rows, n_new_cols=0):
    """
    sha256 ของ "ส่วนแถวเก่า" ใน log: header + n_old_rows แถวแรก หลังตัดคอลัมน์ใหม่ n_new_cols ตัวท้ายออก
    แปลง CRLF -> LF ก่อน (Windows เช็คเอาต์เป็น CRLF ส่วน Railway/git เก็บเป็น LF) จึงเทียบข้ามเครื่องได้
    """
    text = Path(log_path).read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
    lines = text.split("\n")[: n_old_rows + 1]
    if n_new_cols:
        lines = [line.rsplit(",", n_new_cols)[0] for line in lines]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


if __name__ == "__main__":
    # ตรวจหลัง deploy:  python task_a/log_schema.py <log> <จำนวนแถวเก่า> <จำนวนคอลัมน์ใหม่> <sha256 ที่คาด>
    import sys
    path, n_rows, n_new, expected = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    got = old_part_sha256(path, n_rows, n_new)
    print(f"{path}: {got}")
    print("OK แถวเก่าเหมือนเดิมทุก byte" if got == expected else "!! ไม่ตรง -- แถวเก่าถูกแก้")
    sys.exit(0 if got == expected else 1)
