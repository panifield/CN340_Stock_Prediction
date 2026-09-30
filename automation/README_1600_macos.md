# ตั้งรัน 16:02 อัตโนมัติบน macOS

ไฟล์นี้ตั้ง `launchd` ให้เรียก `run_1600.py --commit --push` ทุกจันทร์–ศุกร์
เวลา 16:02 ตาม timezone ของ macOS หากรอบแรก error จะรอ 3 นาทีและลองใหม่
**อีก 1 ครั้ง** โดย retry จะไม่เริ่มหลัง 16:25 เพื่อหลีกเลี่ยง deadline 16:30
ของ `task_b`.

## ติดตั้ง

จาก root ของ repo:

```bash
bash automation/install_1600_launchd_macos.sh
```

เช็กว่างานถูกลงทะเบียน:

```bash
launchctl print gui/$(id -u)/com.cn340.stockprediction.run1600
```

ถอนการตั้งเวลา:

```bash
bash automation/install_1600_launchd_macos.sh --uninstall
```

## สิ่งที่ต้องเตรียม

- macOS timezone ต้องเป็น `Asia/Bangkok` เพื่อให้ 16:02 ตรงเวลาไทย
- เครื่องต้องเปิด, ล็อกอินบัญชีเดิม และ **ไม่ sleep** ในช่วง 16:00–16:30
- internet ต้องใช้งานได้ และ `git push` ต้องยืนยันตัวตนได้โดยไม่ถามรหัส
- หลีกเลี่ยงไฟล์ค้างที่ยังไม่ commit โดยเฉพาะใน `task_b`

## Log และ retry

log รายวันอยู่ที่:

```text
automation/.staging/logs/run_1600_YYYYMMDD.log
```

ระบบ retry เป็นการเรียก `automation/run_1600.py` ใหม่ทั้งชุด ไม่ใช่ retry
เฉพาะ task ที่ fail เพราะ orchestrator ปัจจุบันไม่มี checkpoint ต่อ task.
หากรอบแรกบาง task เขียน/commit prediction ไปแล้วก่อน task ถัดไป fail จึงควร
ตรวจ log หลังจากเกิด retry เพื่อยืนยันว่าไม่มี prediction ซ้ำ.

ทดสอบ launcher เองแบบไม่ commit/push:

```bash
bash automation/run_1600_with_retry.sh --dry-run
```

อย่างไรก็ตาม `--dry-run` ยังผ่าน time/data guard ของแต่ละ task; ไม่ควรใช้
เพื่อ bypass ช่วง 16:00–16:30.
