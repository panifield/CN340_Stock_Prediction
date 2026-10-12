# deploy/ — รันงานรายวันบน Railway (cron)

ไฟล์ที่เกี่ยวข้อง (อยู่ที่ root ของ repo ยกเว้นไฟล์นี้กับสคริปต์):
`Dockerfile` · `railway.toml` (งานเช้า) · `railway.t1600.toml` (งาน 16:02) · `deploy/railway_run.sh` · `.dockerignore` · `.gitattributes`

## แนวคิด
Railway เป็น cron: รันคำสั่ง → ทำงานเสร็จ → จบ (ห้ามค้าง ไม่งั้นรอบถัดไปถูกข้าม) และ filesystem ไม่ถาวร
จึงให้ `railway_run.sh` **git clone repo ล่าสุดทุกครั้ง** แล้วรัน `automation/run_next_day.py` หรือ `run_1600.py`
log ที่เขียนจะถูก `git push` กลับ GitHub (ถ้าไม่ push log หายเมื่อ container ดับ)

| service | config | เวลาไทย | cron (UTC) | คำสั่ง |
|---|---|---|---|---|
| next_day | `railway.toml` (อ่านอัตโนมัติ) | 08:30 จ-ศ | `30 1 * * 1-5` | `railway_run.sh next_day` |
| t1600 | `railway.t1600.toml` (ตั้ง path เอง) | 16:02 จ-ศ | `2 9 * * 1-5` | `railway_run.sh t1600` |

## ตัวแปร (Railway → service → Variables) — ตั้งทั้งสอง service
| ตัวแปร | ค่า | หมายเหตุ |
|---|---|---|
| `DRY_RUN` | `1` (เริ่มต้น) → `0` เมื่อพร้อมบันทึกจริง | `1` = `--dry-run` ไม่ commit/push |
| `GITHUB_TOKEN` | token ที่เขียน repo ได้ | fine-grained: เฉพาะ repo นี้ สิทธิ์ Contents = Read and write |
| `SMOKE_ONLY` | `1` ตอนทดสอบรอบแรก แล้วลบทิ้ง | รันแค่ smoke test |

## ขั้นตอน (ทำตามลำดับ)
1. commit + push ไฟล์เหล่านี้ (เสร็จแล้วค่อยทำข้อถัดไป)
2. service เดิม (`CN340_Stock_Prediction`) → Settings: ตรวจว่า Builder = Dockerfile (อ่านจาก `railway.toml`)
3. ตั้ง `SMOKE_ONLY=1` (+ `GITHUB_TOKEN`) แล้วสั่งรันทันที: `railway cron run` (CLI) หรือแก้ cron ชั่วคราวเป็นอีก 5-10 นาทีข้างหน้า (UTC)
   - ผ่านเมื่อ log ขึ้น `smoke test ผ่าน` และ **ข้อมูล yfinance ดึงได้จาก cloud** (ถ้า Yahoo บล็อก IP จะเห็น `FAIL ... Yahoo ไม่คืนข้อมูล` -- ใช้ Railway ไม่ได้)
   - ถ้าตั้ง token: ต้องเห็น `สิทธิ์ push ผ่าน`
4. ลบ `SMOKE_ONLY` แล้วปล่อย `DRY_RUN=1` อย่างน้อย 1 วันทำการ ดู log ว่าเริ่มตรงเวลา ทำนายได้ครบ
5. เปลี่ยนเป็น `DRY_RUN=0` ใน service ทีละตัว
6. สร้าง service ที่สอง (t1600) จาก repo เดียวกัน → Settings → Config-as-code → path `/railway.t1600.toml` → ตั้งตัวแปรเหมือนกัน
7. **ปิดงานบนเครื่องตัวเองก่อนเปิด `DRY_RUN=0`** (`powershell -File automation\register_next_day.ps1 -Unregister` และ `register_1600.ps1 -Unregister`)
   ถ้าเปิดซ้ำซ้อน สองที่จะ push ชนกันและ guard กันทำนายซ้ำจะปฏิเสธรอบหลัง

## สิ่งที่สคริปต์ทำให้เอง (เมื่อ `DRY_RUN=0`)
- ข้ามวันหยุด SET / เสาร์-อาทิตย์ (อ่าน `task_b/set_holidays.txt`) → exit 0
- t1600 ถ้าเริ่มนอก 16:00–16:29 เวลาไทย → ยกเลิกก่อนทำอะไร (exit 3) กันได้ log ครึ่ง ๆ (task_a2 เขียนไปแล้วแต่ task_b ปฏิเสธ)
- next_day ถ้าเริ่มช่วง 10:00–17:59 → ยกเลิก (exit 3)

## ข้อจำกัดที่รู้แล้ว
- Railway cron "ไม่การันตีนาทีที่เริ่ม อาจคลาดได้ไม่กี่นาที" และ **ถ้ารอบก่อนยังไม่จบ รอบถัดไปถูกข้าม**
- cron ใช้เวลา UTC เสมอ · ช่วงห่างขั้นต่ำ 5 นาที
- เครดิตทดลอง: "17 days or $5.00" หมดอย่างใดอย่างหนึ่งก่อนก็จบ (บริการจะหยุด) -- cron ที่รันไม่กี่นาทีต่อวันใช้เครดิตน้อย
- Yahoo บางเช้าส่ง Close ของวันล่าสุดเป็น NaN → ขั้น fetch ของ task_b หยุดตามที่ออกแบบ (`task_b` ไม่เขียนอะไร) รอบนั้นล้มทั้งตัวคุม
  ต้องรันซ้ำภายหลัง (ตั้ง `DRY_RUN=0` แล้วสั่ง `railway cron run`) -- ปัญหานี้มีอยู่เดิมบนเครื่องตัวเองด้วย ไม่ได้เกิดจาก Railway
- ยังไม่เคย build image นี้จริง (เครื่องที่เขียนไม่มี Docker) ทดสอบแล้วเฉพาะสคริปต์ `railway_run.sh`
