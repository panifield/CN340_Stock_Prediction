# deploy/ — รันงานรายวันบน Railway (cron)

ไฟล์ที่เกี่ยวข้อง: `Dockerfile` · `railway.toml` (งานเช้า) · `railway.t1600.toml` (งาน 16:02) · `deploy/railway_run.sh` ·
`deploy/precheck.py` (+ `deploy/test_precheck.py`) · `.dockerignore` · `.gitattributes`

## แนวคิด
Railway เป็น cron: รันคำสั่ง → ทำงานเสร็จ → จบ (ห้ามค้าง ไม่งั้นรอบถัดไปถูกข้าม) และ filesystem ไม่ถาวร
จึงให้ `railway_run.sh` **git clone repo ล่าสุดทุกครั้ง** แล้วรัน `automation/run_next_day.py` หรือ `run_1600.py`
log ที่เขียนจะถูก `git push` กลับ GitHub (ถ้าไม่ push log หายเมื่อ container ดับ)

| service | config | cron (UTC) | เวลาไทย | คำสั่ง |
|---|---|---|---|---|
| next_day | `railway.toml` | `*/20 1-2 * * 1-5` | 08:00, 08:20 ... 09:40 (ทุก 20 นาที) | `railway_run.sh next_day` |
| t1600 | `railway.t1600.toml` (ตั้ง path เอง) | `2,12,22 9 * * 1-5` | 16:02, 16:12, 16:22 | `railway_run.sh t1600` |

**ทำไมถี่หลายรอบ:** ก่อนรันจริง `deploy/precheck.py` ตรวจก่อน (ไม่เขียนอะไร)
- **next_day:** Yahoo ต้องมี Close ของวัน cutoff ครบทั้ง 2 หุ้น ถ้ายัง NaN → จบเงียบ (exit 0) รอรอบถัดไป · ถ้าถึงรอบ 09:30 ขึ้นไปแล้วยังไม่พร้อม → ล้มเหลว (exit 4)
  (ไม่ปล่อยให้ task_a/task_c ทำนายด้วยข้อมูลเก่ากว่า 1 วันแล้ว task_b หยุด → log ครึ่งเดียวและรันซ้ำไม่ได้)
- **ทั้งสองงาน:** ถ้า log จริงของ target มีครบทุก task แล้ว → ข้าม (exit 0) จึงไม่รันซ้ำหลังสำเร็จ
- **ทำไปแล้วบางส่วน** (มีบาง task แต่ไม่ครบ) → หยุด exit 5 ให้คนดู ไม่รันซ้ำเอง

## ตัวแปร (Railway → service → Variables) — ตั้งทั้งสอง service
| ตัวแปร | ค่า | หมายเหตุ |
|---|---|---|
| `DRY_RUN` | ไม่ตั้ง (= `1`) → `0` เมื่อพร้อมบันทึกจริง | `1` = `--dry-run` ไม่ commit/push (precheck แสดงผลแต่ไม่หยุดการรัน) |
| `GITHUB_TOKEN` | token ที่เขียน repo ได้ | service ที่สองอ้างอิงจาก service แรกได้: `${{CN340_Stock_Prediction.GITHUB_TOKEN}}` |
| `SMOKE_ONLY` | `1` เฉพาะตอนทดสอบสภาพแวดล้อม | ลบทิ้งเมื่อทดสอบเสร็จ (แล้วต้อง Deploy) |

## สร้าง service ที่สอง (t1600)
1. หน้า project → **+ Add** (หรือ **Create**) → **GitHub Repo** → เลือก `panifield/CN340_Stock_Prediction` (repo เดียวกับ service แรก)
2. ตั้งชื่อ service เช่น `CN340_t1600`
3. **Settings** → ตั้งค่า 3 ช่องนี้ (ถ้าหน้า Settings ขึ้นว่าอ่านจาก config file ให้ใส่ path ไฟล์แทน):
   - Config-as-code / Config File path: `/railway.t1600.toml` **หรือ** กรอกเองตรง ๆ:
     - Custom Start Command: `bash /usr/local/bin/railway_run.sh t1600`
     - Cron Schedule: `2,12,22 9 * * 1-5`
   - Builder: Dockerfile (path `/Dockerfile`) · Restart policy: Never
4. **Variables** → `GITHUB_TOKEN` (ใช้ reference ข้างบนก็ได้) · ไม่ต้องตั้ง `DRY_RUN` (ค่าเริ่มต้น = dry-run)
5. Deploy แล้วดู log ว่าขึ้น `mode=t1600 DRY_RUN=1` — ทดสอบเต็มได้เฉพาะ **16:00-16:30 ของวันทำการ** (ตอนอื่นไม่มีแท่ง 15:00 ของวันนี้)
   รอบ dry-run แรกจึงควรปล่อยให้ cron 16:02 ทำงานเอง แล้วเปิดดู Cron Runs / logs

## ลำดับเปลี่ยนเป็นบันทึกจริง (ทำทีละ service)
1. ดู dry-run ผ่านอย่างน้อย 1 วันทำการ (log ขึ้น task ครบ ไม่มี commit/push)
2. **ปิดงานเดียวกันบนเครื่องตัวเองก่อน** (`powershell -File automation\register_next_day.ps1 -Unregister` สำหรับงานเช้า,
   `register_1600.ps1 -Unregister` สำหรับ t1600) -- ถ้าเปิดซ้ำซ้อนสองที่ push ชนกัน แต่ **อย่าปิดก่อนจะถึงขั้นนี้**
   (ระหว่าง Railway ยังเป็น dry-run ให้เครื่องคุณทำงานจริงต่อไป เพื่อไม่ให้วันนั้นไม่มี log)
3. ตั้ง `DRY_RUN=0` → Deploy (การ deploy ทำให้ service รันทันที สคริปต์กันไว้: วันหยุด/นอกเวลา/ทำเสร็จแล้ว จะข้ามเอง)
4. รอบแรกที่บันทึกจริง ดู log ว่ามี `committed` + `pushed` และ commit ขึ้นใน GitHub

## exit code ของ `railway_run.sh`
0 สำเร็จ/ข้ามอย่างตั้งใจ (วันหยุด, ทำเสร็จแล้ว, รอข้อมูล) · 1 clone หรือ task ล้มเหลว · 2 ตั้งค่าผิด · 3 เริ่มผิดเวลา ·
4 ข้อมูล Yahoo ไม่พร้อมจนรอบสุดท้าย · 5 ทำไปแล้วบางส่วน (ต้องมีคนดู) · 6 precheck ผิดพลาด

## ข้อจำกัดที่รู้แล้ว
- Railway cron "ไม่การันตีนาทีที่เริ่ม อาจคลาดได้ไม่กี่นาที" และ **ถ้ารอบก่อนยังไม่จบ รอบถัดไปถูกข้าม** · ใช้เวลา UTC เสมอ
- เครดิตทดลอง "17 days or $5.00" หมดอย่างใดอย่างหนึ่งก่อนก็จบ -- รอบที่ "ยังไม่พร้อม/ทำเสร็จแล้ว" ใช้เวลา ~20 วินาที/รอบ ไม่ใช่ปัญหา
- **Yahoo NaN:** ตัวคุม `automation/run_next_day.py` มี preflight ของตัวเอง (`--commit` เท่านั้น) ตรวจว่า Yahoo มี Close ของวัน cutoff ครบทั้ง 2 หุ้น
  ถ้าไม่ครบจะหยุด **ก่อนเขียนอะไร** (exit 1, ข้ามได้ด้วย `--no-preflight`) -- จึงครอบคลุมทั้งบน Railway และงานบนเครื่องตัวเอง/GitHub Actions
  (`deploy/precheck.py` ใช้ฟังก์ชันเดียวกัน เพิ่มเรื่อง "ทำเสร็จแล้วไหม" และการ retry ของ cron)
  - ไม่ได้แก้ task_b ให้ "ตัดแถว NaN แล้วเดินต่อ" โดยตั้งใจ: task_b คำนวณ target จาก data_cutoff ของตัวเอง จะทำนายวันที่ผ่านไปแล้วลง log จริง ขัดกับ task_a/c
  - ถ้า Yahoo ไม่เติม Close ทั้งวัน รอบ 09:40 ล้มเหลว (exit 4) วันนั้นไม่มี log ของ next_day -- เป็นเจตนา (ดีกว่าได้ log จากข้อมูลเก่า)
  - งานบนเครื่องตัวเอง (Task Scheduler) รันครั้งเดียว 08:30 ไม่มี retry: ถ้า preflight หยุด ต้องรันซ้ำเองภายหลังด้วย
    `python automation/run_next_day.py --commit --push` (ก่อน 10:00 น.)
- partial (exit 5): ถ้า task แรกของงาน t1600 เขียน log แล้วแต่ task ถัดไปล้ม รอบถัดไปจะหยุดที่ precheck เพื่อกัน log ซ้ำ
  ต้องรัน task ที่เหลือเอง (ตัวคุมเดิมไม่รองรับการรันต่อจากกลางทาง)
- ทดสอบ image บนเครื่องที่มี Docker แล้ว (smoke + dry-run next_day ผ่านถึง task_b) · cron จริงบน Railway ยังไม่เคยเห็นผลตามเวลา
