# deploy/ — รันงานรายวันบน Railway (cron, service เดียว)

ไฟล์ที่เกี่ยวข้อง: `Dockerfile` · `railway.toml` · `deploy/railway_run.sh` · `deploy/precheck.py` (+ `deploy/test_precheck.py`) ·
`.dockerignore` · `.gitattributes`

## แนวคิด
Railway เป็น cron: รันคำสั่ง → ทำงานเสร็จ → จบ (ห้ามค้าง ไม่งั้นรอบถัดไปถูกข้าม) และ filesystem ไม่ถาวร
จึงให้ `railway_run.sh` **git clone repo ล่าสุดทุกครั้ง** แล้วรัน `automation/run_next_day.py` หรือ `run_1600.py`
log ที่เขียนจะถูก `git push` กลับ GitHub (ถ้าไม่ push log หายเมื่อ container ดับ)

**service เดียว โหมด `auto`:** cron ยิงทุก 10 นาทีใน UTC 01:xx, 02:xx, 09:xx (= 08:02-09:52 และ 16:02-16:52 น. ไทย จันทร์-ศุกร์ รวม 18 รอบ/วัน)
แล้วสคริปต์เลือกงานจากเวลาไทย:

| เวลาไทยที่เริ่ม | งาน |
|---|---|
| 08:00 – 09:59 | `next_day` (task_a → task_c → task_b) |
| 16:00 – 16:29 | `t1600` (task_a2 → task_b → task_c) |
| เวลาอื่น (เช่น 16:32, 16:42, 16:52 และตอน deploy/redeploy) | จบใน 2-3 วินาที ไม่ clone ไม่ทำอะไร |

(เหตุผลที่ service เดียวพอ: Railway ให้ cron ได้ 1 ตารางต่อ service แต่ตารางเดียวยิงถี่ครอบทั้งสองช่วงได้ แล้วให้สคริปต์เลือกงาน)

**ก่อนรันจริงทุกรอบ** `deploy/precheck.py` ตรวจ (ไม่เขียนอะไร):
- **next_day:** Yahoo ต้องมี Close ของวัน cutoff ครบทั้ง 2 หุ้น ถ้ายัง NaN → จบเงียบ (exit 0) รอรอบถัดไป · ถ้าถึงรอบ 09:50 ขึ้นไปแล้วยังไม่พร้อม → ล้มเหลว (exit 4)
- **ทั้งสองงาน:** ถ้า log จริงของ target มีครบทุก task แล้ว → ข้าม (exit 0) · ถ้ามีบางส่วน → หยุด exit 5 ให้คนดู

## ตั้งค่า service (ทำที่ service `CN340_Stock_Prediction` ที่มีอยู่)
ถ้า Settings ขึ้นว่าอ่านจาก `railway.toml` ได้เลย ถ้าไม่ ให้กรอกเอง (ค่าในหน้าเว็บ/ไฟล์ต้องตรงกัน):
- **Custom Start Command:** `bash /usr/local/bin/railway_run.sh auto`
- **Cron Schedule (UTC):** `2,12,22,32,42,52 1,2,9 * * 1-5`
- Builder: Dockerfile (`/Dockerfile`) · Restart policy: Never

## ตัวแปร (Variables)
| ตัวแปร | ค่า | หมายเหตุ |
|---|---|---|
| `GITHUB_TOKEN` | token ที่เขียน repo ได้ | ต้องมีเมื่อ `DRY_RUN=0` |
| `DRY_RUN` | ไม่ตั้ง (= `1` ทดสอบ) → `0` เมื่อพร้อมบันทึกจริง | `1` = `--dry-run` ไม่ commit/push (precheck แสดงผลแต่ไม่หยุดการรัน) |
| `AUTO_FORCE_MODE` | `next_day` หรือ `t1600` (ใช้ตอนทดสอบเท่านั้น) | บังคับงานโดยไม่ดูเวลา เช่นกด **Run now** ตอนกลางคืน · **ลบทิ้งเมื่อทดสอบเสร็จ** |
| `SMOKE_ONLY` | `1` ตอนทดสอบสภาพแวดล้อมเท่านั้น | ลบทิ้ง |

`AUTO_FORCE_MODE` ข้ามเฉพาะการเลือกงานตามเวลา เมื่อ `DRY_RUN=0` ตัวเช็ควัน/เวลาด้านใน (t1600 ต้อง 16:00-16:29, next_day ห้าม 10:00-17:59) ยังทำงาน

## ลำดับใช้งาน
1. push → service build ใหม่ → Deploy (การ deploy ทำให้ service รัน 1 รอบ: ถ้าอยู่นอกช่วงเวลางานจะจบเองทันที)
2. ทดสอบ pipeline ตอนนี้: ตั้ง `AUTO_FORCE_MODE=next_day` แล้ว Deploy / **Run now** → ดู log (ต้องเห็น `auto -> mode=next_day`, `[precheck]`, `[task_a]`... และไม่มี `committed`/`pushed`) แล้วลบตัวแปรนี้
3. ปล่อย `DRY_RUN` ว่าง (= dry-run) ให้ cron ทำงานเอง 1 วันทำการ ดู **Cron Runs**: เช้าเห็นรอบ 08:02-... และ 16:02 เห็น t1600 ครบ 3 task
   (**ระหว่างนี้อย่าปิด Task Scheduler บนเครื่องตัวเอง** ให้เครื่องบันทึกจริงต่อ เพื่อให้วันนั้นมี log)
4. เมื่อ dry-run ผ่านทั้งสองงาน: ปิดงานบนเครื่อง (`register_next_day.ps1 -Unregister`, `register_1600.ps1 -Unregister`) → ตั้ง `DRY_RUN=0` → Deploy
   (ทำนอกช่วง 08:00-09:59 และ 16:00-16:29 เพื่อไม่ให้รอบ deploy ชนกับงานของเครื่องคุณ)
5. รอบแรกที่บันทึกจริง ดู log ว่ามี `committed` + `pushed` และมี commit ใหม่ใน GitHub

## exit code ของ `railway_run.sh`
0 สำเร็จ/ข้ามอย่างตั้งใจ (นอกช่วงงาน, วันหยุด, ทำเสร็จแล้ว, รอข้อมูล) · 1 clone หรือ task ล้มเหลว · 2 ตั้งค่าผิด · 3 เริ่มผิดเวลา ·
4 ข้อมูล Yahoo ไม่พร้อมจนรอบสุดท้าย · 5 ทำไปแล้วบางส่วน (ต้องมีคนดู) · 6 precheck ผิดพลาด

## ข้อจำกัดที่รู้แล้ว
- Railway cron "ไม่การันตีนาทีที่เริ่ม อาจคลาดได้ไม่กี่นาที" และ **ถ้ารอบก่อนยังไม่จบ รอบถัดไปถูกข้าม** · ใช้เวลา UTC เสมอ
- เครดิตทดลอง "17 days or $5.00" หมดอย่างใดอย่างหนึ่งก่อนก็จบ -- รอบที่ "นอกช่วง/ยังไม่พร้อม/ทำเสร็จแล้ว" ใช้ไม่กี่วินาทีถึง ~20 วินาที ไม่ใช่ปัญหาด้านเครดิต
  ตัวจำกัดจริงคือจำนวนวันของช่วงทดลอง
- **Yahoo NaN:** ตัวคุม `automation/run_next_day.py` มี preflight ของตัวเอง (`--commit` เท่านั้น) ตรวจว่า Yahoo มี Close ของวัน cutoff ครบทั้ง 2 หุ้น
  ถ้าไม่ครบจะหยุด **ก่อนเขียนอะไร** (exit 1, ข้ามได้ด้วย `--no-preflight`) -- จึงครอบคลุมทั้งบน Railway และงานบนเครื่องตัวเอง/GitHub Actions
  - ไม่ได้แก้ task_b ให้ "ตัดแถว NaN แล้วเดินต่อ" โดยตั้งใจ: task_b คำนวณ target จาก data_cutoff ของตัวเอง จะทำนายวันที่ผ่านไปแล้วลง log จริง ขัดกับ task_a/c
  - ถ้า Yahoo ไม่เติม Close ทั้งวัน รอบ 09:50 ล้มเหลว (exit 4) วันนั้นไม่มี log ของ next_day -- เป็นเจตนา (ดีกว่าได้ log จากข้อมูลเก่า)
  - งานบนเครื่องตัวเอง (Task Scheduler) รันครั้งเดียว 08:30 ไม่มี retry: ถ้า preflight หยุด ต้องรันซ้ำเองภายหลังด้วย
    `python automation/run_next_day.py --commit --push` (ก่อน 10:00 น.)
- partial (exit 5): ถ้า task แรกของงาน t1600 เขียน log แล้วแต่ task ถัดไปล้ม รอบถัดไปจะหยุดที่ precheck เพื่อกัน log ซ้ำ
  ต้องรัน task ที่เหลือเอง (ตัวคุมเดิมไม่รองรับการรันต่อจากกลางทาง)
- ทดสอบ image บนเครื่องที่มี Docker แล้ว (smoke + dry-run next_day ผ่านถึง task_b) · cron จริงบน Railway ยังไม่เคยเห็นผลตามเวลา
