# INTEGRATION — การรวม task_b และเชื่อมข้อมูลจาก SETTRADE

เอกสารนี้สำหรับคนที่ merge task_b เข้ากับ task อื่นและเชื่อม SETTRADE API
อ่านข้อ 1–4 ก่อนเขียนโค้ดเชื่อม · คำสั่งทั้งหมดรันจาก root ของ repo

---

## 1. task_b พึ่งอะไรบ้าง

- task_b **ไม่ import โค้ดจาก task อื่น** และทุก path อิงกับโฟลเดอร์ `task_b/` เอง → merge แล้วไม่ชนกัน
- **task_b ไม่ต้องรู้จัก API key** — ตัวเชื่อมกลางดึงข้อมูล แล้วส่งให้ task_b เป็น **ไฟล์**
  (โมเดลอ่านจากไฟล์เสมอ เพื่อให้ SHA-256 ใน prediction log ชี้ไปที่ข้อมูลที่ตรวจย้อนหลังได้)
- แหล่งข้อมูลที่ตกลงกันแล้ว:

| โมเดล | แหล่งข้อมูล | ไฟล์ที่ task_b อ่าน |
|---|---|---|
| แบบที่ 1: ทำนายราคาปิดวัน t (รันหลังตลาดปิดวัน t−1 ถึงก่อน 09:00 วัน t) | Investing.com รายวัน | `task_b/raw_data/{KBANK,ADVANC}_10Y_Cleaned.csv` |
| แบบที่ 2: ณ 16:00 วัน t ทำนาย close_bar16 วัน t | Yahoo รายชั่วโมง | `task_b/raw_data_intraday/` (ล็อก SHA) + snapshot live ใน `task_b/raw_data_intraday_live/` |

**ห้ามผสมแหล่ง:** ข้อมูลจากแหล่งอื่น (รวม SETTRADE) ห้ามนำไปต่อ `raw_data/` หรือป้อนโมเดล 16:00
จนกว่าจะตรวจแล้วว่าตรงกับแหล่งเดิม — Yahoo กับ Investing ต่างกันเฉลี่ย 0.33–0.69 บาทอยู่แล้ว
การเปลี่ยนแหล่งข้อมูลของโมเดลใดต้องเทรนใหม่และเขียนแผนใหม่ก่อน

## 2. ข้อมูลรายวัน (แบบที่ 1) — ส่งผ่าน `tools/append_investing.py` เท่านั้น

ห้ามเขียนทับ `task_b/raw_data/*.csv` ตรง ๆ · แถวเก่าต้องไม่เปลี่ยนแม้แต่ byte เดียว
(ไม่งั้นผล validation และ SHA เปลี่ยน)

ให้ตัวเชื่อมเขียนไฟล์ในรูปแบบเดียวกับที่ดาวน์โหลดจาก Investing.com แล้วส่งเข้าเครื่องมือ:

```bash
python task_b/tools/append_investing.py <ไฟล์ KBANK> <ไฟล์ ADVANC> --check-only   # ตรวจก่อน
python task_b/tools/append_investing.py <ไฟล์ KBANK> <ไฟล์ ADVANC>                # เขียนจริง
```

รูปแบบไฟล์ input ที่เครื่องมือรับ:

```
"Date","Price","Open","High","Low","Vol.","Change %"
"09/25/2026","252.00","253.00","254.00","251.00","8.65M","0.40%"
```

| ข้อกำหนด | |
|---|---|
| ชื่อไฟล์ | ต้องมี `kasikorn` หรือ `kbank` / `advanced info` หรือ `advanc` (ไม่สนตัวพิมพ์) — ใช้จับคู่หุ้น |
| คอลัมน์ | ตรงตามตัวอย่างทุกตัว (header ต่างแม้ตัวเดียว = หยุด) |
| Date | MM/DD/YYYY · จะเรียงใหม่→เก่า หรือเก่า→ใหม่ก็ได้ |
| Price | **ราคาปิด** |
| Vol. | จำนวนหุ้น มีหน่วยต่อท้าย `K` / `M` / `B` (เช่น `8.65M` = 8.65 ล้านหุ้น) · ห้ามว่างหรือ `-` |
| Change % | % เปลี่ยนจากราคาปิดวันก่อน ทศนิยม 2 ตำแหน่ง |
| หุ้น | ต้องส่ง**ทั้งสองหุ้นพร้อมกัน** และสิ้นสุดวันเดียวกัน |
| encoding | UTF-8 (มี BOM ก็ได้) · มี quote ก็ได้ · comma คั่นหลักพันได้ |

เครื่องมือจะแปลงเป็น canonical (`Date,Price,Open,High,Low,Vol. ('000),Change %` · CRLF · ไม่มี BOM),
ตรวจแถวที่ทับของเดิม, เก็บไฟล์ต้นฉบับใน `task_b/raw_data_sources/investing_<YYYYMMDD>/` พร้อม SHA
แล้วรัน `check_raw_update` ให้อัตโนมัติ · รายละเอียดใน `task_b/tools/README.md`

ข้อมูลปัจจุบันใน `raw_data/` สิ้นสุด **2026-09-25**

## 3. ข้อมูลรายชั่วโมง (แบบที่ 2) — ส่งผ่าน `tools/save_intraday_snapshot.py`

รูปแบบ CSV ที่ตัวเชื่อมต้องเขียน:

- คอลัมน์: `Datetime,Adj Close,Close,High,Low,Open,Volume`
- `Datetime` ต้องมี timezone `+07:00` และเป็น **เวลาเริ่มของแท่ง** (แท่ง 15:00 = 15:00→16:00)
- ต้องมีแท่ง 10:00, 11:00, 12:00, 14:00, 15:00 ของวันที่ทำนาย (แท่ง 13:00 ไม่ถูกใช้)
- ควรมีประวัติย้อนหลังหลายวัน (อย่างน้อยตั้งแต่ 2026-09-25) เพื่อให้วันหลังไฟล์ล็อกครบ

เก็บเป็น snapshot (ห้ามวางไฟล์เอง — สคริปต์เขียน SHA + เวลาดาวน์โหลดให้):

```bash
python task_b/tools/save_intraday_snapshot.py KBANK.BK  <ไฟล์.csv> --source yahoo
python task_b/tools/save_intraday_snapshot.py ADVANC.BK <ไฟล์.csv> --source yahoo
```

- เก็บที่ `task_b/raw_data_intraday_live/<YYYYMMDD>/` · **ห้ามแก้ `task_b/raw_data_intraday/`** (ล็อก SHA)
- `--source` ต้องบอกแหล่งจริง · `predict_1600.py` รับเฉพาะ `yahoo` (แหล่งที่ใช้เทรน)
  → ถ้าจะใช้ข้อมูลรายชั่วโมงจาก SETTRADE ต้องเทียบกับ Yahoo และทำแผน Phase 1D รอบสองก่อน
- snapshot ที่ใช้ทำนายต้องดาวน์โหลด **ณ/หลัง 16:00** · snapshot ที่ใช้เป็นผลจริงต้องดาวน์โหลด **หลัง 17:00**

สถานะ: โค้ด official พร้อมแล้ว แต่ **ยังเปิดไม่ได้** จนกว่าจะพิสูจน์ real-time availability
(live dry-run หลายวัน + `tools/check_snapshot_1600.py`) แล้วคนเขียน `task_b/PHASE1D_LIVE_APPROVAL.md` + commit
— ดูขั้นตอนใน `task_b/README.md` หัวข้อ Phase 1D

## 4. API key

- เก็บใน `.env` ที่ root ของ repo · **ต้องเพิ่ม `.env` ใน `.gitignore` ที่ root ก่อน commit ใด ๆ**
  (ตอนนี้ `.gitignore` ที่ root ยังไม่มีบรรทัดนี้)
- ห้าม hardcode key ในโค้ด · ห้ามพิมพ์ key ลง log
- SDK ของ SETTRADE ไม่ต้องเพิ่มใน `task_b/requirements.txt`

## 5. ห้ามเปลี่ยน

| สิ่งที่ห้ามเปลี่ยน | เหตุผล |
|---|---|
| เวอร์ชันใน `task_b/requirements.txt` (numpy 2.1.1 · pandas 2.3.3 · scikit-learn 1.9.1 · xgboost 3.4.1) | ผลของโมเดลขึ้นกับเวอร์ชัน · ถ้าจำเป็นต้องเปลี่ยน ให้รัน regression check (ข้อ 6) แล้วบันทึกผล |
| `task_b/.gitattributes` | กันไม่ให้ git แปลง line ending ของไฟล์ข้อมูล (ไม่งั้น SHA ไม่ตรงหลัง clone) |
| `task_b/raw_data/` (แถวเก่า), `raw_data_intraday/`, `raw_data_sources/` | หลักฐานที่มาของข้อมูล |
| `task_b/results/prediction_log.csv` (แถวเก่า) | append-only · คำทำนายที่บันทึกแล้วห้ามแก้ |
| `task_b/config.py` ส่วน `SPLIT_BY_DATE`, `*_PARAMS`, `CONFIG_TAG`, `TUNING_PLAN.md`, `PHASE1D_PLAN.md` | ค่าที่ล็อกไว้หลังจูน |
| ห้ามรัน `python task_b/main.py` แบบไม่มี `--dev` | จะเปิด historical test (ถูกล็อกไว้) |

## 6. หลัง merge ให้รันเช็ก

```bash
python task_b/tests/test_pipeline.py            # 18/18
python task_b/tests/test_live_eval.py           # 5/5
python task_b/tests/test_append_investing.py    # 10/10
python task_b/tests/test_live_1600.py           # 12/12
python task_b/tests/test_phase1d.py             # 21/21 (~3 นาที)
python task_b/main.py --dev
git diff --stat task_b/results/                 # *_val.csv ต้องไม่เปลี่ยน
```

ถ้าเจอ `UnicodeEncodeError` ให้ตั้ง `PYTHONUTF8=1`

## 7. ขั้นตอนประจำวันของแบบที่ 1 (หลังเชื่อมข้อมูลแล้ว)

```bash
# หลังตลาดปิดวัน t−1
python task_b/tools/append_investing.py <KBANK> <ADVANC>
python task_b/main.py --dev                                  # val ต้องไม่เปลี่ยน
git add task_b/raw_data task_b/raw_data_sources && git commit -m "data: <t−1>"   # ① ข้อมูลก่อน
python task_b/record_outcomes.py && python task_b/report_live.py
python task_b/predict_live.py --target-date <t> --expected-cutoff <t−1>
git add task_b/results && git commit -m "prediction log: <t>" && git push         # ② ก่อน 09:00
```

- `predict_live.py` จะไม่ยอมรันถ้า `*.py` หรือ `raw_data/` ยังไม่ commit หรือ log รอบก่อนยังไม่ commit
- สคริปต์ไม่ตรวจวันหยุด SET — `--target-date` ต้องเป็นวันทำการจริง
- ผลจริงมาจาก `raw_data/` แหล่งเดียวกับที่เทรน · รายงานอยู่ที่ `task_b/results/live_report/`
