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
| แบบที่ 1: ทำนายราคาปิดวัน t (รันหลังตลาดปิดวัน t−1 ถึงก่อน 09:00 วัน t) | Investing.com รายวัน (ดาวน์โหลดเอง) **หรือ** Yahoo รายวันผ่านตัวเชื่อม `tools/fetch_yahoo_daily.py` | `task_b/raw_data/{KBANK,ADVANC}_10Y_Cleaned.csv` |
| แบบที่ 2: ณ 16:00 วัน t ทำนาย close_bar16 วัน t | Yahoo รายชั่วโมง | `task_b/raw_data_intraday/` (ล็อก SHA) + snapshot live ใน `task_b/raw_data_intraday_live/` |

**ห้ามผสมแหล่ง:** ข้อมูลจากแหล่งอื่น (รวม SETTRADE) ห้ามนำไปต่อ `raw_data/` หรือป้อนโมเดล 16:00
จนกว่าจะตรวจแล้วว่าตรงกับแหล่งเดิม — Yahoo กับ Investing ต่างกันเฉลี่ย 0.33–0.69 บาทอยู่แล้ว
การเปลี่ยนแหล่งข้อมูลของโมเดลใดต้องเทรนใหม่และเขียนแผนใหม่ก่อน

**ข้อยกเว้นที่ตรวจแล้ว — Yahoo รายวัน (แบบที่ 1):** ตรวจเมื่อ 2026-09-28 ว่า Open/High/Low/Close รายวันของ Yahoo
ตรงกับ `raw_data/` **เป๊ะ 100%** ทุกวันที่ทับกัน (2,453 วัน × 2 หุ้น — ตัวเลข 0.33–0.69 บาทข้างบนคือแท่ง 16:00
รายชั่วโมง ไม่ใช่รายวัน) · Volume ต่างได้ (Investing ปัด 3 หลัก / Yahoo แก้ย้อนหลัง) → ตัวเชื่อมตรวจ OHLC
ช่วงทับ ~30 วันทำการ **ทุกครั้งที่รัน** ถ้าไม่ตรงแม้วันเดียวจะหยุด · แหล่งของทุกแถวบันทึกใน
`raw_data_sources/<investing|yahoo>_<YYYYMMDD>/`

## 2. ข้อมูลรายวัน (แบบที่ 1) — ส่งผ่าน `tools/append_investing.py` เท่านั้น

**ตัวเชื่อมที่มีแล้ว:** `tools/fetch_yahoo_daily.py` (ดึง Yahoo → ตรวจ → เขียนไฟล์รูปแบบ Investing ลง `.staging/`)
แล้วส่งเข้า `append_investing.py --source yahoo` — หรือรันทั้งหมดด้วย `python task_b/daily_next_day.py` (ข้อ 7)
ไฟล์ Investing ที่ดาวน์โหลดเองยังใช้ได้เหมือนเดิม (`--source investing` เป็นค่า default)

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

สถานะ: official **เปิดใช้ได้** (ยกเลิกขั้นอนุมัติล่วงหน้า 2026-09-28) · ตัวเชื่อม `tools/fetch_yahoo_intraday.py`
ทำขั้นนี้ให้อัตโนมัติ — ดูข้อ 8 และ `task_b/README.md` หัวข้อ Phase 1D

## 4. API key

- เก็บใน `.env` ที่ root ของ repo · `.gitignore` ที่ root มี `.env` และ `.env.*` แล้ว (ตรวจ 2026-09-28)
- ตัวเชื่อม Yahoo ของ task_b ไม่ใช้ API key
- ห้าม hardcode key ในโค้ด · ห้ามพิมพ์ key ลง log
- SDK ของ SETTRADE ไม่ต้องเพิ่มใน `task_b/requirements.txt`

## 5. ห้ามเปลี่ยน

| สิ่งที่ห้ามเปลี่ยน | เหตุผล |
|---|---|
| เวอร์ชันใน `task_b/requirements.txt` (numpy 2.1.1 · pandas 2.3.3 · scikit-learn 1.9.1 · xgboost 3.4.1) | ผลของโมเดลขึ้นกับเวอร์ชัน · ถ้าจำเป็นต้องเปลี่ยน ให้รัน regression check (ข้อ 6) แล้วบันทึกผล · ตรวจด้วย `python task_b/tools/check_env.py` · `predict_live.py` / `predict_1600.py` แบบ official **หยุดเอง**ถ้าเวอร์ชันไม่ตรง · ถ้า task อื่นต้องการเวอร์ชันต่างกัน ให้ใช้ venv แยกของ task_b (`pip install -r task_b/requirements-live.txt`) |
| `task_b/.gitattributes` | กันไม่ให้ git แปลง line ending ของไฟล์ข้อมูล (ไม่งั้น SHA ไม่ตรงหลัง clone) |
| `task_b/raw_data/` (แถวเก่า), `raw_data_intraday/`, `raw_data_sources/` | หลักฐานที่มาของข้อมูล |
| `task_b/results/prediction_log.csv` (แถวเก่า) | append-only · คำทำนายที่บันทึกแล้วห้ามแก้ |
| `task_b/config.py` ส่วน `SPLIT_BY_DATE`, `*_PARAMS`, `CONFIG_TAG`, `TUNING_PLAN.md`, `PHASE1D_PLAN.md` | ค่าที่ล็อกไว้หลังจูน |
| `main.py` ต้องระบุโหมดเสมอ: `--dev` (พัฒนา) หรือ `--open-test` | `python main.py` เปล่า ๆ = error · `--open-test` เปิด historical test ซึ่งต้องมี `PRE_TEST_LOCK.md` ครบ (ห้ามสร้างจนกว่าตัดสินใจเปิด test) |

## 6. หลัง merge ให้รันเช็ก

```bash
python task_b/tests/test_pipeline.py            # 18/18
python task_b/tests/test_live_eval.py           # 5/5
python task_b/tests/test_append_investing.py    # 10/10
python task_b/tests/test_live_1600.py           # 12/12
python task_b/tests/test_phase1d.py             # 21/21 (~3 นาที)
python task_b/tests/test_fetch_yahoo_daily.py   # 9/9
python task_b/tests/test_daily.py               # 8/8
python task_b/tools/check_env.py                # เวอร์ชันตรง pin
python task_b/main.py --dev
git diff --stat task_b/results/                 # *_val.csv ต้องไม่เปลี่ยน
```

ถ้าเจอ `UnicodeEncodeError` ให้ตั้ง `PYTHONUTF8=1`

## 7. ขั้นตอนประจำวันของแบบที่ 1 (next_day)

คำสั่งเดียว (หลัง 18:00 ของวันทำการ t−1):

```bash
python task_b/daily_next_day.py --dry-run            # ทดสอบ: ทุกขั้น แต่ทำนายแบบ dry-run · ไม่ commit
python task_b/daily_next_day.py --commit --push      # official
```

ทำตามลำดับ: `check_env` → `fetch_yahoo_daily` → `append_investing --source yahoo` → `main.py --dev`
(ต้องไม่เปลี่ยน `*_val.csv`) → commit ① ข้อมูล → `record_outcomes` + `report_live` →
`predict_live --target-date <วันทำการถัดไป> --expected-cutoff <วันสุดท้ายของข้อมูล>` → commit ② + push

- **วันทำการถัดไปมาจาก `task_b/set_holidays.txt`** (predict_live เองไม่เดาวันหยุด) · สคริปต์จะไม่ทำนายวันที่เกิน
  `confirmed_through` → ต้องเพิ่มวันหยุด SET จากประกาศทางการแล้วเลื่อนค่านี้ (ตอนนี้ยืนยันถึง 2026-09-25)
- ไม่มีวันใหม่ (วันหยุด) → ข้ามการ append · คำทำนายของ target มีใน log แล้ว → ข้าม (รันซ้ำได้ปลอดภัย)
- official ต้องเริ่มจาก `task_b/` ที่ไม่มีไฟล์ค้างไม่ commit
- ตั้งเวลาอัตโนมัติ: `task_b/automation/register_tasks.ps1` (Windows Task Scheduler) หรือแม่แบบ GitHub Actions
  `task_b/automation/github_actions_task_b_daily.yml` (ต้องคัดลอกไป `.github/workflows/` ที่ root เอง)

ขั้นตอนแบบมือ (ไฟล์ Investing ที่ดาวน์โหลดเอง) ยังใช้ได้:

```bash
python task_b/tools/append_investing.py <KBANK> <ADVANC>
python task_b/main.py --dev                                  # val ต้องไม่เปลี่ยน
git add task_b/raw_data task_b/raw_data_sources && git commit -m "data: <t−1>"   # ① ข้อมูลก่อน
python task_b/record_outcomes.py && python task_b/report_live.py
python task_b/predict_live.py --target-date <t> --expected-cutoff <t−1>
git add task_b/results && git commit -m "prediction log: <t>" && git push         # ② ก่อน 09:00
```

- `predict_live.py` จะไม่ยอมรันถ้า `*.py` หรือ `raw_data/` ยังไม่ commit หรือ log รอบก่อนยังไม่ commit
- ผลจริงมาจาก `raw_data/` ไฟล์เดียวกับที่เทรน · รายงานอยู่ที่ `task_b/results/live_report/`

## 8. โมเดล 16:00 — ใช้งานจริง (อัตโนมัติ)

ตัวเชื่อมรายชั่วโมง `tools/fetch_yahoo_intraday.py` ดึง Yahoo 1h (รูปแบบเดียวกับไฟล์ล็อก ตรวจแล้ว) แล้วบันทึก
snapshot ผ่าน `live_1600.save_snapshot` · คำสั่งประจำวัน (เพื่อนเรียกแค่นี้):

```bash
python task_b/daily_1600.py --phase predict --official --commit --push   # 16:00–16:25
python task_b/daily_1600.py --phase outcome --official --commit --push   # หลัง 17:00
```

- **ไม่มีขั้นอนุมัติล่วงหน้าแล้ว** (ไม่ต้องมี `PHASE1D_LIVE_APPROVAL.md`) · guard อื่นของ `predict_1600.py`
  ยังอยู่ครบ: 16:00–16:30 · โค้ด/log commit แล้ว · snapshot Yahoo หลัง 16:00 แท่งครบ · ห้ามซ้ำ
- real-time availability ตรวจหลังเกิดทุกวัน: `prev_close_match` (outcomes) + `bar15_changed_vs_later`
  (availability_1600.csv) · วันที่ไม่ผ่านให้รายงานแยก/ตัดออก
- ตั้งเวลา: `register_tasks.ps1` (16:02 / 17:15) — ห้ามใช้ cron ของ GitHub เพราะช้าได้เกินเส้นตาย 16:30
- ผล development ทุก config ยังแพ้ Naive (relMAE 1.051–1.057) — การแก้ตรงนั้นต้องเขียนแผน Phase 1D
  รอบใหม่ (pre-register) ห้ามจูนซ้ำบนข้อมูลเดิม
การแก้ตรงนั้นต้องเขียนแผน Phase 1D รอบใหม่ (pre-register) ห้ามจูนซ้ำบนข้อมูลเดิม
