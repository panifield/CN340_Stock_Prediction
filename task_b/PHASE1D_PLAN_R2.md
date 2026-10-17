# PHASE1D_PLAN_R2 — รอบที่สอง: ณ 16:00 ทำนาย "ราคาปิดทางการ" ของวันเดียวกัน · pre-registration

> เขียนวันที่ 2026-10-17 (เสาร์) บน branch `task_b/target_close` **ก่อน** fit โมเดลใด ๆ บน target ใหม่
> ไฟล์นี้คือ "รอบที่สอง" ตามข้อ 19 ของ `PHASE1D_PLAN.md` · **ไม่แก้ `PHASE1D_PLAN.md` (รอบที่ 1)** ซึ่งยังเป็นบันทึกของผลรอบแรก
> ต้อง commit ไฟล์นี้ + โค้ดทุกไฟล์ใน `tune_1600.LOCKED_FILES_R2` ก่อนรัน `tune_1600.py --run --round r2` (สคริปต์ตรวจเอง)
> หลังรันแล้ว ห้ามเปลี่ยน feature / grid / rule / metric — การเปลี่ยนใด ๆ = รอบที่สาม ต้องมีแผนใหม่ + commit ก่อน
> ไม่แตะ daily pipeline (`TUNING_PLAN.md`, next_day) และไม่แตะ task_a / task_a2 / task_c

**การตัดสินใจของผู้ใช้ (2026-10-17):** prediction_type = `same_day_close` · แผน = `task_b/PHASE1D_PLAN_R2.md` ·
branch `task_b/target_close` · ไม่รันโมเดล target เดิมต่อ (เก็บผลเดิมไว้เป็นประวัติ) · ค้นหา 24 config เดิมใหม่ ·
**D1 = ตัวเลือก B** (ประวัติจากไฟล์ล็อก + snapshot Yahoo รายวัน) · **D2 = ราคาปิดของวันทำการก่อนหน้าจริง** ·
ผลจริงของ `same_day_close` มาจาก `raw_data/` (เหมือน next_day)

---

## 1. เหตุผลและสิ่งที่เปลี่ยนจากรอบที่ 1

target รอบที่ 1 (`close_bar16` = ราคาปิดของแท่ง 1h เวลา 16:00 ของ Yahoo) **ไม่ใช่ราคาปิดทางการของ SET**
ตรงกับราคาปิดรายวันเพียง 42.3% (KBANK) / 38.7% (ADVANC) ของ 723 วัน · ต่างเฉลี่ย 0.51 / 0.87 บาท
(ใหญ่กว่าส่วนต่างระหว่างโมเดลกับ Naive หลายเท่า) · task_a, task_a2, task_c ใช้ราคาปิดรายวันทางการทั้งหมดแล้ว
(task_c แก้แบบเดียวกันเมื่อ 2026-09-29) · ผู้ใช้ยืนยัน 2026-10-17 ว่าอาจารย์ไม่ได้กำหนดให้ใช้แท่ง 16:00

| | รอบที่ 1 (คงไว้เป็นประวัติ) | รอบที่ 2 (แผนนี้) |
|---|---|---|
| target | `C16(t)/C15(t) − 1` | **`Close_official(t)/C15(t) − 1`** |
| feature `gap` | `O10(t)/C16(t_prev) − 1` | **`O10(t)/Close_official(วันทำการก่อนหน้า) − 1`** (D2) |
| feature `y_lag1,2,3,5` | lag ของ target รอบ 1 | lag ของ target รอบ 2 (นิยามเดิม: นับตาม eligible labeled days) |
| eligible labeled day | มีแท่ง 10,11,12,14,15 **และ 16** | มีแท่ง 10,11,12,14,15 **และราคาปิดทางการของวันนั้น** (ไม่อ่านแท่ง 16 เลย) |
| feature อื่น 16 ตัว · grid 24 · walk-forward · metric · guard · tie rule | — | **เหมือนรอบที่ 1 ทุกตัวอักษร** |
| prediction_type | `same_day_1600` | **`same_day_close`** |
| config_tag | `phase1d-modeA` | **`phase1d-r2-close`** |
| ผลการค้นหา | `results/phase1d/` | **`results/phase1d_r2/`** |
| live pipeline | `predict_1600.py` | **`predict_close.py`** (ใหม่ · `predict_1600.py` ไม่ถูกแก้) |

ผลรอบที่ 1 ทั้งหมด (`results/phase1d/`, แถว `same_day_1600` ใน `prediction_log.csv`, `outcomes.csv`,
`prediction_log_1600_meta.csv`, `backfill/results/task_b_1600_t1600.csv`) **ไม่ลบ ไม่แก้** · หยุดผลิตแถว
`same_day_1600` ใหม่ตั้งแต่วันที่ merge (`daily_1600.py` ค่าเริ่มต้น `--round r2`; `--round r1` ยังเรียกรอบ 1 ได้ถ้าจำเป็น)

## 2. Time semantics (Asia/Bangkok, +07:00)

| field | ค่า |
|---|---|
| `last_feature_bar_start` | 15:00 ของวัน t (เดิม) |
| `feature_window_end` (= `data_cutoff`) | 16:00 ของวัน t (เดิม) |
| `target_time` (แทน `target_bar_start`) | ราคาปิดทางการของวัน t = ราคาจากรอบประมูลปิดตลาด (~16:35) · บันทึกเป็น 16:35 เพื่อเป็นหลักฐานว่าเกิดหลังเวลาทำนาย |
| `generated_at` / `snapshot_downloaded_at` | เวลาจริง (เดิม) |

- causal: feature bar ทุกแท่งเริ่มก่อนและจบไม่เกิน 16:00 · ราคาปิดทางการของวัน t เกิดหลัง 16:30 เสมอ →
  ไม่มี feature ใดอ่านค่าที่เกิดหลัง 16:00 ของวัน t · ราคาปิดของวัน t และวันหลังจากนั้นถูกตัดทิ้งก่อนสร้าง feature / label เสมอ
  (`build_1600_live_feature` กรอง `index < target`) · มี test ที่วางยาราคาปิดวัน t และอนาคต แล้วผลต้องไม่เปลี่ยน
- **ข้อจำกัดที่ยังอยู่ (ไม่แก้ในรอบนี้):** Yahoo ดีเลย์ ~15 นาที — snapshot ตอน 16:02 อาจยังไม่มีแท่ง 15:00 ฉบับสุดท้าย
  (เห็นราคา ~15:47) · ยังตรวจหลังเกิดด้วย `prev_close_match` / `bar15_changed_vs_later` เหมือนเดิม
  · ผู้ใช้ยืนยันให้คงเวลาทำนาย 16:00 (รัน 16:02) ตามข้อตกลงกับอาจารย์

## 3. Data (D1 = B)

| ช่วง | แหล่ง |
|---|---|
| แท่ง 1h | `raw_data_intraday/*.csv` (Yahoo · ล็อก SHA เดิม: KBANK `b192ad2c…2060` · ADVANC `2097bb53…e79c`) + live snapshot `raw_data_intraday_live/` (เดิม) |
| ราคาปิดทางการ — dev (≤ 2026-09-25) | **`raw_data_daily/<TICKER>_1d_locked.csv`** ล็อก SHA-256 (KBANK `ab244d8f…ec31` · ADVANC `641dab04…a800`) · สร้างครั้งเดียวจาก `raw_data/` ด้วย `tools/make_locked_daily.py` (ตรวจซ้ำด้วย `--check`) |
| ราคาปิดทางการ — live (> 2026-09-25) | **snapshot Yahoo 1d** (`interval=1d`, `auto_adjust=False`) เก็บ byte-for-byte + meta + SHA-256 ที่ `raw_data_daily_live/<YYYYMMDD>/` (ดึงด้วย `tools/fetch_yahoo_daily_close.py` ตั้งแต่ 2026-09-01 ให้ทับไฟล์ล็อก) |
| ผลจริง (outcome) | `raw_data/` (Investing + Yahoo ที่ตรวจ OHLC แล้ว) เหมือน next_day · บันทึกชื่อไฟล์ + SHA-256 ใน `outcomes.csv` |

- วันที่ทับกัน: ใช้ค่าจากไฟล์ล็อก และรายงานจำนวนวันที่ไม่ตรง (ไม่ใช่ error) · แถวที่ `Volume <= 0` หรือ `Close` ว่างของ snapshot **ไม่ถูกใช้** (`daily_close.usable`)
- **เหตุผลที่ไม่ดึง Yahoo 1d มาเป็นประวัติ (ข้อค้นพบ 2026-10-17):** Yahoo 1d ที่ดึงสดมีแถวเสียปี 2024 — Volume=0 ราคาปิดค้าง
  KBANK 65 วัน / ADVANC 47 วัน (ต่างจาก raw_data สูงสุด 9 / 17 บาท) ทั้งที่เป็นวันซื้อขายจริง · cache เก่าของ task_a ตรง raw_data 100% ในช่วงเดียวกัน
  · ถ้า snapshot ประวัติจาก Yahoo วันนี้ จะได้ข้อมูลเสียถาวร (SHA ช่วยให้รันซ้ำได้ แต่ไม่ทำให้ข้อมูลถูก) · เหลือวันใช้ได้แค่ 624 / 644 แถวเทรน
- ผลต่อกฎรอบ 1: ข้อ "ห้ามอ่าน `raw_data/`" ยกเลิกเฉพาะ **ตอนสร้างไฟล์ล็อกครั้งเดียว** (`tools/make_locked_daily.py`) และ **ตอนบันทึกผลจริง** (`record_outcomes.py`)
  · โมดูล Phase 1D (`intraday_1600`, `daily_close`, `predict_close`, `tune_1600`) ยังไม่ import `data_loader` (มี test)
- **guard live:** ต้องมีราคาปิดทางการของ "วันทำการก่อนหน้า" (ตามปฏิทิน `set_holidays.txt`) ใน snapshot มิฉะนั้น **raise ไม่ทำนาย** (ห้ามทำนายด้วยข้อมูลเก่าเงียบ ๆ)
- **dev data:** วันที่ ≤ 2026-09-25 (วันสุดท้ายของไฟล์ 1h ที่ล็อก) · ทั้งหมดคือ development · **ไม่มี historical test**
- ห้ามใช้ข้อมูลหลัง 2026-09-25 ในการเลือก config (รวมผล live/backfill ช่วง 2026-09-28 → 2026-10-15 และ rolling backtest)

### Eligible days (นับก่อน commit — นับวัน ไม่ดูค่า target)

- labeled: มีแท่ง 10, 11, 12, 14, 15 + มีราคาปิดทางการ · live day: มีแท่ง 10, 11, 12, 14, 15
- แท่ง 13 ไม่อ่าน · แท่ง 16 ไม่อ่านเลยในรอบนี้ (ทั้ง feature และ target) · วันไม่ครบ → ตัดทั้งวัน ห้าม ffill

| | KBANK | ADVANC |
|---|---|---|
| วันในไฟล์ 1h | 722 | 722 |
| eligible labeled days | **720** | **719** |
| วันที่ถูกตัด (แท่งที่ขาด) | 2025-03-28 [15] · 2026-04-20 [10, 11, 12, 14, 15] | 2025-03-28 [15] · 2025-09-16 [14] · 2026-04-20 [10, 11, 12, 14, 15] |
| warm-up ที่ตัด (20 eligible rows แรก) | 2023-10-02 → 2023-10-31 | 2023-10-02 → 2023-10-31 |
| แถว NaN/Inf กลางชุดที่ตัด | ไม่มี | ไม่มี |
| **แถวหลังตัด (n)** | **700** (2023-11-01 → 2026-09-25) | **699** (2023-11-01 → 2026-09-25) |

(ชุดวันและขนาดเท่ารอบ 1 ทุกประการ — ต่างจากรอบ 1 เฉพาะ 2025-03-28 ที่ไม่ต้องมีแท่ง 16 แล้ว แต่ขาดแท่ง 15 จึงยังถูกตัด)

## 4. Target

```
y(t) = Close_official(t) / C15(t) − 1          predicted_close = C15(t) × (1 + ŷ)
```

## 5. Features (21 ตัว — ชื่อและลำดับเดิม)

เหมือนข้อ 5 ของรอบที่ 1 ยกเว้น:
- `gap` = `O10(t) / Close_official(วันทำการก่อนหน้า) − 1` (D2 · gap ข้ามคืนจริง เหมือน `overnight_gap` ของ task_c) —
  "วันทำการก่อนหน้า" ในข้อมูลประวัติ = แถวก่อนหน้าในไฟล์ล็อก (ไฟล์มีทุกวันซื้อขาย) · ใน live = ต้องตรงกับวันทำการก่อนหน้าตามปฏิทิน (guard ข้อ 3)
  · ต่างจากรอบ 1 ที่ใช้ C16 ของ eligible labeled day ก่อนหน้า (ข้ามวันที่ถูกตัดได้ ~3 วัน)
- `y_lag1,2,3,5` = lag ของ y รอบ 2 นับตามลำดับ eligible labeled days (นิยาม t_prev เดิม)
- warm-up 20 eligible rows · NaN/Inf กลางชุด → ตัดและรายงาน (เดิม)

## 6–15. Baselines · Models · Grid · Walk-forward · Primary score · Guard · Tie · NaN policy · Metrics · Economic diagnostics

**เหมือนข้อ 6–15 ของ `PHASE1D_PLAN.md` ทุกข้อ** (Naive ŷ = 0 → ราคาปิด = C15 · Mean Return · 3 models ค่าคงที่เดิม ·
ANN seeds search/live [0,1,2] sensitivity range(10) · grid 24 configs เดิม · `walk_forward_splits(n_splits=3, min_train=360)`
· primary = mean relMAE ของ 6 evaluation · guard StdRatio ≥ 0.05 ทุก evaluation · tie 1e-3 · NaN policy เดิม)
ตาราง fold (คำนวณจากข้อมูลจริงก่อน commit — ใช้ขนาด/วันที่เท่านั้น ไม่ได้ดูค่า target):

| หุ้น | fold | train | eval |
|---|---|---|---|
| KBANK (n=700, fold 113) | 1 | 360 แถว 2023-11-01 → 2025-04-25 | 113 แถว 2025-04-28 → 2025-10-15 |
| | 2 | 473 แถว 2023-11-01 → 2025-10-15 | 113 แถว 2025-10-16 → 2026-04-02 |
| | 3 | 586 แถว 2023-11-01 → 2026-04-02 | 113 แถว 2026-04-03 → 2026-09-24 |
| ADVANC (n=699, fold 113) | 1 | 360 แถว 2023-11-01 → 2025-04-25 | 113 แถว 2025-04-28 → 2025-10-16 |
| | 2 | 473 แถว 2023-11-01 → 2025-10-16 | 113 แถว 2025-10-17 → 2026-04-03 |
| | 3 | 586 แถว 2023-11-01 → 2026-04-03 | 113 แถว 2026-04-07 → 2026-09-25 |

Omitted evaluation dates: KBANK 2026-09-25 · ADVANC ไม่มี (ไม่ใช่ holdout/test · ใช้เทรน live ได้)

ต่างจากรอบ 1 เฉพาะ:
- ข้อความบังคับในรายงานเปลี่ยนเป็น "ราคาปิดทางการ = C15 ใน X% ของวัน" โดยคำนวณ X จากข้อมูลจริงตอนรัน (`phase1d_summary.md`)
  ⇒ MAE ยังเอื้อ prediction ใกล้ 0 · **ห้ามวางตัวเลขรอบ 2 ในตารางเดียวกับรอบ 1 หรือ daily model** (คนละ target)

## 16. Live pipeline (`predict_close.py`)

`build_1600_live_feature(bars, target_date, daily_close)` — ลำดับเดิม 1–5 + :
- ราคาปิดทางการที่ใช้ = ทุกวัน < target_date เท่านั้น (ตัดวัน target และอนาคตก่อนสร้าง feature/label เสมอ)
- ชุดเทรน = labeled rows ที่วันที่ < target_date (เดิม)
- guard official ของ `predict_1600.py` ครบ (library pin · 16:00–16:30 · commit แล้ว · log รอบก่อน commit · snapshot Yahoo หลัง 16:00 มีแท่งบังคับ · ห้ามซ้ำ)
  + (7) ต้องมี snapshot ราคาปิดรายวันที่ดาวน์โหลดในวันเดียวกับ target + (8) ต้องมีราคาปิดของวันทำการก่อนหน้า
- dry-run ย้อนหลังจากไฟล์ล็อก: ตรวจ `historical_x_match` (X_live ต้องเท่ากับ X ของ historical builder)

## 17. Logging

- `prediction_log.csv` schema เดิม · `prediction_type = same_day_close` · `config_tag = phase1d-r2-close` ·
  `prev_close := C15` · `predicted_close := C15 × (1 + ŷ)` · `data_cutoff := feature_window_end` ·
  `dataset_sha256 :=` SHA-256 ของ live snapshot รายชั่วโมง (เดิม) · `code_commit :=` HEAD ที่ tree สะอาด
- sidecar ใหม่ `results/prediction_log_close_meta.csv` (ไม่ต่อท้ายไฟล์รอบ 1) = คอลัมน์เดิม โดยเปลี่ยน
  `target_bar_start` → `target_time` และเพิ่ม `daily_close_path/source/sha256/last_date/mismatch_rows/dropped_rows`
- dry-run เขียน `results/dryrun/prediction_close_dryrun.csv` (ไฟล์ใหม่ · `predicted_close_bar16` → `predicted_close`)

## 18. Prospective evaluation (ผลจริง)

- `record_outcomes.py` รองรับ `same_day_close`: `actual_close` = ราคาปิดทางการวัน t จาก `raw_data/` (ผู้ใช้เลือก) ·
  `actual_return = actual_close / C15 − 1` (C15 ที่ log ไว้ตอนทำนาย) · `prev_close_match` = C15 ที่ใช้ทำนาย เทียบแท่ง 15:00 จาก snapshot
  รายชั่วโมงหลัง 17:00 ของวัน t · **บันทึกเมื่อมีทั้งสองอย่าง** (ราคาปิดใน raw_data + snapshot หลังตลาดปิด — ปกติคือเช้าวัน t+2)
- รันอัตโนมัติในรอบเช้าของ `daily_next_day.py` (ไม่ต้องเพิ่ม schedule ใหม่ · ไม่แก้ไฟล์นั้น)
- `report_live.py` แยกหัวข้อ `same_day_close` (หน้าต่างเวลา 16:00–16:30 เหมือน `same_day_1600`) · ไม่รวมกับ `same_day_1600`
- **วิธีวัดหลัก = เทียบกับ C15 ที่ log ไว้** (สิ่งที่ระบบรู้จริงตอนทำนาย) · ตาราง "ปรับฐานด้วย C15 สุดท้าย" เป็นตารางเสริม ·
  ห้ามรายงานเฉพาะแถว `prev_close_match = True` เป็นผลหลัก (คัดวันสงบออก = เอนเอียง)
- ห้ามสรุปว่าชนะ/แพ้ Naive ก่อนมีอย่างน้อย 60 วันทำการ · ทดสอบด้วย Diebold–Mariano

## 19. Backfill รอบ 2 ช่วง 2026-09-28 → 2026-10-15 (ผู้ใช้ขอ)

- ทำหลังได้ผู้ชนะรอบ 2 แล้ว · ใช้โค้ด/ผู้ชนะชุดเดียวกับ live · **ไม่เขียนลง `prediction_log.csv`**
- เขียน `results/phase1d_r2/backfill_same_day_close.csv` พร้อม `is_backfill = True`, `bar_source`, ผลจริง:

| วัน | แท่ง 1h ที่ใช้ | bar_source |
|---|---|---|
| 2026-10-01, 10-02, 10-05, 10-12, 10-14, 10-15 | snapshot จริงตอน ~16:02 (`raw_data_intraday_live/`) | `snapshot_1602` |
| 2026-09-30 | snapshot ล่าสุดก่อน 16:30 (16:22) | `snapshot_before_1630` |
| 2026-09-28, 10-06 | snapshot หลังปิดตลาด (23:54 / 18:27) — ใช้แค่แท่ง 10–15 ของ target | `snapshot_after_close` |
| 2026-09-29, 10-07, 10-08, 10-09 | ไม่มี snapshot → แท่งจาก Yahoo ที่แคชไว้ใน `backfill/data/task_b_1h_snapshots/` | `yahoo_final` |

- ราคาปิดรายวัน: snapshot Yahoo 1d ที่ดึงครั้งเดียวในวัน backfill (เก็บใน `raw_data_daily_live/` พร้อม SHA) ใช้เฉพาะวัน < target ·
  ผลจริงจาก `raw_data/` · ผ่าน leak test แบบเดียวกับ `backfill/leak_test.py` (วางยาข้อมูลหลังเวลาตัด → ผลต้องไม่เปลี่ยน)
- **ไม่ใช่ผลทำนายสด** (สร้างหลังรู้ผล) · ใช้ได้ในฐานะผลนอกตัวอย่างเพราะการเลือก config ใช้ข้อมูล ≤ 2026-09-25 เท่านั้น · ห้ามนำผลช่วงนี้ย้อนกลับไปเลือก config
- เครื่องมือ: `tools/backfill_close_r2.py` (เขียนหลังได้ผู้ชนะ · ไม่อยู่ในไฟล์ที่ล็อกสำหรับการค้นหา)

## 20. การเก็บรอบที่ 1

- ไม่ลบ/แก้ไฟล์ผลรอบ 1 ใด ๆ (ข้อ 1) · `intraday_1600.build_dataset()` ยังสร้าง target รอบ 1 ได้เมื่อไม่ส่ง `daily_close`
  (เพื่อ reproduce `results/phase1d/` และให้ tests รอบ 1 ผ่านเหมือนเดิม) · `predict_1600.py` ไม่ถูกแก้
- `same_day_1600` ใน `record_outcomes.py` / `report_live.py` คงไว้สำหรับแถวเก่า

## 21. ไฟล์ที่แก้ / สร้าง

| ไฟล์ | สิ่งที่ทำ |
|---|---|
| `PHASE1D_PLAN_R2.md` | แผนนี้ (ใหม่) |
| `daily_close.py` | โมดูลราคาปิดรายวัน: ไฟล์ล็อก+SHA · snapshot · merge · guard วันทำการก่อนหน้า (ใหม่) |
| `raw_data_daily/` | ไฟล์ล็อกราคาปิด 2 ไฟล์ + README (ใหม่ · ห้ามแก้) |
| `tools/make_locked_daily.py` | สร้าง/ตรวจไฟล์ล็อกจาก `raw_data/` (ใหม่) |
| `tools/fetch_yahoo_daily_close.py` | ดึง snapshot Yahoo 1d (ใหม่) |
| `intraday_1600.py` | `build_dataset` / `_day_table` / `_features` / `build_1600_live_feature` รับ `daily_close` (ไม่ส่ง = รอบ 1 เหมือนเดิม) |
| `tune_1600.py` | `--round r2` → ใช้ราคาปิดไฟล์ล็อก · เขียน `results/phase1d_r2/` · `LOCKED_FILES_R2` · grid/rule เดิม |
| `predict_close.py` | live pipeline รอบ 2 (ใหม่) |
| `record_outcomes.py` | รองรับ `same_day_close` · คง `next_day` / `same_day_1600` เดิม |
| `report_live.py` | หน้าต่างเวลาของ `same_day_close` |
| `daily_1600.py` | `--round r2` (ค่าเริ่มต้น) · ดึง snapshot ราคาปิดรายวัน · เรียก `predict_close.py` · commit `raw_data_daily_live` |
| `../automation/run_1600.py` | เช็คซ้ำด้วย `same_day_close` |
| `../deploy/precheck.py` | `LOGS["t1600"]` ของ task_b → `same_day_close` |
| tests (ใหม่): `tests/test_phase1d_r2.py` (18) · `tests/test_close_r2.py` (9) · `../deploy/test_precheck.py` (+1) | test รอบ 1 เดิมไม่ถูกแก้ ต้องผ่านเหมือนเดิม |
| เอกสาร: `README.md`, `../KNOWN_ISSUES.md` | อธิบายรอบ 2 + ข้อค้นพบ Yahoo 1d |
| `tools/backfill_close_r2.py` | backfill ข้อ 19 (เขียนทีหลัง) |

**ไม่แก้:** `models.py`, `splits.py`, `config.py`, `weighting.py`, `trading_costs.py`, `metrics_1600.py` (ใช้ร่วมกับ next_day / รอบ 1) ·
`PHASE1D_PLAN.md` · `predict_1600.py` · `live_1600.py` · `raw_data_intraday/` · ไฟล์ผลรอบ 1 · `daily_next_day.py` · task_a / task_a2 / task_c

## 22. ลำดับงานและจุด commit (ทุก commit ต้องได้รับอนุญาตจากผู้ใช้ก่อน)

| ขั้น | งาน | commit | สถานะ |
|---|---|---|---|
| 0 | สร้าง branch `task_b/target_close` จาก `main` | — | ทำแล้ว |
| 1 | ผู้ใช้ตอบ D1, D2 · เติมตาราง eligible days | — | ทำแล้ว |
| 2 | แก้โค้ด + tests (ข้อมูลสังเคราะห์) · รัน test ทั้งหมดของ task_b + deploy + task อื่น | — | ทำแล้ว |
| 3 | **commit #1 "lock"**: แผน + โค้ด + tests + ไฟล์ล็อก (ก่อนเห็น target จริง) | ✓ | **รออนุญาต** |
| 4 | `tune_1600.py --run --round r2` (ข้อมูลจริง ครั้งเดียว) + รายงาน | — | |
| 5 | **commit #2**: `results/phase1d_r2/` | ✓ | |
| 6 | dry-run ย้อนหลังด้วยผู้ชนะจริง + `historical_x_match` | — | |
| 7 | เขียน `tools/backfill_close_r2.py` + backfill ข้อ 19 + leak test | — | |
| 8 | **commit #3**: ผล backfill + snapshot ราคาปิดรายวัน | ✓ | |
| 9 | รัน test ทั้งหมดอีกรอบ (task_b, deploy, task_a/a2/c ไม่กระทบ) + `run_1600.py --dry-run` | — | |
| 10 | ผู้ใช้ตรวจ → merge เข้า `main` **หลังรอบ t1600 (16:30) ของวันนั้น หรือวันหยุด** · push | ✓ (merge) | |

ถ้าไม่เสร็จก่อนวันทำการถัดไป: `main` ยังเป็นโค้ดรอบ 1 → รอบ 16:02 จะผลิตแถว `same_day_1600` ต่อ (เก็บเป็นประวัติรอบ 1)
ไม่เสียหาย · Railway ดึง `main` สดทุกรอบ → **ห้าม push งานที่ยังไม่ผ่าน test เข้า `main`** ·
ก่อน merge ต้อง commit snapshot ราคาปิดรายวันของวันทำการก่อนหน้าให้ครบ (ระบบดึงเองทุกรอบ 16:02 แต่ต้องมีราคาปิดของวันก่อนหน้าใน Yahoo)

## 23. การตัดสินใจที่ตกลงแล้ว

| เรื่อง | ผล |
|---|---|
| D1 แหล่งราคาปิด | **B**: ประวัติ = ไฟล์ล็อกจาก `raw_data/` · live = snapshot Yahoo 1d + SHA |
| D2 ฐานของ `gap` | ราคาปิดทางการของวันทำการก่อนหน้าจริง |
| ผลจริง (outcome) | `raw_data/` |
| ข้อค้นพบ Yahoo เสียปี 2024 กระทบ task อื่น | บันทึกไว้ใน `KNOWN_ISSUES.md` ไม่แก้ในงานนี้ |

## 24. สิ่งที่ผู้วางแผนเห็นแล้วก่อนเขียนแผน (เปิดเผยเพื่อความโปร่งใส)

- ไม่เคย fit โมเดลใดบน target รอบ 2 (ยกเว้น test ที่ใช้ข้อมูลสังเคราะห์ และ dry-run ทดสอบระบบด้วย config id 0 ใน temp dir ซึ่งไม่ได้ดูค่า error)
- เห็นสถิติรวมของ target แล้ว: ราคาปิดทางการ = C15 ใน ~36.8% (KBANK) / 35.1% (ADVANC) ของวัน · `close_bar16` ตรงราคาปิดรายวัน 42.3% / 38.7%
  (ใช้เป็นเหตุผลของแผน ไม่ได้ใช้เลือก config)
- เห็นผลรอบ 1, rolling backtest (ช่วง 2025-12-22 → 2026-10-09) และการทดลองใน scratchpad บน target รอบ 1
  (ปัดลงช่องราคา / feature VWAP) — **ไม่นำมาใช้ในรอบนี้** · grid และ feature คงตามรอบ 1 ทุกตัวยกเว้น gap
- เห็นว่า Yahoo 1d ที่ดึงสดมีแถวเสียปี 2024 (ใช้เป็นเหตุผลเลือก D1 = B)
