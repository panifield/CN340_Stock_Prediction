# งาน B: ทำนายราคาปิด (Regression) — Phase 1C (tuned) + Phase 1D (pre-registered)

ทำนายราคาปิดของวันถัดไป โดยทำนายผ่าน **return** (ผลตอบแทน) ก่อน
แล้วค่อยแปลงกลับเป็นราคาบาท — โฟลเดอร์นี้แยกอิสระจาก `task_a/` และ
`task_c/` โดยสมบูรณ์ ไม่ import ไฟล์จากที่อื่นเลย

โมเดล: **ANN (MLP, เฉลี่ย 10 seeds) + Random Forest + XGBoost**
**จูนแล้วใน Phase 1C** (`CONFIG_TAG = "phase1c-tuned"`) ด้วย walk-forward 3 folds × 2 หุ้น บน train+val
ตามแผนที่ commit ก่อนรันใน `TUNING_PLAN.md` — ผลอยู่ใน `results/tuning_*` (ดู post-hoc notes ท้าย `results/tuning_summary.md`)

> **หลังจูน ตาราง fixed validation ของ `main.py --dev` เป็น descriptive development result
> ไม่ใช่ independent holdout** — 360/362 แถวของ val อยู่ใน evaluation window ที่ใช้เลือก hyperparameter
> (validation reuse for hyperparameter selection)
>
> **historical test (2025-02-26 → 2026-08-28) เคยถูกเปิดอ่านแล้วในรุ่นก่อนของโปรเจกต์**
> การเปิดครั้งต่อไปต้องมี `PRE_TEST_LOCK.md` ใหม่ และต้องประกาศว่าเป็นการอ่าน test **ครั้งที่สอง**

---

## วิธีรัน

```bash
pip install -r requirements.txt
python main.py --dev             # พัฒนา: train/val เท่านั้น ไม่แตะ test
python tests/test_pipeline.py    # 18 tests (daily pipeline)
python tests/test_phase1d.py     # 21 tests (Phase 1D builder/metric · ข้อมูลสังเคราะห์ · ~3 นาที)
```

รันจากโฟลเดอร์ไหนก็ได้ (`python task_b/main.py --dev`)
ผลบันทึกที่ `results/<หุ้น>_taskB_val.csv`
RF / XGBoost ใช้ `n_jobs=-1` → ผล numerically reproducible within floating-point precision
(val CSV ที่ตรวจใน regression check ตรงกันทุก byte ในการรันที่ผ่านมา แต่ไม่ได้รับประกันทุกเครื่อง)

**`python main.py` (ไม่มี `--dev`) จะเปิด test set** ซึ่งถูกล็อกไว้:
ต้องมี `PRE_TEST_LOCK.md` ที่กรอกครบ 10 หัวข้อก่อน ไม่งั้นจะ error ทันที
— **ห้ามสร้างไฟล์นี้จนกว่าผู้ใช้ตัดสินใจเปิด test**
โหมดนี้ตัดข้อมูลที่ `config.HISTORICAL_TEST_END` (2026-08-28) ก่อนสร้าง feature
→ แถวที่ append ภายหลังไม่ไหลเข้า historical test

ถ้าเจอ `UnicodeEncodeError` ตอนพิมพ์ภาษาไทย ให้รันด้วย `PYTHONUTF8=1`

---

## ข้อมูล

- อ่านจาก `raw_data/*_10Y_Cleaned.csv` (investing.com) **แหล่งเดียว**
  ไฟล์หาย = error ทันที ไม่มีการไปดึง Yahoo มาแทน
- ทุกครั้งที่โหลดจะพิมพ์ `sha256` ของไฟล์ ไว้ยืนยันว่าใช้ข้อมูลชุดเดิม
- split ตายตัวด้วยวันที่ (`config.SPLIT_BY_DATE`):

| ชุด | ช่วง | แถว |
|---|---|---:|
| train | 2016-09-26 → 2023-09-01 | 1,688 |
| val | 2023-09-04 → 2025-02-25 | 362 |
| test (historical) | 2025-02-26 → 2026-08-28 | 362 |
| post-historical-test | 2026-08-31 → 2026-09-25 | 20 |

ในโหมด `--dev` แถวของ test ถูกตัดทิ้งตั้งแต่ก่อนสร้าง feature

ข้อมูลใน `raw_data/` ปัจจุบันถึง **2026-09-25** (หุ้นละ 2,453 แถว · ไฟล์เริ่ม 2016-08-26 แต่ 21 แถวแรก
เป็น warm-up ของ feature) · 20 แถวล่าสุด append วันที่ 2026-09-27 จาก Investing.com
(ต้นฉบับ + SHA อยู่ใน `raw_data_sources/investing_20260927/SOURCES.md`)

**สถานะของ 20 แถว 2026-08-31 → 2026-09-25:** post-historical-test, pre-freeze historical rows
ที่ถูกเห็นบางส่วนแล้วผ่าน Yahoo intraday probe → **ไม่ใช่ prospective, ไม่ใช่ส่วนของ historical test,
ไม่ใช่ pristine** · ใช้เทรนโมเดล live ได้ (เป็นอดีตที่รู้ผลแล้ว)

---

## โครงสร้างไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `config.py` | path, แหล่งข้อมูล, วันที่ split, hyperparameter, หัวข้อของ lock |
| `data_loader.py` | อ่าน `raw_data/` + fingerprint |
| `features.py` | market 24 ตัว (shift(1), ลงท้าย `_prev`) + calendar 5 ตัว (`dow_*`, ไม่ shift) |
| `targets.py` | `y_return` = close(t) / close(t-1) − 1 |
| `splits.py` | แบ่งตามวันที่ ไม่สร้าง test partition ในโหมด `--dev` |
| `models.py` | ANN / RF / XGBoost + `SeedAveragedRegressor` |
| `baselines.py` | Naive (return = 0) — คู่แข่งหลัก, Mean Return |
| `evaluate.py` | 5 metric + ตาราง + เทียบกับ Naive + `prediction_shape` (diagnostic) |
| `weighting.py` | recency weights (ปิดอยู่: half_life = None ชนะการจูน) |
| `trading_costs.py` | ต้นทุน SET + breakeven / signal diagnostic (hypothetical — ไม่ใช่ backtest) |
| `diagnostics.py` | วิเคราะห์ข้อมูลก่อนเทรน — **train เท่านั้น** |
| `main.py` | ตัวหลัก + lock gate + ตัดข้อมูลตามโหมด |
| `tune.py` / `TUNING_PLAN.md` | การจูน Phase 1C (ห้ามรันซ้ำ / ห้ามแก้แผน) |
| `predict_live.py` | next-day prediction · dry-run เขียน `results/dryrun/` เท่านั้น |
| `tools/` | `check_raw_update.py` (ตรวจ raw_data ใหม่) · `intraday_probe.py` |
| `intraday_1600.py` / `metrics_1600.py` / `tune_1600.py` / `PHASE1D_PLAN.md` | Phase 1D Mode A (16:00) — pre-registered ที่ `0b09905` · search รันครั้งเดียวแล้ว |
| `report_1600.py` | สร้างรายงาน `results/phase1d/summary_1600.md` จากผลค้นหา (อ่าน CSV อย่างเดียว) |
| `predict_1600.py` | Phase 1D live pipeline — **dry-run เท่านั้น** เขียน `results/dryrun/` |
| `tests/test_pipeline.py` | 18 tests: alignment, dow, NaN, lock gate, seed averaging, shape, weighting, costs, fingerprint, dry-run log, historical test cut |
| `tests/test_phase1d.py` | 21 tests ของ Phase 1D (ข้อมูลสังเคราะห์) |
| `results_reference/` | ผลของเวอร์ชันก่อนหน้า — backup เท่านั้น ห้ามเป็น input |

---

## Phase 1D — 16:00 (Mode A) · development / walk-forward, not a test

ณ 16:00 ทำนาย `close_bar16` ของวันเดียวกัน จาก Yahoo รายชั่วโมง (`raw_data_intraday/` เท่านั้น)
แผน: `PHASE1D_PLAN.md` · ผล: `results/phase1d/summary_1600.md` · หลักฐาน/SHA: `results/phase1d/MANIFEST.md`

- **ไม่มี historical test** — ข้อมูล intraday ทั้งหมดเคยถูก probe แล้ว ตัวเลขทั้งหมดเป็น walk-forward บน development data
- ทุก config แพ้ Naive ด้วย MAE (mean relMAE ผู้ชนะ 1.051–1.057) · R2_OOS ผู้ชนะ ≈ +0.03–0.04
- **prospective evaluation ยังไม่เปิด** · **real-time availability ยังไม่ได้พิสูจน์** ·
  `close_bar16` ≠ official SET close · ไม่มี executable backtest
- official `same_day_1600` ยังไม่เปิด (`predict_live.py` มีแค่ `next_day`) · `predict_1600.py` รองรับ `--dry-run` เท่านั้น
- ห้ามวางตัวเลข Phase 1D ในตารางเดียวกับ daily model

```bash
python predict_1600.py --target-date 2026-09-25 --dry-run   # วันในอดีตเท่านั้น
python report_1600.py                                       # สร้างรายงานจากผลที่มีอยู่
```

---

## อ่านผลยังไง

- ตัดสินด้วย **`MAE_return` เทียบกับ `Baseline: Naive (RW)`**
  (target คือ return และ scale-normalized ทุกวันเทียบกันได้)
- `MAE_baht` / `RMSE_baht` ใช้อธิบายขนาด error เป็นบาท
- `R2_return` < 0 = แย่กว่าการทายค่าเฉลี่ยของชุดนั้น
- ไม่มี `R2_price`: ได้ ~0.98 ทุกโมเดลรวมทั้ง naive — วัดข้อมูล ไม่ได้วัดโมเดล
- ไม่มี `DirAcc` ในเฟส 1 (ต้องกำหนดนโยบายวันราคานิ่งก่อน — เฟส 2)

---

## Live prediction (ทำนายวันข้างหน้าจริง)

```bash
# ทดสอบก่อนเสมอ (ใช้วันใน validation ไม่แตะ test)
python predict_live.py --target-date 2025-02-25 --as-of 2025-02-24 --dry-run

# ของจริง -- ต้องระบุวันเอง สคริปต์ไม่เดาวันทำการให้
python predict_live.py --target-date 2026-09-28
```

official: ผล append ลง `results/prediction_log.csv` ทั้ง 3 โมเดล × 2 หุ้น
dry-run: เขียนลง `results/dryrun/prediction_log_dryrun.csv` เท่านั้น (schema เดิม + `worktree_dirty`)
— production log ไม่ถูกแตะ (18 แถว dry-run เก่าใน production log เป็น legacy ยังไม่ได้ย้าย)
`dataset_sha256` = SHA-256 เต็ม 64 hex (แถวเก่าที่มี 16 ตัว = legacy)

### บันทึกผลจริงและรายงาน live test

หลังอัปเดต `raw_data/` ของวัน t แล้ว (ผลจริงมาจาก Investing แหล่งเดียวกับที่เทรน):

```bash
python record_outcomes.py      # จับคู่ log กับราคาปิดจริง -> results/outcomes.csv (append-only)
python report_live.py          # -> results/live_report/live_report.md + live_summary.csv + live_cumulative.png
```

- นับเฉพาะคำทำนาย official (dry-run ต้องสั่ง `--include-dry-run` และเขียนลงไฟล์/โฟลเดอร์อื่นเท่านั้น)
- metric หลัก: relMAE vs Naive, R2_OOS, win rate vs Naive · รอง: Bias, StdRatio, Rho,
  dir hit rate (เฉพาะวันที่ราคาขยับ), long signal (diagnostic) · ความครบ: วันที่ขาด, คำทำนายที่ออกหลัง 09:00
- n < 20 วัน = ผลยังแกว่งมาก อย่าสรุป · ห้ามรวมกับผล validation เดิม

### ทำไมนี่ไม่ใช่การเปิด test set

| | `main.py --dev` | `predict_live.py` |
|---|---|---|
| วัตถุประสงค์ | historical evaluation | prospective prediction |
| เทรนถึง | `train_end` (2023-09-01) | วันล่าสุดที่รู้ผลแล้ว |
| ประเมินกับ | val ที่กันไว้ | อนาคตที่ยังไม่เกิด |

ช่วง 2025-02-26 → 2026-08-28 **ไม่ใช่อนาคตของการทำนายวันพรุ่งนี้อีกแล้ว**
มันคือ labeled data ที่รู้ผลแล้ว การเอามาเทรนเพื่อทำนายวันข้างหน้าจึงถูกต้อง
แต่ **ห้ามเอาผลจาก `predict_live.py` ไปรายงานเป็นผล test set เด็ดขาด**
— คนละวัตถุประสงค์ คนละไฟล์ คนละ log

### Guard ก่อนเขียน log (เฉพาะ non-dry-run)

| guard | กัน |
|---|---|
| clean tree | `*.py` / `raw_data/` ต้อง commit แล้ว ไม่งั้น `code_commit` ใน log ไม่ตรงโค้ดที่รันจริง |
| pending log | log รอบก่อนต้อง commit แล้ว ไม่งั้นหลักฐานของรอบก่อนอ่อนลง |
| duplicate | ห้ามทำนายซ้ำ key `(prediction_type, ticker, target_date, model)` |
| วันที่ | เสาร์-อาทิตย์ / มีข้อมูลแล้ว / อยู่ในอดีต |

ไม่ตรวจวันหยุดของ SET เพราะปฏิทินไทยเดาไม่ได้ — ผู้ใช้ต้องระบุ `--target-date` ที่ถูกเอง

### หลังรันจริงต้อง commit + push ทันที

```bash
git add task_b/results/prediction_log.csv
git commit -m "prediction log: 2026-09-28"
git push
```

การ push commit ที่บรรจุ prediction log ไปยัง remote repository **ก่อน outcome เกิด**
ทำให้มีหลักฐานภายนอกที่ตรวจย้อนกลับได้ว่า commit hash ใดบรรจุ prediction ชุดใด
จึงแข็งแรงกว่าการเก็บ CSV ไว้เฉพาะในเครื่อง
(หมายเหตุ: timestamp ใน git commit ผู้ใช้กำหนดเองได้ สิ่งที่แข็งคือการมี
บันทึกฝั่ง remote ว่า commit ไหนถูก push เมื่อไหร่ ไม่ใช่ตัว commit timestamp เอง)

ทำนายวันจันทร์ได้ตั้งแต่คืนวันอาทิตย์ เพราะใช้ข้อมูลถึงวันศุกร์

---

## อัปเดตข้อมูล

ดาวน์โหลดจาก **investing.com แหล่งเดิมเท่านั้น** แล้ว **append ต่อท้าย** `raw_data/*.csv`
(แถวเก่าต้องเหมือนเดิมทุก byte) · รูปแบบ canonical: `Date,Price,Open,High,Low,Vol. ('000),Change %`
(วันที่ `MM/DD/YYYY` เรียงเก่า→ใหม่, `Price` = ราคาปิด, ตัวเลขแบบ `247.0`, `Vol.` หน่วยพันหุ้น
เช่น `8.65M` → `8650.0`, CRLF, ไม่มี BOM, ไม่มี quote) · เก็บไฟล์ต้นฉบับไว้ใน `raw_data_sources/`
ตรวจด้วย `python tools/check_raw_update.py <เก่า> <ใหม่> --cross-check raw_data_intraday`

**ต้อง regression check ทุกครั้ง** เพราะ split ใช้วันที่ตายตัว
การเพิ่มข้อมูลท้ายไฟล์ต้องไม่กระทบ train/val เลย:

```bash
python main.py --dev
cp results/KBANK_BK_taskB_val.csv /tmp/before_KBANK.csv
cp results/ADVANC_BK_taskB_val.csv /tmp/before_ADVANC.csv
# ... วางไฟล์ใหม่ ...
python main.py --dev
diff /tmp/before_KBANK.csv results/KBANK_BK_taskB_val.csv    # ต้องว่าง
diff /tmp/before_ADVANC.csv results/ADVANC_BK_taskB_val.csv  # ต้องว่าง
```

diff ไม่ว่าง = ข้อมูลใหม่แก้แถวเก่าด้วย **หยุดหาสาเหตุก่อน** อย่าเดินต่อ
`train: 1688 / val: 362` ต้องเท่าเดิม ส่วน `dataset_fingerprint` จะเปลี่ยน
