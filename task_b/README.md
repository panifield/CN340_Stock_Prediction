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
python main.py --dev             # พัฒนา: train/val เท่านั้น ไม่แตะ test (ต้องระบุโหมดเสมอ)
python tests/test_pipeline.py    # 20 tests (daily pipeline)
python tests/test_phase1d.py     # 21 tests (Phase 1D builder/metric · ข้อมูลสังเคราะห์ · ~3 นาที)
```

### งานตอบ feedback 14 ก.ย. 2026 (ผลรายงานเท่านั้น — ไม่เปลี่ยนโมเดลที่ใช้ทำนายจริง)

```bash
pip install -r requirements-experiments.txt   # = requirements.txt + optuna / scipy / matplotlib / joblib

python split_profile.py                  # ข้อ 5: distribution ของ train/val/test (descriptive)
python stats_tests.py --walkforward      # ข้อ 2: DM test บน walk-forward 6 evaluation
python finetune_experiment.py --run      # ข้อ 7: pretrain + fine-tune (มี --smoke จับเวลา)
python optuna_features.py --run          # ข้อ 3: ค้น feature group + hyperparameter (มี --smoke)
python main.py --dev --curves            # ข้อ 8: REC curve + ANN loss/alpha curve

python tests/test_stats.py                     # 13 tests ของ stats_tests.py
python tests/test_features_experimental.py      # 16 tests ของ features_experimental.py
```

`main.py --dev` เซฟไฟล์ diagnostic เพิ่มด้วย (ข้อ 10 / ข้อ 2 / ข้อ 4):
`*_val_shape.csv`, `*_val_costs.csv`, `*_val_preds.csv`, `*_val_stats.csv`, `cost_sensitivity.csv`
— **`*_taskB_val.csv` (ตารางหลัก) ไม่เปลี่ยนชื่อและไม่เปลี่ยนเนื้อหา**

สถานะ feedback ทั้ง 12 ข้อ → [`FEEDBACK_STATUS.md`](FEEDBACK_STATUS.md)

รันจากโฟลเดอร์ไหนก็ได้ (`python task_b/main.py --dev`)
ผลบันทึกที่ `results/<หุ้น>_taskB_val.csv`
RF / XGBoost ใช้ `n_jobs=1` (เปลี่ยนจาก `-1` เมื่อ 2026-09-29) → รันซ้ำบนเครื่องเดียวกัน
ได้ผล byte-identical · **ข้ามเครื่อง/ข้าม platform ยังไม่การันตี** — XGBoost ต่างได้ราว
0.03–0.08% ระหว่าง Windows กับ Linux จึงต้องตรวจด้วย tolerance ไม่ใช่ byte-ต่อ-byte
(`daily_next_day.check_val_regression`, rtol 0.5% / atol 1e-5 · ดู `KNOWN_ISSUES.md`)

**`python main.py` เปล่า ๆ จะ error** — ต้องเลือก `--dev` หรือ `--open-test` เอง
`--open-test` เปิด historical test ซึ่งถูกล็อกไว้:
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
| `daily_next_day.py` | งานประจำวัน next_day ครบในคำสั่งเดียว (ดึง Yahoo → append → regression → ทำนาย → commit) |
| `daily_1600.py` | งานประจำวันโมเดล 16:00 (snapshot → dry-run/official → ตรวจ availability) |
| `set_holidays.txt` | วันหยุด SET + `confirmed_through` (ใช้หา target_date) |
| `requirements-live.txt` | pin ของโมเดล + yfinance สำหรับตัวเชื่อม |
| `automation/` | `register_tasks.ps1` (Windows Task Scheduler) · แม่แบบ GitHub Actions |
| `tools/` | `fetch_yahoo_daily.py` / `fetch_yahoo_intraday.py` (ตัวเชื่อม) · `check_env.py` · `check_raw_update.py` · `intraday_probe.py` |
| `intraday_1600.py` / `metrics_1600.py` / `tune_1600.py` / `PHASE1D_PLAN.md` | Phase 1D Mode A (16:00) — pre-registered ที่ `0b09905` · search รันครั้งเดียวแล้ว |
| `report_1600.py` | สร้างรายงาน `results/phase1d/summary_1600.md` จากผลค้นหา (อ่าน CSV อย่างเดียว) |
| `predict_1600.py` | Phase 1D live pipeline — **dry-run เท่านั้น** เขียน `results/dryrun/` |
| `tests/test_pipeline.py` | 20 tests: alignment, dow, NaN, lock gate, seed averaging, shape, weighting, costs, fingerprint, dry-run log, historical test cut, **เซฟตาราง shape/costs/preds/stats + cost_sensitivity (ข้อ 10/2/4), ไฟล์ val จริงมีครบทุกโมเดล** |
| `tests/test_phase1d.py` | 21 tests ของ Phase 1D (ข้อมูลสังเคราะห์) |
| `results_reference/` | ผลของเวอร์ชันก่อนหน้า — backup เท่านั้น ห้ามเป็น input |
| **ไฟล์ใหม่ของรอบ feedback 14 ก.ย. 2026** | |
| `REFERENCES.md` | ข้อ 4: แหล่งอ้างอิงทั้งหมด (ข้อมูล / ค่าธรรมเนียม / วิธีการ) + ตารางโยงโค้ด→reference |
| `MODEL_RATIONALE.md` | ข้อ 11: ทำไมทำนาย return · แต่ละโมเดลแก้ปัญหาอะไร · พารามิเตอร์มาจากไหน |
| `FEEDBACK_STATUS.md` | สถานะ feedback ทั้ง 12 ข้อ + ไฟล์หลักฐาน + ที่มาของไฟล์ผล (Windows vs Linux) |
| `split_profile.py` | ข้อ 5: distribution ของ train/val/test + KS / Levene (descriptive · ไม่เทรนบน test) |
| `stats_tests.py` | ข้อ 2: DM test + HLN correction · direction metrics · การแยก MAE วันนิ่ง/วันขยับ |
| `features_experimental.py` | ข้อ 3: จัด 29 feature เป็น 7 กลุ่ม + กลุ่มใหม่ `cross_stock` (**ไม่แก้ `features.py`**) |
| `finetune_experiment.py` | ข้อ 7: pretrain(old) → fine-tune(recent) เทียบ pooled บน walk-forward 6 evaluation |
| `optuna_features.py` | ข้อ 3: Optuna TPE ค้น feature group + hyperparameter (fold 1–2 ค้น · fold 3 ยืนยัน) |
| `requirements-experiments.txt` | pin ของสคริปต์ทดลองข้างบน (**ไม่แก้ `requirements.txt`**) |
| `tests/test_stats.py` | 13 tests ของ `stats_tests.py` (ข้อมูลสังเคราะห์ + ตรวจ HLN กับค่าคำนวณมือ) |
| `tests/test_features_experimental.py` | 16 tests ของ `features_experimental.py` (รวม negative control ของ leak test + ตาข่ายนิรภัย `dev_only`) |
| `results/feedback/` | แผนที่เขียนก่อนรัน (`FINETUNE_PLAN.md`, `OPTUNA_PLAN.md` + SHA-256 — ไม่มี commit คั่น ระหว่างเขียนแผนกับรัน SHA-256 ยืนยันได้แค่ว่าไฟล์ไม่ถูกแก้หลังบันทึก hash ไม่ใช่ pre-registration แบบมี commit จริงเหมือน Phase 1D) · ผล · `WHY_NAIVE_WINS.md` · `DECISIONS.md` |

---

## Phase 1D — 16:00 (Mode A) · development / walk-forward, not a test

ณ 16:00 ทำนาย `close_bar16` ของวันเดียวกัน จาก Yahoo รายชั่วโมง (`raw_data_intraday/` เท่านั้น)
แผน: `PHASE1D_PLAN.md` · ผล: `results/phase1d/summary_1600.md` · หลักฐาน/SHA: `results/phase1d/MANIFEST.md`

- **ไม่มี historical test** — ข้อมูล intraday ทั้งหมดเคยถูก probe แล้ว ตัวเลขทั้งหมดเป็น walk-forward บน development data
- ทุก config แพ้ Naive ด้วย MAE (mean relMAE ผู้ชนะ 1.051–1.057) · R2_OOS ผู้ชนะ ≈ +0.03–0.04
- `close_bar16` ≠ official SET close · ไม่มี executable backtest
- official `same_day_1600`: **เปิดใช้ได้** (ยกเลิกขั้นอนุมัติล่วงหน้า 2026-09-28 — ดู `PHASE1D_PLAN.md` ข้อ 18)
  · real-time availability ตรวจ **หลังเกิดทุกวัน** แทน (ดูด้านล่าง)
  · `predict_live.py` มีแค่ `next_day` (16:00 ใช้ `predict_1600.py`)
- ห้ามวางตัวเลข Phase 1D ในตารางเดียวกับ daily model

```bash
python predict_1600.py --target-date 2026-09-25 --dry-run   # dry-run ย้อนหลังจากไฟล์ล็อก
python report_1600.py                                       # สร้างรายงานจากผลค้นหาที่มีอยู่
```

### ใช้งาน 16:00 ทุกวันทำการ

```bash
python daily_1600.py --phase predict --official --commit --push   # 16:00–16:25: snapshot Yahoo -> ทำนาย -> commit + push
python daily_1600.py --phase outcome --official --commit --push   # หลัง 17:00: snapshot ผลจริง -> ตรวจแท่ง 15:00 -> outcomes
python daily_1600.py --phase predict                              # dry-run (ทดสอบ ไม่เขียน production log)
```

ตั้งเวลาอัตโนมัติ: `automation/register_tasks.ps1` (16:02 / 17:15)

guard ของ `--official`: เวอร์ชัน library ตรง pin · รันในวันเดียวกับ target ช่วง 16:00–16:30 ·
โค้ด/แผน/ผลค้นหา commit แล้ว · log รอบก่อน commit แล้ว · snapshot แหล่ง Yahoo ดาวน์โหลด ณ/หลัง 16:00 ครบทุกหุ้น
และมีแท่ง 10, 11, 12, 14, 15 · ห้ามทำนายซ้ำ key เดิม · ข้อมูลที่ schema เดิมไม่มีช่อง (เวลา/แหล่ง snapshot)
อยู่ใน `results/prediction_log_1600_meta.csv`

**ตรวจ real-time availability หลังเกิด (ทุกวัน อัตโนมัติใน `--phase outcome`):**
- `results/outcomes.csv` → `prev_close_match` = close_bar15 ที่ใช้ทำนาย ตรงกับ snapshot หลัง 17:00 ไหม
  (`report_live.py` นับเป็น `prev_close_mismatch`)
- `results/dryrun/availability_1600.csv` → `bar15_changed_vs_later` = แท่ง 15:00 (OHLCV) เปลี่ยนไหม
- วันที่ไม่ผ่าน = ตอน 16:00 แท่ง 15:00 ยังไม่จบ → **รายงานแยก/ตัดออก** ตอนวิเคราะห์ผล

---

## Phase 1D รอบ 2 — 16:00 ทำนาย "ราคาปิดทางการ" (branch `task_b/target_close`)

รอบ 1 (ด้านบน) ทำนาย `close_bar16` ซึ่งตรงราคาปิดทางการเพียง ~39–42% ของวัน → รอบ 2 เปลี่ยน target เป็น
`ราคาปิดทางการ(t) / C15(t) − 1` เหมือน task_a / task_a2 / task_c · แผน (pre-registration): `PHASE1D_PLAN_R2.md`
· feature อีก 16 ตัว · grid 24 config · walk-forward · metric · guard **เหมือนรอบ 1 ทุกตัวอักษร** (ต่างแค่ target และ `gap`)

| รายการ | รอบ 1 | รอบ 2 |
|---|---|---|
| target | `C16/C15 − 1` | `ราคาปิดทางการ/C15 − 1` |
| `gap` | `O10 / C16(eligible day ก่อนหน้า) − 1` | `O10 / ราคาปิดทางการ(วันทำการก่อนหน้า) − 1` |
| prediction_type · config_tag | `same_day_1600` · `phase1d-modeA` | `same_day_close` · `phase1d-r2-close` |
| ผลค้นหา | `results/phase1d/` | `results/phase1d_r2/` |
| live | `predict_1600.py` | `predict_close.py` (ค่าเริ่มต้นของ `daily_1600.py --round r2`) |
| sidecar | `prediction_log_1600_meta.csv` | `prediction_log_close_meta.csv` |

แหล่งราคาปิดรายวัน (`daily_close.py`): ประวัติ ≤ 2026-09-25 = `raw_data_daily/` (ล็อก SHA · ที่มาและเหตุผลใน README ในโฟลเดอร์นั้น) ·
live = snapshot Yahoo 1d ที่ `raw_data_daily_live/<วัน>/` (ดึงด้วย `tools/fetch_yahoo_daily_close.py` · แถว Volume<=0 ไม่ถูกใช้) ·
ผลจริง (`record_outcomes.py`) = `raw_data/` เหมือน next_day · ถ้าไม่มีราคาปิดของวันทำการก่อนหน้า `predict_close.py` จะไม่ทำนาย

```bash
python tests/test_phase1d_r2.py      # 18 tests (builder/daily_close · smoke ~3 นาที)
python tests/test_close_r2.py        # 9 tests (predict_close / outcome / daily_1600 · ~2 นาที)
python tune_1600.py --run --round r2 # ค้นหา 24 config (ต้อง commit แผน R2 + โค้ดก่อน) -> results/phase1d_r2/
python predict_close.py --target-date 2026-09-24 --dry-run   # replay จากไฟล์ล็อก
python tools/backfill_close_r2.py --run         # backfill 28 ก.ย.–15 ต.ค. (ไม่ใช่ผลสด) -> results/phase1d_r2/backfill_same_day_close.csv
python tools/backfill_close_r2.py --leak-test   # วางยาข้อมูลหลังเวลาตัด ผลต้องไม่เปลี่ยน (เกินกว่า noise ของระบบไม่ได้)
python tests/test_backfill_close_r2.py          # 6 tests ของเครื่องมือ backfill
```

## อ่านผลยังไง

- ตัดสินด้วย **`MAE_return` เทียบกับ `Baseline: Naive (RW)`**
  (target คือ return และ scale-normalized ทุกวันเทียบกันได้)
- `MAE_baht` / `RMSE_baht` ใช้อธิบายขนาด error เป็นบาท
- `R2_return` < 0 = แย่กว่าการทายค่าเฉลี่ยของชุดนั้น
- ไม่มี `R2_price`: ได้ ~0.98 ทุกโมเดลรวมทั้ง naive — วัดข้อมูล ไม่ได้วัดโมเดล
- **`DirAcc` มีแล้ว** แต่อยู่คนละตาราง: `stats_tests.direction_metrics` คืน
  `dir_hit_rate_moved_days` (นับเฉพาะวันที่ `y != 0` เพราะวันราคานิ่งไม่มีทิศที่ถูกให้ทาย
  = นโยบายวันราคานิ่งที่เคยค้างไว้) + `frac_zero_move` / `frac_pred_zero` / IC
  (Pearson, Spearman) · เซฟที่ `results/<หุ้น>_taskB_val_stats.csv` และ
  `results/feedback/walkforward_stats.csv`
  **ไม่อยู่ในตารางหลักและไม่ถูกใช้เลือกโมเดล** (การเลือกยังใช้ `MAE_return` เท่านั้น)
- **DM test มีแล้ว**: `stats_tests.diebold_mariano` (Diebold–Mariano + Harvey–Leybourne–Newbold
  small-sample correction) → ไฟล์เดียวกับด้านบน · อ่านผลสรุปที่
  `results/feedback/WHY_NAIVE_WINS.md`

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
ใช้ `python tools/append_investing.py <ไฟล์ทุกหุ้น>` (ตรวจ + แปลง + append + เก็บต้นฉบับ + check_raw_update ในคำสั่งเดียว ดู `tools/README.md`)

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
