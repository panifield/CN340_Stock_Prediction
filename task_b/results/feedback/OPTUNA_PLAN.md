# OPTUNA_PLAN — แผนค้น hyperparameter + feature group ประกาศก่อนรัน (feedback ข้อ 3)

> **เขียนไฟล์นี้ก่อนรัน `optuna_features.py` ครั้งแรก** · SHA-256 ของไฟล์นี้ถูกคำนวณและ
> บันทึกลงคอลัมน์ `plan_sha256` ของทุกแถวใน `optuna_trials.csv` / `optuna_summary.csv`
> **ห้ามแก้ไฟล์นี้หลังรัน** ถ้าต้องแก้ ให้เขียน `OPTUNA_PLAN_R2.md` ใหม่ และระบุในรายงานว่า
> เป็นรอบที่สอง
>
> รอบนี้ = **รอบที่ 1** · วันที่เขียนแผน: 2026-10-24

---

## 0. คำถามที่ตอบ

feedback ข้อ 3 (14 ก.ย. 2026) ขอให้ใช้ **HyperOpt หา feature** · ตอนนี้ `tune.py` เป็น
grid search **hyperparameter อย่างเดียว** และ feature fix ไว้ 29 ตัวตายตัว ไม่มีกลไก
เปิด/ปิด feature เลย

**คำถาม:** ถ้าปล่อยให้ค้นทั้ง feature group และ hyperparameter พร้อมกันด้วยงบที่กำหนด
จะได้ config ที่ดีกว่า Phase 1C จริงไหม หรือเป็นแค่ผลของ selection bias

---

## 1. ข้อมูลและการแบ่ง (จุดที่สำคัญที่สุดของแผนนี้)

- ข้อมูลตัดที่ `SPLIT_BY_DATE["val_end"] = 2025-02-25` **ก่อนสร้าง feature** แล้ว `assert`
  เหมือน `tune.load_dev_data()` → **ไม่มีทางแตะ test** · **ไม่แตะ fixed val ของ `main.py`**
- fold ชุดเดียวกับ `tune.py` (`walk_forward_splits(n_splits=3, min_train=1250)`)
- **Optuna เห็น fold 1–2 เท่านั้น** → objective = `mean MAE_return` ของ
  **4 evaluation** (2 หุ้น × 2 folds) · primary metric เดียวกับ `tune.py`
- **fold 3 = confirmation** Optuna ไม่เคยเห็น → ประเมินเฉพาะ **หลังจบการค้น** สำหรับ:
  1. ผู้ชนะของแต่ละโมเดล (ANN refit ด้วย **10 seeds**)
  2. config **Phase 1C** (29 feature เดิม = ทุกกลุ่มยกเว้น `cross_stock`)
  3. `Baseline: Naive (RW)`

> **ข้อจำกัดที่ต้องเขียนในรายงาน:** fold 3 **ไม่ใช่ pristine** — มันทับช่วงที่ `tune.py`
> ใช้เลือก config Phase 1C มาก่อน (`TUNING_PLAN.md` ใช้ทั้ง 3 folds) · fold 3 จึงเป็นแค่
> **"ชุดที่ไม่ถูกใช้ในการค้นรอบนี้"** ไม่ใช่ชุดที่ไม่มีใครเคยเห็น · ความเปรียบจึง
> **เอนเข้าข้าง Phase 1C** บน fold 3 และ **เอนเข้าข้างผู้ชนะ Optuna** บน fold 1–2

---

## 2. Feature groups (เปิด/ปิดเป็นกลุ่ม · ต้องเปิดอย่างน้อย 1 กลุ่ม)

นิยามอยู่ใน `features_experimental.GROUPS` · **8 กลุ่ม = 33 คอลัมน์**

| กลุ่ม | คอลัมน์ | n |
|---|---|---|
| `returns` | `ret_1d_prev`, `ret_2d_prev`, `ret_3d_prev`, `ret_5d_prev`, `ret_10d_prev` | 5 |
| `candle` | `hl_range_prev`, `oc_change_prev`, `close_pos_in_range_prev` | 3 |
| `trend` | `close_over_sma5/10/20_prev`, `close_over_ema5/10/20_prev`, `sma5_over_sma20_prev` | 7 |
| `volatility` | `volatility_5d_prev`, `volatility_20d_prev` | 2 |
| `momentum` | `rsi_prev`, `macd_prev`, `macd_signal_prev`, `macd_hist_prev`, `bb_position_prev` | 5 |
| `volume` | `volume_change_prev`, `volume_over_ma20_prev` | 2 |
| `calendar` | `dow_0` … `dow_4` | 5 |
| **`cross_stock`** (ใหม่) | `x_ret_1d_prev`, `x_ret_2d_prev`, `x_ret_5d_prev`, `x_volatility_5d_prev` **ของหุ้นอีกตัว** | 4 |

- 7 กลุ่มแรกรวมกัน = **29 feature เดิมเป๊ะ** (`assert_groups_cover_base_features` บังคับ)
- `cross_stock` อยู่ในไฟล์ใหม่ `features_experimental.py` · **ห้ามแก้ `features.py`**
- `cross_stock` ต้อง `shift(1)` บนปฏิทินของหุ้นคู่เอง แล้ว align ด้วยวันที่ (**inner join**)
  → ผ่าน `verify_no_leak_experimental()` และ `tests/test_features_experimental.py`
  (รวม negative control: ถ้าลืม shift ตัวตรวจต้องจับได้)
- **ห้ามดึงข้อมูลจากอินเทอร์เน็ตเพิ่มในรอบนี้** (SET index, อัตราดอกเบี้ย, ข่าว ฯลฯ)
  → เขียนเป็นงานต่อในรายงาน
- ลำดับคอลัมน์คงที่ตาม `GROUP_NAMES` เสมอ เพราะ `max_features` / `colsample_bytree`
  สุ่ม feature subset ตามลำดับคอลัมน์ → ลำดับเปลี่ยน = รันซ้ำไม่ได้ผลเดิม

---

## 3. Search space ต่อโมเดล

Sampler: `optuna.samplers.TPESampler(seed=0)` (TPE — R13) · **ไม่ใช้ pruner**
(ทุก trial ต้องถูกประเมินครบ 4 evaluation เพื่อให้คะแนนเทียบกันได้)
ทุก trial มี 8 พารามิเตอร์ bool `use_<group>` + hyperparameter ข้างล่าง

ช่วงของ hyperparameter **ครอบ grid ของ `TUNING_PLAN.md` และกว้างกว่าทั้งสองฝั่ง**:

### ANN (MLP) — 40 trials
| พารามิเตอร์ | ช่วง | grid เดิม (TUNING_PLAN) |
|---|---|---|
| `alpha` | log-uniform **0.1 – 50** | `[0.3, 1.0, 3.0, 7.0]` — ผู้ชนะ `7.0` **อยู่ที่ขอบ grid** จึงต้องขยายขึ้นไป |
| `hidden_layer_sizes` | `{(4,2), (8,4), (16,8), (32,16)}` | `[(8,4), (16,8), (32,16)]` — เพิ่ม `(4,2)` ที่เล็กกว่าเดิม |
| ค่าอื่น | คงที่ตาม `ANN_PARAMS` (adam, relu, `max_iter=3000`, `early_stopping=False`) | เดิม |

ANN ใช้ **3 seeds ระหว่างค้น** (เหมือน `tune.TUNING_ANN_SEEDS`) แล้ว **refit ผู้ชนะด้วย 10 seeds**
บน fold 3 · ตัวเลข 10 seeds ใช้รายงานเท่านั้น ห้ามใช้เลือกใหม่

### Random Forest — 60 trials
| พารามิเตอร์ | ช่วง | grid เดิม |
|---|---|---|
| `max_depth` | int **2 – 12** | `[4, 8, 12]` |
| `min_samples_leaf` | int **10 – 100** | `[20, 50]` |
| `max_features` | `{0.3, 0.5, 0.7, 1.0}` | `[1.0, 0.5]` |
| `n_estimators` | คงที่ 400 (`RF_PARAMS`) | เดิม |

### XGBoost — 60 trials
| พารามิเตอร์ | ช่วง | grid เดิม |
|---|---|---|
| `max_depth` | int **1 – 5** | `[2, 3, 4]` |
| `learning_rate` | log-uniform **0.003 – 0.1** | `[0.01, 0.05]` |
| `subsample` | uniform **0.6 – 1.0** | `[0.8, 1.0]` |
| `n_estimators` | int **100 – 600** (step 50) | คงที่ 300 |
| ค่าอื่น | คงที่ตาม `XGB_PARAMS` (`colsample_bytree=0.8`, `reg_lambda=1.0`) | เดิม |

---

## 4. งบตายตัว (ประกาศไว้ ห้ามเพิ่มหลังเห็นผล)

| โมเดล | trials |
|---|---|
| Random Forest | **60** |
| XGBoost | **60** |
| ANN (MLP) | **40** |
| **รวม** | **160** |

`TPESampler(seed=0)` · trial รันทีละตัวตามลำดับ (TPE เป็นลำดับ — รันขนานจะเปลี่ยนลำดับ
การเสนอค่าและรันซ้ำไม่ได้ผลเดิม)

---

## 5. Guard และกฎการให้คะแนน

- **validity guard** = `StdRatio ≥ 0.05` **ทุก evaluation** (ค่าเดียวกับ `TUNING_PLAN.md`)
- ไม่ผ่าน guard → objective คืน **ค่า penalty** = `10.0` (ใหญ่กว่า MAE_return ที่เป็นไปได้
  ทุกค่าหลายพันเท่า) และบันทึก `passed_guard = False` ลง `optuna_trials.csv`
  (ไม่ลบ trial ทิ้ง — ต้องรายงานว่า guard ตัดอะไรไปบ้าง)
- **ผู้ชนะ** = trial ที่ `passed_guard = True` และ objective ต่ำสุด
  (ถ้าไม่มี trial ใดผ่าน guard → **หยุด** และรายงานว่าโมเดลนั้นยุบเป็นค่าคงที่ทุกค่า
  ห้ามลดเกณฑ์หลังเห็นผล)
- ไม่มีกฎเสมอ — TPE ให้ค่าต่อเนื่อง โอกาสเสมอเป๊ะเป็นศูนย์ในทางปฏิบัติ
  (ถ้าเสมอจริง เลือก trial ที่ `number` น้อยกว่า)

---

## 6. การขนานและ reproducibility

- **ประเมิน 4 evaluation ของแต่ละ trial พร้อมกัน** ด้วย `joblib.Parallel`
  (`backend="loky"`, `n_jobs = min(4, max(1, os.cpu_count() - 2))`)
- **ทุกโมเดล `n_jobs = 1`** (RF / XGB บังคับจาก `RF_PARAMS` / `XGB_PARAMS`)
- `joblib.Parallel` คืนผลตามลำดับที่ส่งเข้าไปเสมอ → ค่าเฉลี่ยของ 4 evals deterministic
- บันทึก trial ลง `results/feedback/optuna_<model>.db` (sqlite) เพื่อ resume ได้
  **ห้าม resume ข้ามการแก้โค้ด** — ถ้าแก้โค้ดต้องลบ `.db` แล้วเริ่มใหม่ทั้งหมด
  และระบุในรายงาน · ไฟล์ `.db` อยู่ใน `.gitignore` (ไม่ commit)
- ก่อนรันเต็ม: **smoke run 3 trials/โมเดล** ลงโฟลเดอร์ชั่วคราวเพื่อจับเวลาต่อ trial
  แล้วประมาณเวลารวม · ถ้าประมาณ **เกิน 60 นาที** → หยุดแล้วถามผู้ใช้ก่อน (เงื่อนไขหยุดข้อ 9.3)

---

## 7. รายงานที่จะเขียน (ทั้งหมดใน `results/feedback/`)

1. **`optuna_trials.csv`** — ทุก trial ทุกโมเดล: `number`, พารามิเตอร์ทุกตัว, กลุ่มที่เปิด,
   `MAE_return` ต่อ eval (4 คอลัมน์), objective, `StdRatio` ต่อ eval, `min_StdRatio`,
   `passed_guard`, `seconds`
2. **`optuna_summary.csv`** — ผู้ชนะต่อโมเดล vs Phase 1C **บน fold 1–2 และบน fold 3**
   · `relMAE` · DM vs Naive และ vs Phase 1C **บน fold 3**
3. **feature group importance** (3 มุมมองพร้อมกัน เพราะมุมเดียวเชื่อไม่ได้):
   - ความถี่ที่แต่ละกลุ่มถูกเปิดใน **top-10 trials** (เทียบ baseline การสุ่ม = 5/10)
   - `optuna.importance.get_param_importances` (fANOVA)
   - **ablation ของผู้ชนะ**: ปิดทีละกลุ่ม ประเมินบน fold 1–2
4. **เทียบกับ null**: รัน config ผู้ชนะ **1 ตัว** ด้วย `y` ที่สลับแบบสุ่ม **ใน train เท่านั้น**
   (`np.random.default_rng(seed).permutation`, **3 permutations**, seed 0/1/2) ·
   `y` ของ eval ไม่ถูกสลับ · ดูว่าส่วนต่างระหว่างผู้ชนะกับ Phase 1C **ใหญ่กว่าที่ noise ทำได้ไหม**
5. **รูป** (`optuna_history.png`, `optuna_group_frequency.png`, `optuna_fold3.png`):
   optimization history · ความถี่ของ feature group · ผู้ชนะ vs Phase 1C บน fold 3
6. **`OPTUNA_RESULTS.md`** — ภาษาง่าย + ตรวจคำทำนายข้อ 8 ทีละข้อ
   + **ย้ำว่าไม่ได้นำผู้ชนะไปใช้กับโมเดล live** และถ้าจะนำไปใช้ต้องเป็นรอบใหม่ที่ผู้ใช้ตัดสินใจ

**ไม่ว่าผลจะเป็นอย่างไร ห้ามนำผู้ชนะไปใส่ `config.py` / `features.py`** (กติกาข้อ 1.5
ของงานรอบนี้) · การเปลี่ยนโมเดลที่ใช้ทำนายจริงต้องเป็นรอบใหม่ มี `CONFIG_TAG` ใหม่
และผู้ใช้เป็นคนตัดสินใจ

---

## 8. คำทำนายล่วงหน้า (ตรวจอัตโนมัติในรายงาน)

| # | คำทำนาย |
|---|---|
| 1 | ผู้ชนะ Optuna ได้ objective (mean MAE fold 1–2) **ต่ำกว่า** Phase 1C **ทุกโมเดล** — เพราะค้นบน fold 1–2 โดยตรง (selection bias) ไม่ใช่เพราะดีกว่าจริง |
| 2 | ส่วนต่างบน **fold 3** (ผู้ชนะ − Phase 1C) **เล็กกว่า** ส่วนต่างบน fold 1–2 ทุกโมเดล (shrinkage) |
| 3 | **ไม่มีโมเดลใด** ที่ผู้ชนะ Optuna ชนะ Naive **อย่างมีนัยสำคัญ** บน fold 3 (DM p one-sided ≥ 0.05 ทั้งสองหุ้น) |
| 4 | กลุ่ม `cross_stock` ถูกเปิดใน **≤ 7 จาก 10** ของ top-10 trials ทุกโมเดล (= ไม่ชัดกว่าการสุ่มมาก) |
| 5 | **ไม่มีกลุ่มใด** ถูกเปิดครบ **10/10** ใน top-10 ของทั้งสามโมเดลพร้อมกัน (= ไม่มีกลุ่มที่ "จำเป็น" ชัดเจน) |
| 6 | ANN ผู้ชนะมี `alpha > 7.0` (เกินขอบ grid เดิม — ซึ่งเป็นเหตุผลที่ต้องขยายช่วง) |
| 7 | **ablation**: ปิดกลุ่มใดกลุ่มหนึ่งของผู้ชนะ ทำให้ objective แย่ลง **ไม่เกิน 3%** ทุกกลุ่ม ทุกโมเดล (= ไม่มีกลุ่มไหนจำเป็นจริง) |
| 8 | **null test**: `|ผู้ชนะ − Phase 1C|` บน fold 3 **เล็กกว่าหรือเท่ากับ** ช่วงที่ 3 permutations ทำได้ (= ส่วนต่างอยู่ในระดับ noise) |
| 9 | มี **≥ 1** trial ที่ไม่ผ่าน guard `StdRatio ≥ 0.05` แต่ **< 25%** ของ trials ในทุกโมเดล |
