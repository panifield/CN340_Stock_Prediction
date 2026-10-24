# FINETUNE_PLAN — แผน pretrain + fine-tune ประกาศก่อนรัน (feedback ข้อ 7)

> **เขียนไฟล์นี้ก่อนรัน `finetune_experiment.py` ครั้งแรก** · SHA-256 ของไฟล์นี้ถูกคำนวณและ
> บันทึกลงคอลัมน์ `plan_sha256` ของทุกแถวใน `finetune_results.csv` / `finetune_summary.csv`
> **ห้ามแก้ไฟล์นี้หลังรัน** ถ้าต้องแก้ ให้เขียน `FINETUNE_PLAN_R2.md` ใหม่ และระบุในรายงานว่า
> เป็นรอบที่สอง
>
> รอบนี้ = **รอบที่ 1** · วันที่เขียนแผน: 2026-10-24

---

## 0. คำถามที่ตอบ

feedback ข้อ 7 ขอ "pretrain ด้วยข้อมูลเก่า แล้ว fine-tune ด้วยข้อมูลใหม่"
ตอนนี้ repo มีแค่ **recency weighting** (`weighting.py` — ถ่วงน้ำหนักครั้งเดียว) ซึ่ง
`weighting.py` docstring เขียนไว้ตรง ๆ ว่า *เลือกวิธีถ่วงน้ำหนัก ไม่ใช่ pre-train แล้ว fine-tune
เพราะแบบหลังเพิ่ม hyperparameter และเสี่ยงให้โมเดลลืมข้อมูลเก่า* — คือ **เหตุผลเชิงทฤษฎี
ที่ยังไม่เคยทดลองจริง** · รอบนี้ทดลองจริง

**คำถามเดียว:** การเทรน 2 ขั้น (old → recent) ให้ MAE_return ต่ำกว่าการเทรนรวมครั้งเดียวไหม

---

## 1. ข้อมูลและการแบ่ง

- ข้อมูลตัดที่ `SPLIT_BY_DATE["val_end"] = 2025-02-25` **ก่อนสร้าง feature**
  โดยใช้ `tune.load_dev_data()` ตัวเดียวกับ `tune.py` (มี `assert` ของตัวเองว่าไม่เลย val_end)
  → **ไม่มีทางแตะ test** · ไม่แตะ fixed val ของ `main.py` ด้วย
- walk-forward **3 folds × 2 หุ้น = 6 evaluation** · fold ชุดเดียวกับ `tune.py`
  (`walk_forward_splits(n_splits=3, min_train=1250)`)
- ภายใน **train ของแต่ละ fold** แบ่งเป็น:
  - `recent` = **R แถวท้าย** ของ train (ใหม่สุด)
  - `old` = ทุกแถวที่เหลือ
  - **R ∈ {250, 500}** (วันทำการ ≈ 1 ปี และ 2 ปี)
- `old` และ `recent` ไม่ซ้อนทับกัน และ `old` อยู่ก่อน `recent` ทางเวลาทั้งหมด (จะ `assert`)
- ห้าม shuffle ข้ามเวลาในทุกขั้น

### Scaler — จุดที่ leak ได้ง่ายที่สุด

`StandardScaler` ของ **X และของ y** fit จาก **train ทั้ง fold (old + recent) ครั้งเดียว**
แล้วใช้ค่าเดียวกันกับทุกขั้นและทุก arm ของ fold นั้น

เหตุผล: ถ้า fit scaler แยกต่อขั้น (old ใช้ scaler ของ old, recent ใช้ของ recent)
ความต่างระหว่าง arm จะมาจาก scaler ไม่ใช่จากกลยุทธ์การเทรน ·
scaler เห็นแต่ข้อมูล train ของ fold นั้น **ไม่เคยเห็นแถว eval** → ไม่ leak

---

## 2. กลยุทธ์ที่จะลอง (ทุกตัวใช้ hyperparameter Phase 1C จาก `config.py` ตรง ๆ)

| arm | ทำอะไร | configs ต่อโมเดล |
|---|---|---|
| `pooled` | เทรนบน train ทั้ง fold ครั้งเดียว = **แบบปัจจุบัน** | 1 (**ตัวเทียบหลัก**) |
| `old_only` | เทรนบน `old` เท่านั้น | 2 (R = 250, 500) |
| `recent_only` | เทรนบน `recent` เท่านั้น | 2 (R = 250, 500) |
| `pretrain_finetune` | เทรนบน `old` แล้วเทรนต่อบน `recent` (รายละเอียดต่อโมเดลข้างล่าง) | ANN 8 · RF 4 · XGB 4 |
| `recency_weighted` | เทรนครั้งเดียวบน train ทั้ง fold โดยถ่วงน้ำหนักด้วย half-life ที่ชนะ stage 1 | 1 |

### 2.1 `pretrain_finetune` ต่อโมเดล

**ANN (MLP)** — fine-tune จริง
1. fit บน `old` ตามปกติ (`max_iter = 3000`, ค่าอื่นจาก `ANN_PARAMS`)
2. เรียก `partial_fit(recent)` ต่อ **E ครั้ง** (MLPRegressor ทำ 1 iteration ต่อการเรียก)
   - **E ∈ {50, 200}**
3. learning rate ตอน fine-tune: **×1 (เท่าเดิม 1e-3)** หรือ **×0.1 (1e-4)**
   (ตั้งผ่าน `learning_rate_init` ของทั้ง estimator และ `_optimizer` เพราะ Adam
   คำนวณ `learning_rate` ใหม่จาก `learning_rate_init` ทุก iteration)
4. ทำซ้ำ **10 seeds** (`ANN_SEEDS`) แล้วเฉลี่ยการทำนาย เหมือนนิยามโมเดลเดิม
→ 2 (R) × 2 (E) × 2 (lr) = **8 configs**

**XGBoost** — fine-tune จริง
1. train บน `old` ด้วย `n_estimators = 300` (ค่า Phase 1C)
2. เทรนต่อบน `recent` อีก **K rounds** ด้วย `xgb_model=<booster ของขั้น 1>`
   - **K ∈ {50, 150}**
→ 2 (R) × 2 (K) = **4 configs**

**Random Forest** — **ไม่ใช่ fine-tune จริง** (ประกาศไว้ชัดเจน)
RF ปรับน้ำหนักต้นไม้ที่สร้างแล้วไม่ได้ จึงใช้ **analogue**: `warm_start=True` แล้ว **เพิ่มต้นไม้
ใหม่ K ต้นที่เทรนบน `recent`** ต่อจาก 400 ต้นเดิมที่เทรนบน `old`
- **K ∈ {100, 400}**
- การทำนายคือค่าเฉลี่ยของต้นไม้ทั้ง `400 + K` ต้น → เป็น **การผสมแบบถ่วงน้ำหนักคงที่**
  ระหว่างโมเดล old กับโมเดล recent (น้ำหนัก `400/(400+K)` : `K/(400+K)`)
  **ไม่ใช่การปรับพารามิเตอร์เดิมด้วยข้อมูลใหม่** ซึ่งเป็นความหมายของคำว่า fine-tune
→ 2 (R) × 2 (K) = **4 configs**

### 2.2 `recency_weighted`

อ่าน half-life ที่ชนะ stage 1 **จากไฟล์ที่มีอยู่** (`results/tuning_scores.csv` แถว
`stage = 1` และ `winner = True` ซึ่งเป็นสรุปของ `results/tuning_stage1_halflife.csv`)
**ไม่จูนใหม่** · ถ้าค่าที่อ่านได้เป็น `None` arm นี้จะเหมือน `pooled` ทุกประการ
→ รายงานตรง ๆ ว่า arm นี้ **ว่างเปล่า** และเพิ่ม arm ข้อมูลประกอบ
`recency_weighted_best_finite` = half-life **ที่ไม่ใช่ None ที่ได้คะแนนดีที่สุด** ใน stage 1
(ยังอ่านจากไฟล์ ไม่จูนใหม่) เพื่อให้มีตัวเทียบที่ถ่วงน้ำหนักจริงอยู่ในตาราง
— ประกาศไว้ตรงนี้ก่อนรัน และติดป้ายในรายงานว่าเป็น **arm ข้อมูลประกอบ ไม่ใช่ผู้เข้าแข่งขัน**

### 2.3 Baseline ในตาราง

ทุก evaluation มีแถว `naive` (`baselines.baseline_naive_return` → return = 0)
เพื่อให้เทียบ `relMAE` ได้ตรง ๆ **ห้ามเพิ่ม baseline อื่น**

---

## 3. Metric และ guard

- **primary** = `mean MAE_return` ของ **6 evaluation** (เหมือน `tune.py` เป๊ะ)
- **secondary (รายงานเท่านั้น ไม่ใช้ตัดสิน)** = `mean relMAE` (= MAE/MAE_naive ต่อ eval
  แล้วเฉลี่ย), `min StdRatio`, DM vs `pooled`, DM vs `naive`
- **validity guard** = `StdRatio ≥ 0.05` **ทุก evaluation** (ค่าเดียวกับ `TUNING_PLAN.md`)
  ไม่ผ่าน = บันทึก `passed_guard = False` และ **ตัดออกจากการตัดสิน** (ไม่ลบแถว)
- DM test: `stats_tests.diebold_mariano`, `loss = "abs"`, `h = 1`, HLN correction,
  α = **0.05 one-sided** · ไม่ใช้ p-value จัดอันดับ ใช้แค่ตัดสินว่า "ต่างจริงไหม"

---

## 4. กฎตัดสิน (ประกาศล่วงหน้า ห้ามเปลี่ยนหลังเห็นผล)

สรุปว่า **"fine-tune ช่วย"** เฉพาะเมื่อมี config ของ `pretrain_finetune` อย่างน้อยหนึ่งตัว
ที่ผ่าน **ทั้งสามข้อ**:

1. `mean MAE_return` **ต่ำกว่า** `pooled` ของโมเดลเดียวกัน
2. DM vs `pooled`: `p_one_sided_model_better < 0.05` ใน **≥ 4 จาก 6 evaluation**
3. `passed_guard = True` (StdRatio ≥ 0.05 ทุก evaluation)

ถ้าไม่มี config ใดผ่านทั้งสามข้อ → สรุปว่า **"ไม่ช่วยภายใต้ protocol นี้"**

**ไม่ว่าผลจะเป็นอย่างไร ห้ามนำผู้ชนะไปใส่ `config.py`** — รอบนี้เป็นผลรายงานเท่านั้น
(กติกาข้อ 1.5 ของงานรอบนี้) การเปลี่ยนโมเดลที่ใช้ทำนายจริงต้องเป็นรอบใหม่ที่ผู้ใช้ตัดสินใจ
และต้องมี `CONFIG_TAG` ใหม่

---

## 5. คำทำนายล่วงหน้า (ตรวจอัตโนมัติในรายงาน)

| # | คำทำนาย |
|---|---|
| 1 | **ไม่มี** config `pretrain_finetune` ใดผ่านกฎตัดสินข้อ 4 ในโมเดลใดเลย |
| 2 | `recent_only` แย่กว่า `pooled` **ทุกโมเดล ทุกค่า R** (ข้อมูลน้อยลง 3–6 เท่า) |
| 3 | `old_only` อยู่ใกล้ `pooled` มาก (`mean MAE_return` ต่างกัน < 2%) ทุกโมเดล เพราะ stage 1 เลือก half-life = None แปลว่าแถวเก่ายังมีประโยชน์ |
| 4 | **ทุก arm** ยังมี `mean relMAE > 1` (แพ้ Naive) และ **ไม่มี arm ใด** ชนะ Naive อย่างมีนัยสำคัญเกิน 1 ใน 6 evaluation |
| 5 | ANN ไวต่อ fine-tune มากที่สุด (ช่วง `mean MAE_return` ระหว่าง config ของ `pretrain_finetune` กว้างกว่า RF และ XGB) |
| 6 | `E = 200` ให้ `mean MAE_return` **สูงกว่าหรือเท่ากับ** `E = 50` ที่ R และ lr เดียวกัน (fine-tune นานขึ้น = ลืมข้อมูลเก่ามากขึ้น) อย่างน้อย 3 ใน 4 คู่ |
| 7 | lr `×0.1` ให้ `mean MAE_return` **ต่ำกว่าหรือเท่ากับ** lr `×1` ที่ R และ E เดียวกัน อย่างน้อย 3 ใน 4 คู่ |

---

## 6. Reproducibility

- ทุกโมเดล `n_jobs = 1` · seed ตายตัว (`RANDOM_STATE = 42`, `ANN_SEEDS = range(10)`)
- รันทีละ config ตามลำดับ ไม่ขนาน (ความเร็วไม่ใช่ข้อจำกัดของงานนี้)
- **ตรวจความถูกต้องของเส้นทาง manual scaling**: arm `pooled` ถูก implement ด้วย
  scaler ที่เขียนเอง (เพราะ `partial_fit` ต้องเข้าถึงข้อมูลที่ scale แล้วตรง ๆ)
  จึงต้องยืนยันว่าให้ผลเท่ากับเส้นทาง `Pipeline` + `TransformedTargetRegressor` ของ
  `tune.make_model()` — เทียบ `MAE_return` ของ `pooled` กับเส้นทาง Pipeline
  ทุก evaluation ด้วย `rtol = 1e-6` และบันทึกผลลงคอลัมน์ `pooled_matches_pipeline`
  ถ้าไม่ตรง = **หยุด** (แปลว่าเส้นทาง manual ไม่เทียบเท่า ผลทั้งตารางเชื่อไม่ได้)

---

## 7. ผลลัพธ์ที่จะเขียน

- `results/feedback/finetune_results.csv` — หนึ่งแถวต่อ (arm config × evaluation)
- `results/feedback/finetune_summary.csv` — หนึ่งแถวต่อ arm config
- `results/feedback/finetune.png` — รูปเทียบ arm
- `results/feedback/FINETUNE_RESULTS.md` — สรุป + ตรวจคำทำนายข้อ 5 ทีละข้อ
  + อ้างผลของ task_a (`task_a/results/five_analyses/4_pretrain_finetune.csv` — อ่านอย่างเดียว)
