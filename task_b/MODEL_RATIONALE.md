# MODEL_RATIONALE.md — เหตุผลที่เลือกโมเดลชุดนี้ (feedback ข้อ 11)

> **ไฟล์นี้ทำอะไร / ตอบ feedback ข้อไหน**
> feedback ข้อ 11 (14 ก.ย. 2026) ขอเหตุผลว่าทำไมเลือกโมเดลเหล่านี้ แต่ละตัวแก้ปัญหาอะไร
> และค่าพารามิเตอร์มาจากไหน · `task_a` มีไฟล์แบบนี้อยู่แล้ว (`task_a/results/model_rationale.md`)
> แต่ task_b ยังไม่มี — เป็นงาน regression คนละแบบ จึงเขียนใหม่ทั้งหมด
>
> **ตัวเลขทุกตัวมาจากไฟล์ผลจริง** ไม่มีเลขจากความจำ:
> `results/<TICKER>_taskB_val.csv` · `_val_shape.csv` · `_val_stats.csv` · `_val_costs.csv` ·
> `results/tuning_scores.csv` · `results/tuning_summary.md` · `results/split_profile.csv` ·
> `results/feedback/walkforward_summary.csv`
> reference ที่อ้าง (R1, R3, R4, R5, R6, R8, R9, R11, R13, R14) ดู [`REFERENCES.md`](REFERENCES.md)

---

## 1. การตัดสินใจที่สำคัญที่สุด: ทำนาย **return** ไม่ใช่ราคาดิบ

นี่ไม่ใช่เรื่องรายละเอียด แต่เป็นสิ่งที่กำหนดหน้าตาของผลทั้งหมด (เหตุผลเต็มอยู่ใน
`targets.py` docstring และหัวไฟล์ `features.py`) · สามเหตุผล:

**(1) ราคาดิบเป็น non-stationary · return เป็น stationary**
ราคามี trend ระยะยาว (KBANK 70.75 → 257.00 บาท · ADVANC 142.00 → 404.00 บาท
ในช่วงข้อมูลทั้งหมด — `results/split_profile.csv` คอลัมน์ `price_min` / `price_max`)
โมเดลที่เรียนจากช่วงราคาหนึ่งจะใช้กับอีกช่วงไม่ได้

**(2) tree-based model extrapolate นอก range ที่เห็นตอน train ไม่ได้**
Random Forest ถูกจำกัดด้วยค่าเฉลี่ยใน leaf โดยธรรมชาติ ส่วน boosted tree แม้ไม่ถูก bound
แบบเดียวกัน (เป็นผลรวมของหลายต้น) แต่ยังมีข้อจำกัดเหมือนกัน ·
`StandardScaler` แก้ไม่ได้เพราะเป็นการแปลงเชิงเส้น — ลำดับของค่ายังเหมือนเดิม
นี่เป็นเหตุผลเดียวกับที่ `features.py` **ไม่มี feature ราคาดิบเลย** (ไม่มี `close_prev`,
`open_prev`, …) ทุก feature เป็น return / อัตราส่วน / ตัวเลข 0–100

**(3) ถ้าทำนายราคาดิบจะได้ R² ~0.98–0.99 ซึ่งหลอก**
เพราะโมเดลแค่เรียนรู้ว่า "พรุ่งนี้ ≈ วันนี้" · `evaluate.py` จึงมีกฎเขียนกำกับไว้ว่า
**ห้ามเพิ่ม R² ที่คำนวณบนราคา (`R2_price`)** เข้าตารางผล — ค่านั้นวัดข้อมูล ไม่ได้วัดโมเดล

พอทำนาย return เสร็จค่อยแปลงกลับเป็นราคาด้วย `Close_pred(t) = Close(t−1) × (1 + ret_pred)`
ซึ่ง `Close(t−1)` รู้อยู่แล้วตอนทำนาย → ไม่ leak · `MAE_baht` ในตารางหลักมาจากทางนี้
และเป็น **secondary interpretability metric** เท่านั้น ไม่ใช้เลือกโมเดล

> **ราคาที่ต้องจ่าย:** การทำนาย return ทำให้งานยากขึ้นมาก จาก R² ~0.98 (หลอก)
> เหลือ `R2_return` −0.003 ถึง +0.024 (จริง — `results/*_taskB_val.csv`)
> นี่คือตัวเลขที่ซื่อสัตย์ และเป็นเหตุผลที่ผลออกมา "ไม่หวือหวา"

---

## 2. Baseline — ส่วนที่สำคัญที่สุดของโปรเจกต์

| baseline | ทำนายอะไร | ทำหน้าที่อะไร | ไฟล์ |
|---|---|---|---|
| **Naive (RW)** | `return = 0` ทุกวัน (พรุ่งนี้ = วันนี้) | **คู่แข่งตัวจริงตัวเดียว** = random walk ตามนิยาม (R1, R2) | `baselines.baseline_naive_return` |
| Mean Return | `return = mean(y_train)` | จับว่า drift ระยะยาวมีค่าพอทำนายไหม + เป็น sanity check ว่าเครื่องมือทดสอบไวจริง | `baselines.baseline_mean_return` |

**ทำไม Naive คือคู่แข่งที่ถูกต้อง** — EMH แบบ weak form (Fama 1970, R1) บอกว่าข้อมูลราคา
ในอดีตเพียงอย่างเดียวไม่ควรทำนายราคาในอนาคตได้พอจะทำกำไร · ถ้ารายงานบอกว่า
"XGBoost ได้ MAE 0.00942 ดีที่สุด" แต่ไม่บอกว่า Naive ได้ 0.00944 อาจารย์ถามคำถามเดียวก็จบ
→ `evaluate.results_table` จึงบังคับให้ **baseline อยู่ในตารางเดียวกับโมเดล** ทุกตาราง
และ `compare_to_baseline()` เทียบกับ Naive ของชุดเดียวกันเสมอ

**Mean Return เป็น positive control ที่ใช้ได้จริง** — มันแพ้ Naive **อย่างมีนัยสำคัญ**
ทั้งสองหุ้น (KBANK DM = +3.52, p two-sided = 0.00048 · ADVANC DM = +2.65,
p two-sided = 0.0084 — `results/<TICKER>_taskB_val_stats.csv`) → พิสูจน์ว่า DM test
ที่เราใช้ **จับความต่างได้เมื่อมีความต่างจริง** ไม่ใช่เครื่องมือที่ตอบ "ไม่ต่าง" ทุกครั้ง

**จุดอ่อนของ Naive** — ไม่มีทางใช้ตัดสินใจซื้อขาย (ไม่เคยบอกทิศ) และได้เปรียบ MAE
แบบ "ฟรี" จากวันราคานิ่ง 11–19% ของวัน (ดู `results/feedback/WHY_NAIVE_WINS.md`)
การที่มันชนะจึงไม่ใช่หลักฐานว่า "ML ไร้ประโยชน์" แต่คือหลักฐานว่า **สัญญาณอ่อน**

---

## 3. ทำไม 3 ตระกูลนี้ (และแต่ละตัวแก้ปัญหาอะไร)

เลือก **3 แนวทางที่จับ non-linearity คนละกลไก** เพื่อให้คำตอบ "ไม่มีสัญญาณ" น่าเชื่อถือ:
ถ้าสามกลไกที่ไม่เหมือนกันเลยให้ผลเท่ากันหมด แปลว่าข้อจำกัดอยู่ที่ข้อมูล ไม่ใช่ที่การเลือกโมเดล

### 3.1 ANN (MLP) — neural / gradient-based

| | |
|---|---|
| **ตอบโจทย์อะไร** | เป็นโมเดลที่ยืดหยุ่นที่สุดในชุด ประมาณฟังก์ชันไม่เชิงเส้นได้แทบทุกรูปแบบ ถ้ามีสัญญาณที่ tree มองไม่เห็น (เช่น ผลรวมเชิงเส้นของหลาย feature ที่ผ่าน non-linearity) ANN ควรจับได้ |
| **แก้ปัญหาอะไร** | tree แบ่งพื้นที่เป็นกล่องแกนขนาน (axis-aligned) · ANN สร้างขอบเขตเฉียงและผสม feature ได้ตรง ๆ |
| **อ้างอิง** | backpropagation: Rumelhart, Hinton & Williams 1986 (**R3**) · implementation: `sklearn.neural_network.MLPRegressor` (**R7**) |
| **จุดอ่อน / สิ่งที่ต้องคุม** | (ก) **capacity ล้นง่ายที่สุด** — v1 ใช้ `(64, 32)` + `alpha=1e-3` ได้ `R2_return` val = **−2.39** (KBANK) / **−2.37** (ADVANC) = overfit หนัก (บันทึกไว้ใน `config.py`) · (ข) **ผลขึ้นกับ random seed** — รายงาน seed เดียวคือเลือกผลที่ถูกใจได้ · (ค) ต้อง standardize ไม่งั้นไม่ converge |
| **สิ่งที่ทำเพื่อคุม** | `(8, 4)` + `alpha = 7.0` (L2 แรงกว่า default ของ sklearn 70,000 เท่า) · **seed averaging 10 seeds** (`ANN_SEEDS = range(10)`) ซึ่งนับเป็นส่วนหนึ่งของ "นิยามโมเดล" ไม่ใช่การจูน · ปิด `early_stopping` เพราะ `MLPRegressor` จะ **shuffle** แบ่ง val ของตัวเองออกจาก train ซึ่งขัดหลัก "ห้าม shuffle" ของ time series · `max_iter = 3000` (เดิม 500 ซึ่งชน ceiling ทั้ง 20/20 fit — การแก้นี้มาจาก training convergence ล้วน ๆ ไม่ได้ดู MAE ของ val) |

### 3.2 Random Forest — bagging

| | |
|---|---|
| **ตอบโจทย์อะไร** | ลด **variance** ด้วยการเฉลี่ยต้นไม้หลายต้นที่เทรนบน bootstrap sample + สุ่ม feature subset (Breiman 1996 **R4** / 2001 **R5**) |
| **แก้ปัญหาอะไร** | ข้อมูล return noise สูงมาก (SD 0.011–0.019 แต่สัญญาณระดับ 1e-4) → โมเดลตัวเดียวจะจับ noise · การเฉลี่ย 400 ต้นทำให้คำตอบนิ่งขึ้นโดยไม่ต้องปรับ bias |
| **จุดอ่อน** | (ก) extrapolate ไม่ได้ (ค่าทำนายถูก bound ด้วยค่าเฉลี่ยใน leaf) — จัดการแล้วด้วยการทำนาย return · (ข) ต้นไม้ลึกจำ noise ได้หมด (default ของ sklearn คือ `max_depth=None`, `min_samples_leaf=1`) · (ค) **n_jobs=-1 ทำให้ผลไม่ reproducible** — ทดสอบรัน `main.py --dev` ซ้ำ 5 ครั้งด้วยข้อมูลเดียวกัน เจอ diff 2 ใน 4 ครั้ง → เปลี่ยนเป็น `n_jobs=1` (2026-09-29, ดู `KNOWN_ISSUES.md`) |
| **สิ่งที่ทำเพื่อคุม** | `max_depth = 4`, `min_samples_leaf = 50` (บังคับให้ leaf ใหญ่พอ), `max_features = 0.5`, `n_jobs = 1` |

### 3.3 XGBoost — boosting

| | |
|---|---|
| **ตอบโจทย์อะไร** | ลด **bias** แบบลำดับ: ต้นไม้ต้นถัดไปแก้ residual ของต้นก่อน (Chen & Guestrin 2016 **R6**) จับ interaction ที่ลึกกว่าได้ด้วยต้นไม้ตื้น ๆ หลายต้น |
| **แก้ปัญหาอะไร** | Random Forest เฉลี่ยต้นไม้อิสระ จึงเก่งลด variance แต่ไม่ค่อยช่วยถ้า bias เป็นปัญหา · boosting เข้าหาสัญญาณอ่อน ๆ ได้ทีละน้อยผ่าน learning rate ต่ำ |
| **จุดอ่อน** | (ก) overfit ง่ายกว่า RF ถ้า `learning_rate` สูงหรือต้นไม้ลึก · (ข) hyperparameter เยอะ · (ค) ปัญหา reproducibility เดียวกับ RF (และ **รุนแรงกว่า**: ต่างข้าม platform ~0.08% ดูข้อ 6) |
| **สิ่งที่ทำเพื่อคุม** | `max_depth = 2` (ตื้นมาก), `learning_rate = 0.01` (ต่ำ), `subsample = 0.8`, `colsample_bytree = 0.8`, `reg_lambda = 1.0`, `n_estimators = 300`, `n_jobs = 1` = ชุดค่าสาย conservative สำหรับ tabular ที่ noise สูง |

### 3.4 ทำไมไม่มี ARIMA / GARCH / linear regression — **เพราะไม่ได้ทำ (ข้อจำกัด)**

พูดตรง ๆ: **ยังไม่ได้ทดลอง** ไม่ได้มีเหตุผลเชิงทฤษฎีมาตัดออก · บันทึกไว้เป็นข้อจำกัดของงานนี้
และเป็นงานต่อ ไม่ใช่การเลือกที่พิจารณาแล้ว

| โมเดลที่ขาด | ถ้าทำจะได้อะไร | ความสำคัญ |
|---|---|---|
| **Linear regression / Ridge / Lasso** บน 29 feature เดียวกัน | จุดเทียบที่สำคัญที่สุดที่ขาดไป: ถ้า linear ได้ผลเท่า ANN/RF/XGB แปลว่า non-linearity ไม่ได้ช่วยอะไรเลย (ซึ่ง IC ~0.1 ชี้ไปทางนั้น) · Lasso ยังให้ feature selection ที่ตีความได้ | **สูง — ควรทำก่อนอย่างอื่น** ใช้เวลารันน้อยมาก |
| **ARIMA** | จับ autocorrelation ของ return โดยตรงจากอนุกรมเวลา ไม่ผ่าน feature ที่เราออกแบบเอง | กลาง — `features.py` มี `ret_1/2/3/5/10d` ซึ่งครอบ AR term ส่วนใหญ่อยู่แล้วทางอ้อม |
| **GARCH** | ทำนาย **volatility** ไม่ใช่ทิศทาง → ตอบคำถามคนละข้อกับงาน B (ซึ่งทำนายราคาปิด) แต่จะอธิบายได้ว่าทำไมบางช่วงทำนายยากกว่าช่วงอื่น (fold 2 ของ KBANK) | กลาง — เปลี่ยนนิยามงาน ต้องมี target ใหม่ |

ข้อจำกัดอื่นที่ควรบันทึกไว้คู่กัน: ไม่มี LSTM / Transformer (ข้อมูล 2,050 แถวน้อยเกินไป
สำหรับโมเดลสาย sequence ที่มีพารามิเตอร์เยอะ) และไม่มี ensemble ของ 3 โมเดลนี้

---

## 4. ค่าพารามิเตอร์แต่ละตัวมาจากไหน

ทุกค่าใน `config.py` ตอนนี้มาจาก **`tune.py` รันวันที่ 2026-09-27 ตามแผนที่ commit ไว้ก่อน**
ใน [`TUNING_PLAN.md`](TUNING_PLAN.md) · protocol: expanding window **3 folds × 2 หุ้น = 6 evaluation**
บน train+val (ตัดที่ `val_end` 2025-02-25 ก่อนสร้าง feature · ไม่แตะ test) ·
**17 configs ต่อโมเดลเท่ากันทุกตัว** (half-life 5 + hyperparameter 12) ·
จัดอันดับด้วย **mean MAE_return** อย่างเดียว ในบรรดา config ที่ผ่าน validity guard
`StdRatio ≥ 0.05` ทุก evaluation · ผลเต็มทุก config (ไม่ใช่แค่ผู้ชนะ) อยู่ใน
`results/tuning_scores.csv`, `tuning_stage1_halflife.csv`, `tuning_stage2_params.csv`,
[`results/tuning_summary.md`](results/tuning_summary.md)

| พารามิเตอร์ | ค่าที่ใช้ | ค่า Phase 1 เดิม | ที่มา |
|---|---|---|---|
| `RECENCY_HALF_LIFE` (ทั้ง 3 โมเดล) | `None` (ไม่ถ่วงน้ำหนัก) | `None` | stage 1 · grid `[None, 1000, 500, 250, 125]` → `None` ชนะทั้ง 3 โมเดล (`tuning_stage1_halflife.csv`) |
| ANN `alpha` | `7.0` | `1.0` | stage 2 · grid `[0.3, 1.0, 3.0, 7.0]` |
| ANN `hidden_layer_sizes` | `(8, 4)` | `(16, 8)` | stage 2 · grid `[(8,4), (16,8), (32,16)]` · ผู้ชนะกับที่สองต่างกันเพียง **−5.2e-07** (< `TIE_TOL` = 1e-5) → ตัดสินด้วยกฎเสมอที่ประกาศไว้ล่วงหน้า: เลือก hidden เล็กกว่า |
| RF `max_depth` / `min_samples_leaf` / `max_features` | `4` / `50` / `0.5` | `8` / `20` / `1.0` | stage 2 · `depth 4` ทุกตัวเสมอกัน → กฎเสมอเลือก depth ตื้นสุดแล้ว score ต่ำสุด |
| XGB `max_depth` / `learning_rate` / `subsample` | `2` / `0.01` / `0.8` | `3` / `0.05` / `0.8` | stage 2 · ผู้ชนะกับที่สองต่าง +1.29e-05 (เกิน `TIE_TOL`) → ตัดสินด้วยคะแนนตรง ๆ |
| ANN `max_iter` | `3000` | `500` | **ไม่ใช่การจูน** — มาจาก training convergence: ceiling 500 ชนทั้ง 20/20 fit โดย loss ยังลดต่อ ทดสอบ ceiling 3000 แล้วทุก seed หยุดเองที่ 502–869 iterations (คอมเมนต์ใน `config.py`) |
| `ANN_SEEDS` | `range(10)` | — | ส่วนหนึ่งของนิยามโมเดล ไม่ใช่การจูน (ดู 3.1) · ตอนจูนใช้ 3 seeds เพื่อลดเวลา แล้ว refit ผู้ชนะด้วย 10 seeds เป็นรายงาน |
| `n_jobs = 1` (RF, XGB) | `1` | `-1` | **ไม่ใช่การจูน** — มาจาก reproducibility (ดู 3.2) |
| `CONFIG_TAG` | `"phase1c-tuned"` | `"phase1b-untuned"` | ป้ายกำกับที่บันทึกลง `prediction_log.csv` ทุกแถว เพื่อแยกว่าแถวไหนมาจากโมเดลชุดไหน |

**ผลของการจูนทั้งหมดบน walk-forward 6 evaluation** (`tuning_summary.md`
บรรทัด "เทียบ config Phase 1" ของแต่ละโมเดล) — เทียบ config Phase 1 เดิมกับค่าที่จูนแล้ว:

| โมเดล | config Phase 1 | หลังจูน | เปลี่ยน |
|---|---|---|---|
| ANN (MLP) | 0.0098041429 | 0.0092176323 | **−5.982%** |
| XGBoost | 0.0094120968 | 0.0091660517 | **−2.614%** |
| Random Forest | 0.0091966726 | 0.0091666015 | −0.327% |

คำทำนายข้อ 2 ของ `TUNING_PLAN.md` ("primary_score ดีขึ้นจาก config Phase 1 **ไม่เกิน 2%**
ทุกโมเดล") จึง **ไม่ตรง** — ANN และ XGBoost ดีขึ้นเกิน 2% (`tuning_summary.md` บันทึกว่า
"ไม่ตรง" ไว้แล้ว) · ที่ประเมินต่ำไปคือ ANN ซึ่ง config Phase 1 overfit อยู่ การลด capacity
จึงให้ผลมากกว่าที่คาด

**แต่การจูนไม่ได้เปลี่ยนข้อสรุป**: หลังจูนแล้วทั้งสามโมเดลยัง mean relMAE > 1 บน
walk-forward (1.0025–1.0061) และ **ไม่มีโมเดลใดชนะ Naive อย่างมีนัยสำคัญแม้ 1 ใน 6
evaluation** (ข้อ 5.2) → การจูนย้ายโมเดลจาก "แพ้ Naive ชัดเจน" มาเป็น "เสมอ Naive"
ไม่ใช่ "ชนะ Naive"

---

## 5. ผลที่ได้จริงเทียบ Naive

### 5.1 fixed val (362 แถว) — `results/<TICKER>_taskB_val.csv` + `_val_stats.csv`

**ADVANC.BK**

| | MAE_return | relMAE | R2_return | StdRatio | DM (HLN) | p one-sided |
|---|---|---|---|---|---|---|
| **XGBoost** | 0.009425 | **0.9983** | +0.0239 | 0.0934 | −0.240 | 0.405 |
| ANN (MLP) | 0.009434 | **0.9993** | +0.0154 | 0.0780 | −0.123 | 0.451 |
| *Baseline: Naive (RW)* | *0.009441* | *1.0000* | *−0.0034* | *0.0000* | *0.000* | *1.000* |
| Random Forest | 0.009457 | 1.0017 | +0.0177 | 0.1044 | +0.210 | 0.583 |
| Baseline: Mean Return | 0.009475 | 1.0036 | −0.0016 | 0.0000 | +2.650 | 0.996 |

**KBANK.BK**

| | MAE_return | relMAE | R2_return | StdRatio | DM (HLN) | p one-sided |
|---|---|---|---|---|---|---|
| *Baseline: Naive (RW)* | *0.008375* | *1.0000* | *−0.0024* | *0.0000* | *0.000* | *1.000* |
| Baseline: Mean Return | 0.008382 | 1.0009 | −0.0028 | 0.0000 | +3.523 | 1.000 |
| Random Forest | 0.008390 | 1.0019 | +0.0082 | 0.0960 | +0.280 | 0.610 |
| XGBoost | 0.008402 | 1.0033 | +0.0018 | 0.0634 | +0.730 | 0.767 |
| ANN (MLP) | 0.008463 | 1.0106 | −0.0031 | 0.0994 | +1.511 | 0.934 |

(`MAE_return` ของ XGBoost เป็นค่าที่รันบนเครื่อง Windows ของรอบนี้ · ค่าที่ commit ไว้ใน repo
มาจาก container Linux และต่างกัน ~0.03–0.08% ตามที่ `KNOWN_ISSUES.md` บันทึกไว้ —
ดู `results/feedback/DECISIONS.md` ข้อ 1)

**ผลภายใต้ squared loss (MSE) ต่างจาก absolute loss ข้างบน** — DM test (p one-sided,
คอลัมน์ `dm_sq_*` ใน `results/<TICKER>_taskB_val_stats.csv`): ADVANC ANN (MLP) **0.0327**
(< 0.05) · ADVANC XGBoost / Random Forest 0.0591 (ทั้งคู่) · KBANK Random Forest 0.134 ·
KBANK XGBoost 0.220 · KBANK ANN (MLP) 0.527 — ตรงข้ามกับ absolute loss ที่ไม่มีช่องใดเข้าใกล้
0.05 เลย รายละเอียดวิธีตีความว่าทำไมสอง loss ต่างกัน ดู
[`results/feedback/WHY_NAIVE_WINS.md`](results/feedback/WHY_NAIVE_WINS.md) ข้อ 5

### 5.2 walk-forward 6 evaluation — `results/feedback/walkforward_summary.csv`

**Absolute loss (MAE)**

| โมเดล | mean MAE_return | mean relMAE | evals ที่ MAE ต่ำกว่า Naive | ชนะอย่างมีนัยสำคัญ | mean IC | hit rate (วันที่ราคาขยับ) |
|---|---|---|---|---|---|---|
| Random Forest | 0.009167 | 1.0025 | 2 / 6 | **0 / 6** | +0.1122 | 54.9% |
| XGBoost | 0.009172 | 1.0029 | 2 / 6 | **0 / 6** | +0.0772 | 54.0% |
| ANN (MLP) | 0.009199 | 1.0061 | 1 / 6 | **0 / 6** | +0.0933 | 55.3% |

**Squared loss (MSE)** — จาก `walkforward_stats.csv` คอลัมน์ `dm_sq_*`: evals ที่ MSE
ต่ำกว่า Naive สูงกว่า absolute loss มาก (Random Forest 6/6, ANN / XGBoost 5/6) และมีนัยสำคัญ
ก่อนปรับหลายการทดสอบรวม **3/18 evaluation ทุกโมเดล** (ANN 2/6, Random Forest 1/6, XGBoost
0/6 — ทั้งสามช่องเป็นหุ้น ADVANC) **แต่ไม่เหลือสักช่องหลัง Bonferroni correction**
(α/6 = 0.00833) → สัญญาณที่เห็นภายใต้ MSE มีอยู่จริงแต่ยังไม่แข็งพอให้สรุปว่าชนะ ·
ตารางเต็ม + การตีความ ดู [`results/feedback/WHY_NAIVE_WINS.md`](results/feedback/WHY_NAIVE_WINS.md) ข้อ 5

---

## 6. สิ่งที่เรียนรู้

**(1) ทั้งสามตระกูลให้ผลเท่ากันหมด** — mean MAE_return ต่างกันในช่วง
0.009167 → 0.009199 (**0.35%**) และ mean relMAE 1.0025 → 1.0061 ทั้งสามตัว
เรียงลำดับเดียวกับ Naive ภายใน 1% · เมื่อสามกลไกที่ไม่เหมือนกันเลย (neural / bagging /
boosting) ให้คำตอบเดียวกัน แปลว่า **ข้อจำกัดอยู่ที่ข้อมูล ไม่ใช่ที่การเลือกโมเดล**
(เหตุผลเดียวกับที่ `task_a` สรุปไว้ด้วยโมเดลชุดอื่นบนงาน classification)

**(2) อันดับโมเดลไม่คงที่ข้ามหุ้นและข้ามชุด** — บน fixed val ADVANC ชนะด้วย XGBoost
แต่ KBANK ดีที่สุดคือ Random Forest · บน walk-forward Random Forest นำเล็กน้อย ·
ส่วนต่างระดับ 1e-5 ถึง 1e-4 ซึ่ง **เล็กกว่าความต่างข้าม platform ของ XGBoost เองที่
~0.03–0.08%** → การประกาศผู้ชนะจากตัวเลขเหล่านี้ไม่มีความหมาย

**(3) การแยกส่วน MAE อธิบายกลไกได้ แต่ไม่ใช่หลักฐานของฝีมือโดยตัวมันเอง — ส่วน MSE ให้
สัญญาณอ่อน ๆ ที่น่าสนใจกว่า**
`contrib_zero_days` บวกทุกช่อง เพราะ Naive มี error = 0 เป๊ะบนวันนิ่ง 15% ของวัน แล้ว
`contrib_moved_days` ติดลบ 16 จาก 18 evaluation ตามมาเกือบเป็นผลทางบัญชี (เพื่อหักล้างกัน
เมื่อ MAE รวมใกล้ Naive) การแบ่งวันตาม `y` จริงก็ใช้ตอนทำนายไม่ได้ด้วย — ไม่ใช่กลยุทธ์จริง
หลักฐานที่น่าเชื่อกว่าคือ IC เป็นบวก 16/18 evaluation และผล DM ภายใต้ squared loss (MSE)
ที่โมเดล loss ต่ำกว่า Naive 16/18 evaluation โดยมีนัยสำคัญก่อนปรับหลายการทดสอบ 3/18
(ไม่เหลือหลัง Bonferroni) → รายละเอียดเต็มใน
[`results/feedback/WHY_NAIVE_WINS.md`](results/feedback/WHY_NAIVE_WINS.md)

**(4) การคุม capacity สำคัญกว่าการเลือกตระกูลโมเดล** — ANN v1 `(64,32)` + `alpha=1e-3`
ได้ `R2_return` −2.39 · ANN ตอนนี้ `(8,4)` + `alpha=7.0` ได้ −0.003 ถึง +0.015
= เปลี่ยนจาก "แย่กว่าการทายค่าเฉลี่ย 2.4 เท่า" เป็น "เสมอ" โดยไม่ได้เปลี่ยนตระกูลโมเดลเลย

**(5) reproducibility ไม่ฟรี** — ต้องแก้ `n_jobs=-1` → `1` ทั้ง RF และ XGB
เพื่อให้รันซ้ำบนเครื่องเดียวกันได้ค่าเท่าเดิม และ **ยังไม่พอ** สำหรับข้าม platform
→ regression check ของ `daily_next_day.py` ต้องมี tolerance (rtol 0.5%, atol 1e-5)

**(6) สิ่งที่ยังไม่รู้** — ยังไม่มี linear baseline (ข้อ 3.4) ซึ่งเป็นช่องว่างที่ใหญ่ที่สุด ·
และ **historical test ยังไม่ถูกเปิด** (ไม่มี `PRE_TEST_LOCK.md`) ทุกตัวเลขในไฟล์นี้
เป็น train/val เท่านั้น ซึ่ง fold 2–3 ของการจูนทับช่วง val ของ `main.py` →
ตัวเลขบน val หลังจูน **ไม่ใช่การประเมินอิสระ** (มองในแง่ดีเกินจริง)
