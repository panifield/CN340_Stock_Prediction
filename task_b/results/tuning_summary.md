# tuning_summary — รันวันที่ 2026-09-27

> สร้างโดย `tune.py` ตามแผนที่ commit ไว้ก่อนใน `TUNING_PLAN.md`
> primary_score = mean MAE_return ของ 6 evaluation (2 หุ้น x 3 folds) · guard = StdRatio >= 0.05 ทุก evaluation · ANN จัดอันดับด้วย 3 seeds
> StdRatio ไม่ถูกใช้จัดอันดับ config การจัดอันดับใช้ mean MAE_return เพียงตัวเดียว แต่ใช้เป็น validity guard ที่ประกาศไว้ล่วงหน้าเพื่อกันคำตอบเสื่อม (degenerate solution)

## ANN (MLP)

### Stage 1

| config | primary_score | min_StdRatio | passed_guard | rank |
|---|---|---|---|---|
| half_life=None **(winner)** | 0.0098041429099838 | 0.347006675384314 | True | 1 |
| half_life=500 | 0.0098395331258773 | 0.3576799530962607 | True | 2 |
| half_life=1000 | 0.0099292972779663 | 0.3528589424983981 | True | 3 |
| half_life=250 | 0.0100991830048557 | 0.4153757969718257 | True | 4 |
| half_life=125 | 0.0106801342259611 | 0.4683569187562789 | True | 5 |

- ผู้ชนะ: `half_life=None` primary_score = 0.0098041429099838
- ค่าที่แพ้ที่ใกล้ที่สุด: `half_life=500` = 0.0098395331258773 (ต่างกัน +3.539e-05)
- ถูก guard ตัด: ไม่มี

### Stage 2

| config | primary_score | min_StdRatio | passed_guard | rank |
|---|---|---|---|---|
| alpha=7.0, hidden_layer_sizes=(16, 8) | 0.0092171125066359 | 0.0683470720968301 | True | 1 |
| alpha=7.0, hidden_layer_sizes=(8, 4) **(winner)** | 0.0092176323397794 | 0.0836285486541986 | True | 2 |
| alpha=7.0, hidden_layer_sizes=(32, 16) | 0.0092207815964384 | 0.0697315173816962 | True | 3 |
| alpha=3.0, hidden_layer_sizes=(8, 4) | 0.0094178742266437 | 0.2370293318255817 | True | 4 |
| alpha=3.0, hidden_layer_sizes=(16, 8) | 0.0095507262417281 | 0.2716557216084917 | True | 5 |
| alpha=1.0, hidden_layer_sizes=(8, 4) | 0.0095848366773046 | 0.2524992647478434 | True | 6 |
| alpha=3.0, hidden_layer_sizes=(32, 16) | 0.0095941045692985 | 0.3163874003403569 | True | 7 |
| alpha=0.3, hidden_layer_sizes=(8, 4) | 0.0097944646136331 | 0.2779386496565267 | True | 8 |
| alpha=1.0, hidden_layer_sizes=(16, 8) | 0.0098041429099838 | 0.347006675384314 | True | 9 |
| alpha=0.3, hidden_layer_sizes=(16, 8) | 0.0100913332852858 | 0.3957053285570719 | True | 10 |
| alpha=1.0, hidden_layer_sizes=(32, 16) | 0.0101167591552927 | 0.4513105326988141 | True | 11 |
| alpha=0.3, hidden_layer_sizes=(32, 16) | 0.0106985598881497 | 0.5199101958467153 | True | 12 |

- ผู้ชนะ: `alpha=7.0, hidden_layer_sizes=(8, 4)` primary_score = 0.0092176323397794
- ค่าที่แพ้ที่ใกล้ที่สุด: `alpha=7.0, hidden_layer_sizes=(16, 8)` = 0.0092171125066359 (ต่างกัน -5.198e-07) -- คะแนนต่ำกว่าผู้ชนะแต่ต่างไม่ถึง 1e-05 จึงตัดสินด้วยกฎเสมอ (TUNING_PLAN ข้อ 6)
- ถูก guard ตัด: ไม่มี

- เทียบ config Phase 1 (half_life=None, ค่าเดิม) = 0.0098041429099838 → สุดท้าย 0.0092176323397794 (-5.982% บน validation แบบ walk-forward)
- ANN ผู้ชนะ: 3 seeds = 0.0092176323397794 · 10 seeds = 0.0091994159854457 (ต่าง -1.822e-05) -- ตัวเลข 10 seeds ใช้รายงานเท่านั้น
- ANN ผู้ชนะ 10 seeds: min StdRatio = 0.0636 (3 seeds 0.0836) · ผ่าน guard = True

## Random Forest

### Stage 1

| config | primary_score | min_StdRatio | passed_guard | rank |
|---|---|---|---|---|
| half_life=None **(winner)** | 0.009196672585679 | 0.1047305943874653 | True | 1 |
| half_life=1000 | 0.0092051314835683 | 0.1230126930045151 | True | 2 |
| half_life=500 | 0.0092257325230583 | 0.1430139812115879 | True | 3 |
| half_life=250 | 0.0093329575937414 | 0.1719995104047587 | True | 4 |
| half_life=125 | 0.0093916199347058 | 0.1940809834851155 | True | 5 |

- ผู้ชนะ: `half_life=None` primary_score = 0.009196672585679
- ค่าที่แพ้ที่ใกล้ที่สุด: `half_life=1000` = 0.0092051314835683 (ต่างกัน +8.459e-06)
- ถูก guard ตัด: ไม่มี

### Stage 2

| config | primary_score | min_StdRatio | passed_guard | rank |
|---|---|---|---|---|
| max_depth=4, min_samples_leaf=50, max_features=0.5 **(winner)** | 0.0091666015178395 | 0.0894120456168855 | True | 1 |
| max_depth=4, min_samples_leaf=20, max_features=0.5 | 0.0091671179630177 | 0.0541989737155012 | True | 2 |
| max_depth=4, min_samples_leaf=50, max_features=1.0 | 0.0091688946759372 | 0.0941975001745847 | True | 3 |
| max_depth=4, min_samples_leaf=20, max_features=1.0 | 0.0091747984001957 | 0.0536534258450805 | True | 4 |
| max_depth=8, min_samples_leaf=50, max_features=0.5 | 0.009183441751646 | 0.1107090085277587 | True | 5 |
| max_depth=8, min_samples_leaf=50, max_features=1.0 | 0.0091839530309518 | 0.1144922612636989 | True | 6 |
| max_depth=12, min_samples_leaf=50, max_features=0.5 | 0.0091867691008478 | 0.1116077293067619 | True | 7 |
| max_depth=8, min_samples_leaf=20, max_features=0.5 | 0.0091874572957216 | 0.0980496145272296 | True | 8 |
| max_depth=12, min_samples_leaf=50, max_features=1.0 | 0.0091883179209266 | 0.115010620879875 | True | 9 |
| max_depth=8, min_samples_leaf=20, max_features=1.0 | 0.009196672585679 | 0.1047305943874653 | True | 10 |
| max_depth=12, min_samples_leaf=20, max_features=0.5 | 0.009205312398258 | 0.1309665042270151 | True | 11 |
| max_depth=12, min_samples_leaf=20, max_features=1.0 | 0.0092229145478523 | 0.1324694631406687 | True | 12 |

- ผู้ชนะ: `max_depth=4, min_samples_leaf=50, max_features=0.5` primary_score = 0.0091666015178395
- ค่าที่แพ้ที่ใกล้ที่สุด: `max_depth=4, min_samples_leaf=20, max_features=0.5` = 0.0091671179630177 (ต่างกัน +5.164e-07)
- ถูก guard ตัด: ไม่มี

- เทียบ config Phase 1 (half_life=None, ค่าเดิม) = 0.009196672585679 → สุดท้าย 0.0091666015178395 (-0.327% บน validation แบบ walk-forward)

## XGBoost

### Stage 1

| config | primary_score | min_StdRatio | passed_guard | rank |
|---|---|---|---|---|
| half_life=None **(winner)** | 0.0094120967658686 | 0.2609287508308834 | True | 1 |
| half_life=1000 | 0.0094811721190067 | 0.2565251216849443 | True | 2 |
| half_life=500 | 0.0095608603754211 | 0.2729842788896833 | True | 3 |
| half_life=250 | 0.0097276511832124 | 0.3102142295518202 | True | 4 |
| half_life=125 | 0.0100408122551934 | 0.3419263156983662 | True | 5 |

- ผู้ชนะ: `half_life=None` primary_score = 0.0094120967658686
- ค่าที่แพ้ที่ใกล้ที่สุด: `half_life=1000` = 0.0094811721190067 (ต่างกัน +6.908e-05)
- ถูก guard ตัด: ไม่มี

### Stage 2

| config | primary_score | min_StdRatio | passed_guard | rank |
|---|---|---|---|---|
| max_depth=2, learning_rate=0.01, subsample=0.8 **(winner)** | 0.0091660516967173 | 0.0568624544327635 | True | 1 |
| max_depth=2, learning_rate=0.01, subsample=1.0 | 0.0091789446387659 | 0.0761458420089378 | True | 2 |
| max_depth=3, learning_rate=0.01, subsample=0.8 | 0.0091845133096951 | 0.0809665648415852 | True | 3 |
| max_depth=4, learning_rate=0.01, subsample=0.8 | 0.0091928383965295 | 0.0947340342739835 | True | 4 |
| max_depth=3, learning_rate=0.01, subsample=1.0 | 0.0092007552930019 | 0.0846641683228797 | True | 5 |
| max_depth=4, learning_rate=0.01, subsample=1.0 | 0.0092513149828516 | 0.1306353999375968 | True | 6 |
| max_depth=2, learning_rate=0.05, subsample=1.0 | 0.0092922394531591 | 0.1720511474622209 | True | 7 |
| max_depth=2, learning_rate=0.05, subsample=0.8 | 0.0093103487200274 | 0.201907598723718 | True | 8 |
| max_depth=3, learning_rate=0.05, subsample=0.8 | 0.0094120967658686 | 0.2609287508308834 | True | 9 |
| max_depth=4, learning_rate=0.05, subsample=0.8 | 0.0094955100013751 | 0.3034790195168149 | True | 10 |
| max_depth=3, learning_rate=0.05, subsample=1.0 | 0.0095068736897869 | 0.231718928094493 | True | 11 |
| max_depth=4, learning_rate=0.05, subsample=1.0 | 0.0096801193793304 | 0.3172982494008726 | True | 12 |

- ผู้ชนะ: `max_depth=2, learning_rate=0.01, subsample=0.8` primary_score = 0.0091660516967173
- ค่าที่แพ้ที่ใกล้ที่สุด: `max_depth=2, learning_rate=0.01, subsample=1.0` = 0.0091789446387659 (ต่างกัน +1.289e-05)
- ถูก guard ตัด: ไม่มี

- เทียบ config Phase 1 (half_life=None, ค่าเดิม) = 0.0094120967658686 → สุดท้าย 0.0091660516967173 (-2.614% บน validation แบบ walk-forward)

## คำทำนายก่อนรัน (TUNING_PLAN ข้อ 9) — ตรง / ไม่ตรง

| คำทำนาย | ผล | สิ่งที่เกิดจริง |
|---|---|---|
| 1. half-life: >= 2 ใน 3 โมเดลได้ None/1000 และไม่มีโมเดลใดได้ 125 | ตรง | ได้ {'ANN (MLP)': None, 'Random Forest': None, 'XGBoost': None} |
| 2. primary_score ดีขึ้นจาก config Phase 1 ไม่เกิน 2% ทุกโมเดล | **ไม่ตรง** | ANN (MLP) -5.982%, Random Forest -0.327%, XGBoost -2.614% |
| 3. ANN ผู้ชนะมี alpha >= 3.0 และ ANN แย่สุดใน 3 โมเดล | ตรง | ANN {'alpha': 7.0, 'hidden_layer_sizes': (8, 4)} · scores {'ANN (MLP)': 0.0092176323397794, 'Random Forest': 0.0091666015178395, 'XGBoost': 0.0091660516967173} |
| 4. XGB ผู้ชนะมี learning_rate = 0.01 | ตรง | {'max_depth': 2, 'learning_rate': 0.01, 'subsample': 0.8} |
| 5. RF ผู้ชนะมี min_samples_leaf = 50 | ตรง | {'max_depth': 4, 'min_samples_leaf': 50, 'max_features': 0.5} |
| 6. >= 1 config ไม่ผ่าน guard แต่ไม่มีโมเดลใดไม่ผ่านทุกค่า | **ไม่ตรง** | ถูกตัด 0 configs จาก 51 |
| 7. ตารางหลัก main.py --dev: ส่วนต่างโมเดลที่เลือกกับ Naive < 1% ทั้งสองหุ้น | (ตรวจหลังนำค่าเข้า config) | ดูหัวข้อด้านล่าง |

## ข้อจำกัด
- ค่าที่จูนได้เหมาะกับ training set 1,250–1,782 แถว (1,688 ของ main.py อยู่ในช่วงนี้)
- fold 2–3 ทับกับ validation split ของ main.py → ตาราง val ของ main.py หลังจูน ไม่ใช่การประเมินอิสระ (มองในแง่ดีเกินจริง) · generalization gap เคยเกิดแล้ว (KBANK RF)
- ANN จัดอันดับด้วย 3 seeds → ranking มี noise มากกว่ารายงานสุดท้าย 10 seeds
- ผลทั้งหมดเป็น validation แบบ walk-forward ไม่ใช่ test

## ผลคำทำนายข้อ 7 (ตรวจหลังนำค่าเข้า config แล้วรัน `main.py --dev`)

| หุ้น | โมเดลที่เลือกจาก val | MAE_return | Naive | ส่วนต่าง | % ของ Naive |
|---|---|---|---|---|---|
| KBANK | Random Forest | 0.008390 | 0.008375 | +0.000016 (แพ้) | 0.19% |
| ADVANC | XGBoost | 0.009430 | 0.009441 | −0.000011 (ชนะ) | 0.12% |

→ **ตรง** (ส่วนต่าง < 1% ทั้งสองหุ้น ไม่มีการชนะขาด)

ข้อสังเกต: ตาราง val ของ `main.py` ไม่ใช่การประเมินอิสระของการจูน (fold 2–3 ทับช่วง val)
และ KBANK RF หลังจูน **แย่ลง** บน val split เดิม (0.008335 → 0.008390) ทั้งที่ walk-forward ดีขึ้น
→ เตือนอีกครั้งว่าผลบน validation หนึ่งช่วงไม่เสถียร

## Post-hoc interpretation notes (เพิ่มภายหลัง — ไม่มีตัวเลขใดเปลี่ยน)

> เพิ่มโดยมือหลัง tune.py รันเสร็จ (Session 1, 2026-09-27) · เนื้อหาด้านบนและตัวเลขทุกตัวไม่ถูกแก้
> tune.py ไม่ได้ถูกแก้หรือรันใหม่เพื่อสร้างหัวข้อนี้

**Validation reuse for hyperparameter selection** (คำนวณใหม่จาก implementation จริง:
truncate ถึง val_end -> features/targets -> prepare_xy -> walk_forward_splits(n_splits=3, min_train=1250))

| หุ้น | แถว fixed validation (หลัง train_end) | อยู่ใน eval window ของการจูน | ไม่ถูกครอบคลุม |
|---|---|---|---|
| KBANK | 362 | 360 | 2025-02-24, 2025-02-25 |
| ADVANC | 362 | 360 | 2025-02-24, 2025-02-25 |

eval windows: fold 1 2021-11-11 -> 2022-12-19 · fold 2 2022-12-20 -> 2024-01-19 · fold 3 2024-01-22 -> 2025-02-21 (ทั้งสองหุ้น)
- นี่คือ validation reuse ไม่ใช่ leakage: ไม่มีข้อมูลอนาคตเข้า feature/target
- ตาราง fixed validation ของ `main.py --dev` หลังจูน = descriptive development result
  **ไม่ใช่ independent holdout** (360/362 แถวถูกใช้เลือก hyperparameter แล้ว)

**StdRatio guard**
- ทั้ง 51 searched configs ผ่าน guard → guard **ไม่ได้ตัด config ใดในรอบนี้**
- ผู้ชนะ stage 2 มี low-amplitude predictions (shrinkage เข้าหาค่าเฉลี่ย) — min StdRatio จาก `results/tuning_scores.csv`
  (3-seed search สำหรับ ANN):

| โมเดล | ผู้ชนะ stage 2 | min StdRatio (6 evaluation) |
|---|---|---|
| ANN (MLP) | alpha 7.0, (8, 4) | 0.0836 |
| Random Forest | depth 4, leaf 50, max_features 0.5 | 0.0894 |
| XGBoost | depth 2, lr 0.01, subsample 0.8 | 0.0569 |

  ช่วงของผู้ชนะ ≈ 0.057–0.089
- ANN 10-seed refit (sensitivity แยก ไม่นับในช่วงข้างบน): min StdRatio 0.0636
- guard ยังมีหน้าที่กันกรณี degenerate ที่รุนแรงกว่านี้ (StdRatio < 0.05) แต่ในรอบนี้ไม่มีกรณีดังกล่าวเกิดขึ้น

**alpha = 7** ถูกเลือกซ้ำภายใต้ validation procedure ที่ต่างกัน (alpha sweep รุ่นก่อนบน fixed val split — `results_reference/ann_sweep_val_29feat.csv` แถว selected: (16,8), alpha 7.0 — กับ walk-forward รอบนี้)
= some robustness to the choice **ไม่ใช่ independent confirmation** เพราะทั้งสองครั้งใช้ช่วงข้อมูลที่ทับกัน

**Recency weighting:** half_life = None ชนะทั้ง 3 โมเดล

**จำนวน config:** stage 1 (5) + stage 2 (12) = 17 searched configs ต่อโมเดล × 3 = 51
แถว `2_refit10` ของ ANN = post-selection sensitivity/refit record **ไม่ใช่ searched configuration**
