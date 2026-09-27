# TUNING_PLAN — ประกาศแผนการจูนก่อนรัน (Phase 1C §4, feedback ข้อ 11)

> เขียนและ commit **ก่อน** รัน `tune.py` ครั้งแรก (2026-09-27) · ห้ามแก้ไฟล์นี้หลังเห็นผล
> ถ้าต้องขยาย grid หรือเปลี่ยนเกณฑ์ ต้องเขียนไฟล์ใหม่ + commit ก่อน และรายงานว่าเป็น "รอบที่สอง"
>
> ข้อมูลที่ใช้: train+val เท่านั้น (ตัดที่ `SPLIT_BY_DATE["val_end"] = 2025-02-25` ก่อนสร้าง feature)
> **ไม่แตะ test set** · ไม่มี flag ใดใน `tune.py` เปิด test ได้

---

## 1. Grid ที่จะลอง (ครบทุกค่า)

### Stage 1 — half-life (5 configs ต่อโมเดล)

```python
HALF_LIFE_GRID = [None, 1000, 500, 250, 125]   # หน่วย = วันทำการ (แถว)
```

hyperparameter คงที่ที่ค่า Phase 1 (`ANN_PARAMS` / `RF_PARAMS` / `XGB_PARAMS` ใน config ณ commit นี้):

- ANN: `hidden_layer_sizes=(16, 8)`, `alpha=1.0`, adam, max_iter 3000, early_stopping False
- RF: `n_estimators=400`, `max_depth=8`, `min_samples_leaf=20`, `max_features=1.0` (ค่า default ของ sklearn)
- XGB: `n_estimators=300`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`

half-life เลือก**แยกต่อโมเดล** แต่ค่าเดียวใช้ทั้งสองหุ้น → `RECENCY_HALF_LIFE[name]`

### Stage 2 — hyperparameter (12 configs ต่อโมเดล)

half-life ถูก fix ที่ค่าที่ชนะ stage 1 ของโมเดลนั้น · พารามิเตอร์ที่ไม่อยู่ใน grid คงค่า Phase 1

```python
ANN_GRID = [                      # 4 alpha x 3 hidden = 12
    {"alpha": a, "hidden_layer_sizes": h}
    for a in [0.3, 1.0, 3.0, 7.0]
    for h in [(8, 4), (16, 8), (32, 16)]
]

RF_GRID = [                       # 3 depth x 2 leaf x 2 max_features = 12
    {"max_depth": d, "min_samples_leaf": l, "max_features": f}
    for d in [4, 8, 12]
    for l in [20, 50]
    for f in [1.0, 0.5]
]

XGB_GRID = [                      # 3 depth x 2 lr x 2 subsample = 12
    {"max_depth": d, "learning_rate": lr, "subsample": ss}
    for d in [2, 3, 4]
    for lr in [0.01, 0.05]
    for ss in [0.8, 1.0]
]
```

## 2. จำนวน configs ต่อโมเดล — เท่ากัน

| | Stage 1 | Stage 2 | รวม |
|---|---|---|---|
| ANN (MLP) | 5 | 12 | **17** |
| Random Forest | 5 | 12 | **17** |
| XGBoost | 5 | 12 | **17** |

(แก้ความไม่เท่าเทียมเดิม ANN 88 / XGB 27 / RF 0)
จำนวน fit = 17 × 6 evaluation × 3 โมเดล = 306 (ANN แต่ละ fit มี 3 seeds) + refit ผู้ชนะ ANN ด้วย 10 seeds อีก 6 fit

## 3. Validation scheme

- expanding window ด้วย `splits.walk_forward_splits(X, n_splits=3, min_train=1250)`
- `N_SPLITS = 3`, `MIN_TRAIN = 1250`
- บน train+val 2,050 แถว (หลัง `prepare_xy`) → fold_size = (2050 − 1250) // 3 = 266
  - fold 1: train[0:1250] eval[1250:1516]
  - fold 2: train[0:1516] eval[1516:1782]
  - fold 3: train[0:1782] eval[1782:2048]
- เตรียมข้อมูลแยกต่อหุ้นเหมือน `main.py`: truncate ถึง val_end → `build_features` / `build_targets`
  → `prepare_xy()` → `walk_forward_splits()` (ห้ามต่อสองหุ้นเป็นตารางเดียว)
- โมเดลห่อด้วย `TransformedTargetRegressor(StandardScaler)` เหมือน `main.py`
- น้ำหนัก recency คำนวณใหม่ **ทุก fold ทุกหุ้น** จากขนาด train ของ fold นั้น (mean = 1 ภายใน fold,
  แถวใหม่สุดของ fold หนักสุด) ส่งผ่าน `model__sample_weight`
- ANN ใช้ 3 seeds `[0, 1, 2]` ตอนจูน (ใช้เลือก config เท่านั้น) → winner ถูก freeze →
  refit ด้วย 10 seeds (`ANN_SEEDS`) บน 6 evaluation เดิม แล้วรายงานคู่กัน ·
  **ห้ามเลือก config ใหม่จากตัวเลข 10 seeds** · ถ้าต่างมาก รายงานเป็น seed sensitivity

## 4. เกณฑ์ตัดสิน (primary — ตัวเดียว)

```
primary_score = mean(MAE_return ของ 6 evaluation)   # KBANK x 3 folds + ADVANC x 3 folds
                                                     # น้ำหนักเท่ากันทุกหุ้น ทุก fold
```

ต่ำสุดชนะ · ค่าชุดเดียวต่อโมเดลใช้ทั้งสองหุ้น · ห้ามใช้ MAE_baht รวมข้ามหุ้น

## 5. Validity guard (ไม่ใช่ ranking)

- config ต้องมี `StdRatio >= 0.05` **ทุกหุ้น ทุก fold** (6/6) จึงผ่าน (`passed_guard = True`)
- StdRatio = `prediction_shape()` ของ `evaluate.py` (std(pred) / std(true))
- ไม่มี upper bound — StdRatio สูงรายงานตามจริง ไม่ตัดทิ้ง
- StdRatio ไม่ถูกใช้จัดอันดับ config การจัดอันดับใช้ mean MAE_return เพียงตัวเดียว
  แต่ใช้เป็น validity guard ที่ประกาศไว้ล่วงหน้าเพื่อกันคำตอบเสื่อม (degenerate solution)
- config ที่ไม่ผ่าน guard เก็บไว้ในไฟล์ผลลัพธ์ทั้งหมด ไม่ลบ

## 6. กฎตัดสินเสมอ

"เสมอ" = config ที่ผ่าน guard และมี primary_score ต่างจากค่าต่ำสุด (ในกลุ่มที่ผ่าน guard) **น้อยกว่า 1e-5**
ในกลุ่มนี้ เลือกตามลำดับ:

- **stage 1:** half-life ที่ถ่วงน้ำหนักน้อยกว่า — `None` > `1000` > `500` > `250` > `125`
- **stage 2 ANN:** hidden เล็กกว่า (จำนวน neuron รวมน้อยกว่า) แล้ว alpha สูงกว่า
- **stage 2 RF / XGB:** `max_depth` ตื้นกว่า · ถ้ายังเหลือหลายตัว เลือก primary_score ต่ำกว่า
  (สเปคกำหนดแค่ความลึก — ไม่ตั้งลำดับ "ความเรียบง่าย" ของพารามิเตอร์อื่นเอง)

## 7. ถ้าทุก config ของโมเดลใดไม่ผ่าน guard

หยุดและรายงานทันที · **ห้ามลดเกณฑ์ 0.05 หลังเห็นผล** · ผลแบบนี้คือข้อค้นพบ:
"โมเดลนี้ภายใต้ grid นี้ยุบเป็นค่าคงที่ทุกค่า"

## 8. สิ่งที่จะรายงาน — ทุก config

- `results/tuning_stage1_halflife.csv` — หนึ่งแถวต่อ evaluation:
  stage, model, config_id, half_life, ticker, fold, train_rows, eval_rows, MAE_return, StdRatio, Bias
- `results/tuning_stage2_params.csv` — เหมือนกัน + คอลัมน์พารามิเตอร์ที่เปลี่ยน
- `results/tuning_scores.csv` — หนึ่งแถวต่อ config: stage, model, config_id, primary_score,
  min_StdRatio, passed_guard, rank (+ คอลัมน์ winner)
- `results/tuning_summary.md` — ผู้ชนะแต่ละโมเดล + ค่าที่แพ้ที่ใกล้ที่สุด + config ที่ถูก guard ตัด
  + primary_score 3 seeds vs 10 seeds ของ ANN + คำทำนายข้อ 9 ตรง/ไม่ตรง
- full precision ทุกไฟล์ (ไม่ round ก่อนเขียน)

ข้อจำกัดที่จะเขียน:
- ค่าที่จูนได้เหมาะกับ training set 1,250–1,782 แถว (1,688 ที่ `main.py` ใช้อยู่ในช่วงนี้)
- fold 2–3 ประเมินบนช่วงที่ทับกับ validation split ของ `main.py` → ตาราง val ของ `main.py`
  หลังจูน **ไม่ใช่การประเมินที่เป็นอิสระ** อีกต่อไป (มองในแง่ดีเกินจริง) ·
  generalization gap เคยเกิดแล้ว (KBANK RF ชนะ val แต่แพ้ test ในรอบก่อน)
- ANN จัดอันดับด้วย 3 seeds → ranking มี noise มากกว่ารายงานสุดท้าย 10 seeds

## 9. คำทำนายก่อนรัน (เขียนก่อนเห็นผล — จะรายงานตรง/ไม่ตรงทุกข้อ)

1. **half-life:** อย่างน้อย 2 ใน 3 โมเดลจะได้ `None` หรือ `1000` · ไม่มีโมเดลใดได้ `125`
   (สัญญาณอ่อนมาก การลดน้ำหนักข้อมูลเก่ามักเพิ่ม variance มากกว่าได้ความสดใหม่)
2. **ขนาดการปรับปรุง:** primary_score ของ config สุดท้ายดีขึ้นจาก config Phase 1
   (stage 1 half-life=None) **ไม่เกิน 2%** ในทุกโมเดล
3. **ANN:** ผู้ชนะ stage 2 จะมี `alpha >= 3.0` (regularize มากขึ้น) และ ANN ยังมี
   primary_score แย่ที่สุดใน 3 โมเดล
4. **XGB:** ผู้ชนะ stage 2 จะมี `learning_rate = 0.01`
5. **RF:** ผู้ชนะ stage 2 จะมี `min_samples_leaf = 50`
6. **Guard:** อย่างน้อย 1 config จากทั้งหมด 51 จะไม่ผ่าน guard (คาดว่าเป็น RF depth 4 / leaf 50
   หรือ XGB lr 0.01) แต่ไม่มีโมเดลใดที่ทุก config ไม่ผ่าน
7. **ตารางหลักหลังจูน (`main.py --dev`):** ส่วนต่าง MAE_return ของโมเดลที่เลือกกับ Naive
   ยังน้อยกว่า 1% ของ MAE ของ Naive ทั้งสองหุ้น (ไม่มีการชนะขาด)
