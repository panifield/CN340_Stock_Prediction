# Task C — ข้อมูลสำหรับสไลด์นำเสนอ

> ผลในเอกสารนี้เป็นผลจากโหมด `python main.py --dev` วันที่ใช้ config
> ปัจจุบัน จึงเป็น **validation result** เพื่อเลือกโมเดล/ปรับค่าเท่านั้น
> ยังไม่ได้ใช้ test set ในการตัดสินผล

## 1. โจทย์และ Target

- โจทย์: ทำนายทิศทางราคาปิดหุ้นวันถัดไป
- Target: `y_updown = 1` เมื่อ `Close[t] > Close[t-1]`; มิฉะนั้นเป็น `0`
  (ลงหรือราคาเท่าเดิม)
- การจัดเวลา: input ของแถว `t` ใช้ข้อมูลไม่เกินวัน `t-1`; จึงไม่มีข้อมูล
  ราคาของวันที่ต้องทำนายหลุดเข้าไป
- หุ้นที่ทดลอง: `KBANK.BK` และ `ADVANC.BK`
- ข้อมูลจริงหลังทำความสะอาด: หุ้นละ 2,433 trading days

## 2. Input และการ Preprocess

### 2.1 Raw data ก่อนเข้าโมเดล

ข้อมูลรายวันมาจาก Yahoo Finance (`auto_adjust=False`) และ cache เป็น CSV โดยมี
5 field ดิบต่อวัน:

| Field ดิบ | ความหมาย | ใช้โดยตรงเป็น input หรือไม่ |
|---|---|---|
| `Open` | ราคาเปิด | ไม่ใช้โดยตรง; ใช้สร้าง `oc_change` |
| `High` | ราคาสูงสุดของวัน | ไม่ใช้โดยตรง; ใช้สร้าง range และ ATR |
| `Low` | ราคาต่ำสุดของวัน | ไม่ใช้โดยตรง; ใช้สร้าง range และ ATR |
| `Close` | ราคาปิด | ไม่ใช้โดยตรง; ใช้สร้าง target, return และ technical indicators |
| `Volume` | ปริมาณซื้อขาย | ไม่ใช้โดยตรง; ใช้สร้าง change/ratio ของ volume |

เหตุผลที่ตัด raw OHLCV ออกจาก input คือราคาหุ้นและ volume มีระดับ (scale) ที่
เปลี่ยนตามเวลา เช่น train อาจอยู่ช่วงราคา 100 บาท แต่ test อยู่ช่วง 200 บาท
โมเดลอาจจำ “ระดับราคา” แทนที่จะเรียนรู้รูปแบบการเคลื่อนไหว จึงแปลงเป็น return,
ratio หรือ indicator ก่อน

### 2.2 Cleaning raw data

ก่อนสร้าง feature ระบบทำตามลำดับนี้:

1. เรียง index ตามวันที่ และตัดวันที่ซ้ำโดยเก็บแถวแรก
2. ตัดแถวที่ `Close` เป็น missing หรือ `Close <= 0` เพราะสร้าง target/ratio ไม่ได้
3. ไม่เติมค่าราคา/volume ดิบทันที: missing ที่เหลือจะส่งผลเป็น `NaN` ใน feature
   ที่เกี่ยวข้อง แล้วจัดการในขั้น imputation หลัง split
4. หาก `Volume = 0` โปรแกรมเตือน และตอนคำนวณ `volume_change` จะเปลี่ยน
   denominator 0 เป็น `NaN` เพื่อไม่ให้เกิด `inf`

### 2.3 Feature engineering และ normalization ก่อนเข้าโมเดล

ใช้ input ทั้งหมด **33 features**. ตารางนี้แสดงว่า raw field ถูกแปลงเป็นอะไร
และ normalization อยู่ตรงไหน:

| กลุ่ม | จำนวน | Feature | สูตร/การแปลงจาก raw data |
|---|---:|---|---|
| Return | 5 | `ret_1d`, `ret_2d`, `ret_3d`, `ret_5d`, `ret_10d` | `Close[t] / Close[t-k] - 1` — percent return |
| รูปแท่งเทียน | 3 | `hl_range`, `oc_change`, `close_pos_in_range` | `(High-Low)/Close`, `(Close-Open)/Open`, `(Close-Low)/(High-Low)` |
| Trend / MA | 5 | `close_over_sma5`, `close_over_ema5`, `close_over_sma20`, `close_over_ema20`, `sma5_over_sma20` | ราคา/MA และ MA สั้น/MA ยาว จึงไม่มีหน่วยบาท |
| Volatility / momentum | 7 | `volatility_5d`, `volatility_20d`, `rsi`, `macd`, `macd_signal`, `macd_hist`, `bb_position` | volatility คือส่วนเบี่ยงเบนมาตรฐานของ return; MACD ทุกตัวหารด้วย Close; RSI อยู่ช่วง 0–100; BB position เป็นตำแหน่งใน Bollinger Band |
| Volatility เพิ่มเติม | 2 | `bb_width`, `atr_norm` | ความกว้าง Bollinger/MA และ ATR/Close |
| Momentum / breakout | 4 | `up_streak`, `prev_direction`, `dist_from_high20`, `dist_from_low20` | ความต่อเนื่อง/ทิศทางราคา และ `Close/rolling max(or min) - 1` |
| Volume แบบ relative | 2 | `volume_change`, `volume_over_ma20` | percent change ของ volume และ `Volume/MA20(Volume)` |
| วันในสัปดาห์ | 5 | `dow_0` ถึง `dow_4` | one-hot encoding; ค่า 0 หรือ 1 |

**Winsorization:** `ret_*` และ `volume_change` ถูก clip ที่ quantile 1st–99th
percentile แบบ expanding โดยเริ่มเมื่อมีข้อมูลอย่างน้อย 60 วัน เพื่อจำกัด outlier
จากวันข่าวแรง/ข้อมูลผิดปกติ ทั้ง quantile และ feature ณ วัน `t` ใช้ข้อมูลไม่เกิน
วัน `t` เท่านั้น

**ป้องกัน leakage:** หลังคำนวณทุก feature แล้ว จึง `shift(1)` ทั้งตาราง ดังนั้น
input จริงของแถว `t` จะลงท้าย `_prev` และเท่ากับ feature ที่คำนวณจากวัน `t-1`.
ส่วน target ของแถวเดียวกันคือการเปลี่ยนจาก `Close[t-1]` ไป `Close[t]`.

### 2.4 Missing values และ Imputation

`NaN` เกิดได้ตามปกติจาก rolling window (เช่น MA20/ATR14), return ช่วงต้นข้อมูล,
expanding winsorization 60 วันแรก หรือ raw field ที่หายไป

1. ก่อน split: align index ของ `X` กับ target แล้วตัดเฉพาะแถวที่ missing ตั้งแต่
   50% ของ feature ขึ้นไป; ข้อมูลที่ใช้จริงจึงเริ่ม 2016-08-30 และเหลือ 2,431
   rows ต่อหุ้น
2. หลัง split: ใช้ `SimpleImputer(strategy="median")` โดย **fit median จาก
   train set เท่านั้น** แล้วนำ median เดิมไปเติม validation/test
3. เหตุผลที่ใช้ median: ทนต่อ outlier มากกว่า mean โดยเฉพาะ return และ volume
   change ที่อาจมีวันที่ผันผวนสูง

### 2.5 Standardization ก่อนเทรน

ลำดับ pipeline คือ `median imputation → scaling (เฉพาะ ANN/Logistic) → model`

| โมเดล | ทำ Standardization หรือไม่ | วิธี |
|---|---|---|
| ANN | ใช่ | `z = (x - μ_train) / σ_train` ด้วย `StandardScaler`; fit `μ, σ` จาก train เท่านั้น |
| Logistic Regression | ใช่ | ใช้ `StandardScaler` แบบเดียวกับ ANN เพื่อให้ coefficient/gradient ไม่ถูกครอบงำด้วย scale |
| XGBoost | ไม่ | ใช้ median imputation เท่านั้น เพราะ tree split ไม่ไวต่อ scale ของ feature |

ดังนั้น raw data ถูก normalize เป็น ratio/return ตั้งแต่ feature engineering แล้ว
ANN และ Logistic Regression ยังถูก standardize ซ้ำในระดับ feature หลัง imputation
เพื่อให้ทุก column มี scale ใกล้เคียงกันขณะ optimize

### Feature selection

- เลือก feature แบบ domain-driven และลดความซ้ำซ้อนก่อนเทรน
- ตัด raw OHLCV ออก; เก็บเฉพาะ feature ที่เป็น return, ratio หรือ indicator
- ลด MA windows จาก `[5, 10, 20]` เหลือ `[5, 20]` เพื่อลด multicollinearity
- การรันและผลในเอกสารนี้ใช้ครบ **33 features** ทุกตัว; ยังไม่มีการใช้ L1
  หรือ Top-K feature selection

## 3. การแบ่งข้อมูลตามเวลา

ห้าม shuffle เพราะเป็น time series. หลังตัดแถวต้นช่วงที่ยังสร้าง target/feature
ไม่ได้ เหลือหุ้นละ 2,431 rows และแบ่งเหมือนกันทั้งสองหุ้น:

| ชุดข้อมูล | สัดส่วน | จำนวนแถว | ช่วงเวลา | การใช้งาน |
|---|---:|---:|---|---|
| Train | 70% | 1,701 | 2016-08-30 – 2023-08-24 | fit model, imputer และ scaler |
| Validation | 15% | 365 | 2023-08-25 – 2025-02-20 | เลือก hyperparameter/model |
| Test | 15% | 365 | 2025-02-21 – 2026-08-28 | เก็บไว้ประเมินครั้งสุดท้ายเท่านั้น |

## 4. โมเดล, Loss และ Regularization

| โมเดล | การตั้งค่าหลัก | Loss function | Regularization |
|---|---|---|---|
| ANN (`MLPClassifier`) | input 33 → hidden `(16, 8)` → output 1, ReLU, learning rate `0.0005`, max 150 iterations | Binary Cross-Entropy: `BCE = -(1/n)Σ[y log(p) + (1-y)log(1-p)]` | L2 weight decay, `alpha = 0.01`; `J = BCE + alpha/(2n)Σ||W||²` |
| Logistic Regression | `lbfgs`, `max_iter=2000`, `class_weight='balanced'` | weighted Binary Cross-Entropy | L2, `C=1.0` (C เป็น inverse ของความแรง regularization; แนวคิดคือ λ ∝ `1/C`) |
| XGBoost | 400 trees, depth 4, learning rate 0.05, subsample/colsample 0.8 | binary logistic log loss | L2 บน leaf weights, `reg_lambda = 1.0` |

รายละเอียดสูตรและฟังก์ชันคำนวณอยู่ใน `loss_function.py`

### ตารางสรุป Loss function และ Regularization (พร้อมใช้ในสไลด์)

| โมเดล/ขั้นตอน | Loss ที่ลดระหว่าง train | สูตร objective แบบย่อ | Regularization | ค่าที่ตั้งในรอบนี้ |
|---|---|---|---|---|
| ANN | Binary Cross-Entropy (BCE) | `J = BCE + alpha/(2n) Σ||W||²` | L2 บน ANN weights (ไม่รวม bias) | `alpha = 0.01` |
| Logistic Regression | weighted BCE เพราะ `class_weight="balanced"` | `J ≈ weighted BCE + (1/(2C))Σw²` | L2 บน coefficients | `C = 1.0`, `penalty="l2"`, solver `lbfgs` |
| XGBoost | binary logistic log loss | `J = Σlogloss + (lambda/2)Σw_leaf²` | L2 บน leaf weights | `reg_lambda = 1.0` |

> หมายเหตุ: `alpha`, `C`, และ `reg_lambda` เป็นชื่อพารามิเตอร์คนละ library
> จึงเทียบเลขตรง ๆ ไม่ได้: `alpha` สูง = L2 ANN แรงขึ้น, `C` ต่ำ = L2/L1
> Logistic Regression แรงขึ้น, `reg_lambda` สูง = L2 XGBoost แรงขึ้น

### ค่า loss ที่วัดได้จริงของ ANN (ใช้ครบ 33 features)

| หุ้น | จำนวน feature | Train BCE | L2 term | Train objective = BCE + L2 | Validation BCE | Train–validation gap |
|---|---:|---:|---:|---:|---:|---:|
| KBANK | 33 | 0.6260 | 0.0001 | 0.6261 | 0.7065 | 0.0804 |
| ADVANC | 33 | 0.6367 | 0.0001 | 0.6369 | 0.6797 | 0.0428 |

## 5. ตรวจ Overfitting ของ ANN

| หุ้น | Train BCE | L2 term | Train objective | Validation BCE | สรุป |
|---|---:|---:|---:|---:|---|
| KBANK | 0.6260 | 0.0001 | 0.6261 | 0.7065 | ช่องว่าง 0.0804 — ไม่ overfit รุนแรง |
| ADVANC | 0.6367 | 0.0001 | 0.6369 | 0.6797 | ช่องว่าง 0.0428 — ไม่ overfit รุนแรง |

ANN ทั้งสองหุ้นเทรนครบเพดาน 150 iterations; หากจะจูนต่อให้ตัดสินด้วย
validation เท่านั้น และยังไม่ดู test

## 6. ผลบน Validation Set

### KBANK.BK

| โมเดล | Accuracy | Precision | Sensitivity / Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| ANN | 0.5425 | 0.4828 | 0.1697 | 0.2511 | 0.5148 |
| Logistic Regression | 0.5151 | 0.4620 | 0.4424 | 0.4520 | 0.5056 |
| XGBoost | **0.5836** | **0.5528** | 0.4121 | **0.4722** | **0.6126** |
| Majority baseline | 0.5479 | – | 0.0000 | – | – |

เลือก **XGBoost** สำหรับ KBANK เพราะ Accuracy สูงสุดและชนะ majority baseline
`0.0357`

### ADVANC.BK

| โมเดล | Accuracy | Precision | Sensitivity / Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| ANN | **0.6137** | **0.6000** | 0.3057 | 0.4051 | **0.5905** |
| Logistic Regression | 0.5808 | 0.5119 | **0.5478** | **0.5292** | 0.5853 |
| XGBoost | 0.5644 | 0.4902 | 0.3185 | 0.3861 | 0.5323 |
| Majority baseline | 0.5699 | – | 0.0000 | – | – |

เลือก **ANN** สำหรับ ADVANC หากใช้ Accuracy เป็นเกณฑ์: ชนะ majority baseline
`0.0438` แต่ Logistic Regression มี F1/Recall สูงกว่า จึงควรรายงานทั้งสอง
metric ไม่ดู Accuracy เพียงอย่างเดียว

## 7. Confusion Matrix และ Sensitivity / Specificity

นิยาม: `Sensitivity = TP / (TP + FN)` และ `Specificity = TN / (TN + FP)`

### KBANK — XGBoost (โมเดลที่เลือกตาม Accuracy)

| Actual \ Predicted | ลง/นิ่ง (0) | ขึ้น (1) |
|---|---:|---:|
| ลง/นิ่ง (0) | TN = 145 | FP = 55 |
| ขึ้น (1) | FN = 97 | TP = 68 |

- Accuracy = `(145 + 68) / 365` = **0.5836**
- Sensitivity = `68 / (68 + 97)` = **0.4121**
- Specificity = `145 / (145 + 55)` = **0.7250**

### ADVANC — ANN (โมเดลที่เลือกตาม Accuracy)

| Actual \ Predicted | ลง/นิ่ง (0) | ขึ้น (1) |
|---|---:|---:|
| ลง/นิ่ง (0) | TN = 176 | FP = 32 |
| ขึ้น (1) | FN = 109 | TP = 48 |

- Accuracy = `(176 + 48) / 365` = **0.6137**
- Sensitivity = `48 / (48 + 109)` = **0.3057**
- Specificity = `176 / (176 + 32)` = **0.8462**

## 8. ประเด็นสรุปสำหรับพูดในสไลด์

- โมเดลทำนายทิศทางได้ดีกว่า majority baseline ใน validation สำหรับ KBANK
  (XGBoost) และ ADVANC (ANN ตาม Accuracy)
- ค่า Sensitivity ต่ำกว่า Specificity ทั้งสองหุ้น: โมเดลจับวันขึ้นได้ยากกว่า
  และมีแนวโน้มทายลง/นิ่งมากกว่า
- `flat day` มี KBANK 12.37% และ ADVANC 14.47% ของข้อมูลทั้งหมด แต่ถูกจัดใน
  class 0 ร่วมกับวันลง; นี่เป็นข้อจำกัดของนิยาม target ปัจจุบัน
- ขั้นถัดไปที่ควรทดลอง: tune probability threshold บน validation หรือแยก
  `ลง / นิ่ง / ขึ้น` เป็น 3 classes โดยห้ามใช้ test เพื่อเลือกค่า
