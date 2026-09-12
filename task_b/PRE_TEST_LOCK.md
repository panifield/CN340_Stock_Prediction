# Pre-Test Lock — งาน B

**ล็อกเมื่อ:** 2026-09-12
**commit ล่าสุดก่อนล็อก:** `26f278b` — Tune XGBoost on val with the ANN selection rule (for fairness)
**commit ของการล็อก:** ไฟล์นี้ถูก commit พร้อมงาน error analysis / rolling baselines / ensemble
ก่อนเปิด test — เวลาของ commit นั้นคือหลักฐานว่าทุกอย่างด้านล่างถูกกำหนดไว้ก่อนเห็น test

> ตั้งแต่ commit นี้เป็นต้นไป **ห้ามแก้ค่าใด ๆ ด้านล่าง** จนกว่าจะรายงานผล test เสร็จ

---

## DATA

| | |
|---|---|
| แหล่งข้อมูล | investing.com — `raw_data/KBANK_10Y_Cleaned.csv`, `raw_data/ADVANC_10Y_Cleaned.csv` |
| `DATA_SOURCE` | `"investing"` — ถ้าไฟล์หายจะ `raise` ทันที **ไม่ fallback ไป Yahoo** |
| ช่วงเวลา | 2016-08-26 ถึง 2026-08-28 (2,433 แถวต่อหุ้น) |
| การตัดแถว | ตัด 20 แถวแรกที่ rolling window ยังคำนวณไม่ครบ (ไม่ใช้ median เติม) |
| การแบ่ง | ตามเวลา ห้าม shuffle — train 1,688 / val 362 / test 362 แถว |
| สัดส่วน | `TRAIN_RATIO = 0.70`, `VAL_RATIO = 0.15`, `SPLIT_BY_DATE = None` |
| ช่วง train | 2016-09-26 → 2023-09-01 |
| ช่วง val | 2023-09-04 → 2025-02-25 |
| ช่วง test | 2025-02-26 → 2026-08-28 **(ยังไม่เปิด)** |

---

## FEATURES

- **29 features** — `USE_RAW_PRICE_LEVELS = False` (ตัดกลุ่มราคาดิบ 5 ตัวตาม ablation H1)
- `shift(1)` ทั้งตารางทีเดียวตอนท้าย → แถววัน t ใช้ข้อมูลถึงวัน t-1 เท่านั้น
- กลุ่ม: Return lag (5) · รูปแท่งเทียน (3) · อัตราส่วน MA (7) · ความผันผวน (2) · Momentum (5) · ปริมาณซื้อขาย (2) · วันในสัปดาห์ (5)
- `LAG_DAYS = [1,2,3,5,10]` · `MA_WINDOWS = [5,10,20]` · `VOL_WINDOWS = [5,20]` · `RSI_PERIOD = 14`
- **known issue ที่ยอมรับแล้ว:** `dow_*` ถูก shift ไปด้วย จึงเป็นวันในสัปดาห์ของ t-1
  ผิดจริงเฉพาะหลังวันหยุด 120/2,050 แถว (5.9%) — ทดสอบโดยตัด dow ออกทั้งกลุ่มแล้ว
  MAE ของ RF เปลี่ยน < 0.2% และข้อสรุป DM ไม่เปลี่ยน จึงไม่แก้ (ดู `features.py`)

---

## MODELS — ล็อกค่าทั้งหมด

### ANN (MLP) — จูนบน val 140 ชุด
```
MLPRegressor(hidden_layer_sizes=(16,8), activation="relu", alpha=7.0,
             solver="adam", learning_rate_init=1e-3, max_iter=500,
             early_stopping=False, n_iter_no_change=20)
seed averaging: seed 0-9 น้ำหนักเท่ากัน (ไม่เลือก seed ที่ดีที่สุด)
```
`early_stopping = False` เพราะถ้าเปิด `MLPRegressor` จะ shuffle แบ่ง validation ของตัวเอง
ซึ่งขัดกับหลัก "ห้าม shuffle" ของ time series

### Random Forest — **ไม่ได้จูน**
```
RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=20,
                      max_features=ค่าเริ่มต้น (1.0))
```
ทดสอบความไว 3 config (`max_features=0.3`, `min_samples_leaf=50`) → MAE เปลี่ยนแค่
−0.09% ถึง +0.34% (`results/rf_check_val_29feat.csv`) **ผลนั้นไม่ได้ถูกใช้เลือกค่า**

### XGBoost — จูนบน val 27 ชุด
```
XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.01,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
```
> ⚠️ **แก้จากเอกสารงาน:** เอกสาร `task_b_pre_test_tasks.md` ระบุว่า XGB คือ
> `max_depth=4, lr=0.05 (ไม่ได้จูน)` ซึ่งเป็นค่า**ก่อน**การจูนเมื่อ 2026-09-12
> ค่าที่ล็อกจริงคือค่าด้านบน และ XGB **ถูกจูนแล้ว** 27 ชุด
> (`results/xgb_sweep_val_29feat.csv`) เหตุผล: ANN ได้กวาด 140 ชุดแต่ XGB ได้ 1 ชุด
> การรายงานว่า "XGB แย่ที่สุด" จึงไม่เป็นธรรม — ใช้กฎการเลือกเดิมของ ANN ทุกตัวอักษร

**กฎการเลือกที่ใช้กับทั้ง ANN และ XGB (ประกาศก่อนดูผลทุกครั้ง):**
ในบรรดาชุดที่ `StdRatio` อยู่ใน [0.10, 0.20] ทั้งสองหุ้น เลือกตัวที่ `MAE_return`
เฉลี่ยต่ำสุด เสมอกันเลือกตัวที่เล็กกว่า — ถ้าไม่มีชุดไหนผ่านให้คงค่าเดิม

**จำนวน config ที่กวาดบน val (`SWEEP_HISTORY` ใน `config.py`):**
ANN 140 · XGBoost 27 · Random Forest 0

---

## SELECTED MODEL — เลือกจาก val ด้วย `MAE_return`

| หุ้น | โมเดลที่เลือก | val MAE_return |
|---|---|---|
| KBANK.BK | **Random Forest** | 0.008330 |
| ADVANC.BK | **ANN (MLP)** | 0.009436 |

ใช้ `MAE_return` ไม่ใช่ `MAE_baht` เพราะ `MAE_baht` คูณด้วยระดับราคาซึ่งไต่ขึ้นตลอด
ช่วงที่ประเมิน ทำให้วันท้ายชุดมีน้ำหนักมากกว่าวันต้นชุดโดยไม่มีเหตุผล

---

## ALTERNATIVE — ประกาศล่วงหน้า

**`Ensemble (1/3 each)`** = ค่าเฉลี่ยการทำนายของ ANN + RF + XGB น้ำหนัก 1/3 **ตายตัว**

- ห้ามหาน้ำหนักที่ดีที่สุดจาก val
- **รายงานทั้ง selected model และ ensemble เสมอ** ห้ามเลือกอันใดอันหนึ่งหลังเห็นผล
- ถ้อยคำที่ต้องใช้ในรายงาน (ห้ามเขียนสั้นกว่านี้):
  > Equal-weight ensemble รวมโมเดลโดยไม่ทำ hyperparameter หรือ weight tuning เพิ่มจาก
  > validation จึงไม่เพิ่ม selection bias รอบใหม่จากการเลือกน้ำหนัก แต่**ไม่ได้ลบ**
  > selection bias ที่เกิดจากการพัฒนาโมเดลต้นทางบน validation โดยเฉพาะ ANN ที่ผ่าน
  > การกวาด 140 ชุด เหตุผลที่ใช้ ensemble คือโมเดลทั้ง 3 ตัวมี error ที่ไม่เหมือนกัน
  > (RF ดีบน KBANK, ANN ดีบน ADVANC) การเฉลี่ยจึงลด variance ของการทำนาย ซึ่งเป็น
  > เหตุผลเชิงหลักการที่ประกาศได้ล่วงหน้าโดยไม่ต้องดูผล
- **ห้ามเขียนว่า** "Rho ของ ensemble อยู่ตรงกลางเสมอ" — ผิดทางคณิตศาสตร์ และข้อมูลจริงก็ค้าน:
  บน val ของ ADVANC สมาชิกได้ Rho +0.1757 / +0.1655 / +0.1644 แต่ ensemble ได้ **+0.1818**
  ซึ่งสูงกว่าสมาชิกทุกตัว (เพราะ correlation ของค่าเฉลี่ยขึ้นกับ covariance ระหว่างโมเดลด้วย)

---

## BASELINES

| baseline | นิยาม | ทำนายค่าคงที่ไหม |
|---|---|---|
| Naive (RW) | return = 0 (พรุ่งนี้เท่าวันนี้) — **คู่แข่งหลัก** | ใช่ |
| Mean Return | ค่าเฉลี่ย return ของ train | ใช่ |
| Always Up | `abs(mean return ของ train)` เป็นบวกเสมอ (ทิศทางมาจาก prior) | ใช่ |
| Rolling Mean 5d / 10d / 20d | `mean(R(t-k)...R(t-1))` | **ไม่ใช่** |

Rolling mean ใช้ข้อมูลถึงวัน t-1 เท่านั้น (`rolling(k).mean().shift(1)`) และมีการตรวจ
leak ด้วยการคำนวณมือเทียบทีละวัน (`verify_rolling_no_leak` ใน `baselines.py`)
เพราะไม่ใช่ตัวทำนายค่าคงที่ จึงมี `StdRatio` / `Rho` / `DirAcc` ให้เทียบกับโมเดลได้

---

## METRICS ที่ต้องมีครบในตาราง

`MAE_return` · `RMSE_return` · `R2_return` · `R2_OOS` · `StdRatio` · `Rho` ·
`Rho_lo` · `Rho_hi` · `Bias` · `DirAcc` · `FlatRate` · `MAE_baht` · `RMSE_baht` ·
`R2_price` · `DM_z` · `DM_p`

- `R2_OOS = 1 − SSE(โมเดล)/SSE(Naive)` เทียบกับคู่แข่งที่ประกาศไว้
  (ต่างจาก `R2_return` ที่เทียบกับค่าเฉลี่ยของชุดที่ประเมินเอง ซึ่ง ณ เวลาทำนายยังไม่รู้)
- `DM_z` / `DM_p` = Diebold-Mariano เทียบกับ Naive, loss = `|error ของ return|`,
  ความแปรปรวนแบบ Newey-West (HAC) lag = 5 · `z < 0` = โมเดลดีกว่า

---

## ERROR ANALYSIS แยกตามสภาวะตลาด

- แบ่งด้วย `volatility_20d_prev` เทียบ **median ของ train เท่านั้น**
  (KBANK = 0.014272 · ADVANC = 0.011683)
- และแบ่งตามทิศทางจริง: วันขึ้น (y > 0) / วันลง (y < 0) / วันนิ่ง (y = 0)
- **ห้ามใช้ median ของชุดที่ประเมิน** เพราะจะบังคับให้สองกลุ่มใหญ่เท่ากันเสมอ
  และทำให้เทียบ val กับ test ไม่ได้ (เส้นแบ่งคนละเส้น)
- ใช้ prediction ที่มีอยู่แล้ว ไม่มีการเทรนใหม่

### ⚠️ ข้อควรระวังสองข้อที่ต้องเขียนคู่กับตาราง regime เสมอ

1. **ขนาดกลุ่มไม่เท่ากัน โดยเฉพาะ KBANK** — บน val ได้ vol สูง 41 วัน / vol ต่ำ 321 วัน
   เพราะช่วง val ของ KBANK ผันผวนน้อยกว่า train มาก (SD ของ return บน val เหลือราว
   0.55 เท่าของ train และ `volatility_20d` เฉลี่ยต่ำกว่า train 0.66 SD)
   ตัวเลขในกลุ่ม vol สูงจึงเป็น **small sample** ส่วน ADVANC แบ่งได้ 189 / 173 ซึ่งสมดุลกว่า
   → บน test สัดส่วนจะเปลี่ยนอีก เพราะเส้นแบ่งตรึงไว้ที่ median ของ train (ตั้งใจให้เป็นแบบนั้น)

2. **ข้อสรุปเรื่อง "โมเดลช่วยตอน vol สูงหรือ vol ต่ำ" ไม่ทนต่อการเปลี่ยนเส้นแบ่ง**
   บน val ของ KBANK — ใช้ median ของ train: RF ชนะ Naive **+3.91%** (vol สูง) / **+0.08%** (vol ต่ำ)
   แต่ถ้าใช้ median ของ val: **+0.15%** / **+0.94%** ซึ่งกลับทิศกันคนละเรื่อง
   → **ห้ามเขียนเป็น "ข้อค้นพบ" ว่าสัญญาณกระจุกที่ vol สูงหรือ vol ต่ำ**
   ให้รายงานเป็นข้อสังเกตพร้อมระบุเส้นแบ่งที่ใช้เสมอ และบอกว่าผลพลิกได้เมื่อเปลี่ยนเส้นแบ่ง

---

## TEST POLICY

1. Test ถูกใช้ **ครั้งเดียว** — รัน `python main.py` (ไม่ใส่ `--dev`) หนึ่งครั้ง
2. **ไม่เลือกโมเดลหลังเห็น test** — ใช้ที่ล็อกไว้ข้างบน (KBANK = RF, ADVANC = ANN)
3. **ไม่จูนพารามิเตอร์ใด ๆ จาก test**
4. **ไม่ refit บน train+val** — คงหลัก "เทรนบน train เท่านั้น" ให้สม่ำเสมอ
   (ข้อเสียที่ยอมรับ: โมเดลไม่เห็นข้อมูล 1.5 ปีล่าสุดก่อน test = ประเมินแบบระมัดระวัง)
5. **รายงานทั้ง selected model และ ensemble** ไม่ cherry-pick
6. ถ้าผล test ออกมาไม่สวย → รายงานตามจริง ห้ามย้อนกลับไปแก้แล้วรันใหม่
7. diagnostics ใช้แถวของ train เสมอ ไม่มีสถิติของ test หลุดเข้ารายงาน

---

## คำทำนายก่อนเปิด TEST (เขียนก่อนเห็นผล — ใช้ตรวจว่าเราเข้าใจปัญหาจริงไหม)

1. **Rho บน test จะต่ำกว่าบน val โดยเฉพาะ ANN** (ผ่านการคัด 140 ชุด)
   ส่วน RF ที่ไม่เคยจูนจะลดน้อยกว่า
   *ค่าอ้างอิงบน val:* KBANK — ANN +0.0068 / RF +0.1218 / XGB +0.0909 / Ensemble +0.0956 ·
   ADVANC — ANN +0.1757 / RF +0.1655 / XGB +0.1644 / Ensemble +0.1818
2. **DM test จะให้ p > 0.05 ทุกโมเดล** = ไม่มีตัวไหนชนะ Naive อย่างมีนัยสำคัญ
3. **ผลต่าง MAE เทียบ Naive จะยังต่ำกว่าเกณฑ์ noise 2%**
4. **|Bias| ของ ADVANC จะยังเล็ก (< 0.15)** แม้ราคาช่วง test จะห่างจากกรอบ train
   มากกว่า val — ข้อนี้คือการทดสอบกลไก H1 **นอกกลุ่มตัวอย่าง** (วินิจฉัยบน val →
   ทำนายผลบน test) *ค่าอ้างอิงบน val:* ADVANC ANN −0.046 / RF −0.053 / XGB −0.045
5. **Rolling mean baselines จะยังแพ้ Naive อย่างมีนัยสำคัญ** (บน val แพ้ทุกตัว
   ทั้งสองหุ้น p ≤ 0.002) = วิธีที่ปรับตามข้อมูลล่าสุดแบบง่าย ๆ ไม่ได้ช่วย

---

## Checklist ก่อนเปิด test

- [x] Error analysis แยก regime (train median) — มีในรายงานและ csv แยกไฟล์
- [x] Rolling mean baselines 5/10/20 + ตรวจ leak ด้วยการคำนวณมือ
- [x] Equal-weight ensemble 1/3 + รายงานคู่กับ selected model เสมอ
- [x] `python main.py --dev` ผ่านทั้งสองหุ้น ไม่มี error
- [x] `verify_no_leak()` และ `leak_check()` ผ่าน
- [x] คอลัมน์ครบทุกตัวตาม METRICS (รวม `R2_OOS`, `DM_z`, `DM_p`)
- [x] ข้อความรายงานแยกตาม stage (ไม่พิมพ์ "test ยังไม่เปิด" ตอนรันบน test)
- [x] diagnostics ใช้ train เสมอทุกโหมด
- [x] `task_a/` และ `task_c/` ไม่ถูกแก้
- [ ] ไฟล์นี้ + `results/` ทั้งหมด commit แล้ว ← **ทำก่อนรัน test**
