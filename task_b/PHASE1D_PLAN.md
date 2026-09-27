# PHASE1D_PLAN — Mode A (ณ 16:00 ทำนาย close_bar16 ของวันเดียวกัน) · pre-registration

> เขียนใน Session 1 (2026-09-27) **ก่อน** fit โมเดลใด ๆ บน target จริงของ Phase 1D
> ต้องให้คน commit ไฟล์นี้ + `intraday_1600.py` + `tune_1600.py` + `metrics_1600.py` + ทุกโมดูลที่ import
> ก่อน Session 2 · `tune_1600.py --run` ปฏิเสธการรันถ้าไฟล์ที่ล็อกยังไม่ commit
> หลัง Session 2 ห้ามเปลี่ยน feature / grid / rule / metric — การเปลี่ยนใด ๆ = "รอบที่สอง" ต้องมีแผนใหม่ + commit ก่อน
> ไฟล์นี้ไม่แทนที่ `TUNING_PLAN.md` (daily model) และไม่แตะ daily pipeline

---

## 1. Task และสถานะการตัดสินใจ

**Mode A** — ณ 16:00 ของวัน t ทำนาย `close_bar16` ของวัน t

สถานะการตัดสินใจ: **ผู้ใช้ระบุ Mode A ใน instruction วันที่ 2026-09-27**
(ไม่มีหลักฐานใน repo ว่าอาจารย์อนุมัติ — ไม่อ้างการอนุมัติใด ๆ)

## 2. Time semantics (Asia/Bangkok, +07:00) — 3 สถานะแยกกัน

| | สถานะ |
|---|---|
| (1) bar semantics: timestamp = เวลาเริ่มแท่ง (15:00 = 15:00→16:00 · 16:00 = 16:00→ปิดตลาด) | **working assumption / unconfirmed** |
| (2) price alignment: `close_bar16` เทียบ Yahoo interval `1d` | **ยังไม่ได้ตรวจ** (Session 1 ไม่มีไฟล์ Yahoo 1d แนบมา) · ถ้าตรวจในอนาคต = price-alignment check เท่านั้น ไม่ใช่หลักฐานเรื่อง timestamp |
| (3) real-time availability: แท่ง 15:00 สมบูรณ์ ณ 16:00 จริงหรือไม่ (ความล่าช้าของ Yahoo) | **ยังไม่ได้พิสูจน์** |

- `close_bar16` ≠ official SET close (ตรงกับ Investing close เป๊ะแค่ 45.6% KBANK / 40.5% ADVANC ของวัน)
- historical dry-run ตรวจได้เพียง causal feature construction — ไม่ได้พิสูจน์ latency หรือการออก prediction ณ 16:00

เวลาที่บันทึกแยกกัน:

| field | ค่า |
|---|---|
| `last_feature_bar_start` | 15:00 ของวัน t |
| `feature_window_end` (= `data_cutoff_timestamp`) | 16:00 ของวัน t |
| `target_bar_start` | 16:00 ของวัน t |
| `snapshot_downloaded_at` | เวลาที่ได้ snapshot จริง (live เท่านั้น) |
| `generated_at` | เวลาที่ออก prediction จริง |

เงื่อนไข causal: ทุก feature bar มี `bar_start < target_bar_start` และ `feature_window_end ≤ target_bar_start`
(กรณีนี้เท่ากันที่ 16:00) · ห้ามใช้ `data_cutoff_timestamp < target_bar_start` แทน
→ `intraday_1600.check_causal()`

## 3. Data

- `raw_data_intraday/*.csv` (Yahoo) เท่านั้น · อ่านด้วย `utf-8-sig` · SHA-256 ต้องตรงค่าที่ล็อก
  (KBANK `b192ad2c…2060` · ADVANC `2097bb53…e79c`)
- **ห้ามอ่าน `raw_data/` (Investing)** ใน feature/target ของ Phase 1D ทุกกรณี (`intraday_1600` ไม่ import `data_loader`)
- ข้อมูล historical ทั้งหมด = development ที่เคย probe แล้ว · **ไม่มี historical test · ห้ามเรียกช่วงใดว่า test**

### Eligible days

- eligible labeled day: มีแท่ง 10, 11, 12, 14, 15 **และ** 16 · live day: มีแท่ง 10, 11, 12, 14, 15
- แท่ง 13: **ไม่อ่านเลย** (ไม่ใช่ optional) · วันไม่ครบ → ตัดทั้งวัน ห้าม ffill

ผลนับจากข้อมูลจริง (Session 1 — นับวัน ไม่ได้ดู target):

| | KBANK | ADVANC |
|---|---|---|
| วันในไฟล์ | 722 | 722 |
| eligible labeled days | **720** | **719** |
| วันที่ถูกตัด (แท่งที่ขาด) | 2025-03-28 [15, 16] · 2026-04-20 [10, 11, 12, 14, 15] | 2025-03-28 [15, 16] · **2025-09-16 [14]** · 2026-04-20 [10, 11, 12, 14, 15] |
| warm-up ที่ตัด (20 eligible rows แรก) | 2023-10-02 → 2023-10-31 | 2023-10-02 → 2023-10-31 |
| แถว NaN/Inf กลางชุดที่ตัด | ไม่มี | ไม่มี |
| **แถวหลังตัด (n)** | **700** (2023-11-01 → 2026-09-25) | **699** (2023-11-01 → 2026-09-25) |

## 4. Target

```
y(t) = C16(t) / C15(t) − 1          predicted_close_bar16 = C15(t) × (1 + ŷ)
```

## 5. Features (21 ตัว — ห้ามอ่านแท่ง 16 ของวัน t · ห้ามอ่าน Volume ของแท่ง 10)

| กลุ่ม | features |
|---|---|
| Returns (5) | C11/C10−1 · C12/C11−1 · C14/C12−1 · C15/C14−1 · C15/O10−1 |
| State (3) | range = (max H − min L ของแท่ง 10,11,12,14,15) / C15 · pos = (C15 − min L)/(max H − min L) (ถ้า max H = min L → 0.5) · gap = O10(t)/C16(t_prev) − 1 |
| Volume (4) | Vcum = V11+V12+V14+V15 · share12 = V12/Vcum · share14 = V14/Vcum · share15 = V15/Vcum · relvol = log(Vcum(t) / mean(Vcum ของ 20 eligible days ก่อนหน้า ไม่รวม t)) |
| Target lags (4) | y(t−1), y(t−2), y(t−3), y(t−5) |
| Calendar (5) | one-hot day-of-week ของวัน t (ไม่ shift) |

- ห้ามใช้ raw volume level (ไม่ stationary / tree extrapolate ไม่ได้ — บทเรียน H1) · Vcum = 0 → assert fail (ข้อมูลจริง: ไม่มี)
- lag / rolling / t_prev นับตามลำดับ **eligible labeled days** ของหุ้นนั้น ก่อน drop warm-up
  t_prev = eligible labeled day ก่อนหน้า (ข้ามวันที่ไม่ eligible) · นิยามเดียวกันทั้ง historical และ live builder
- Warm-up: 20 eligible rows แรกของแต่ละหุ้น (จาก relvol; ครอบคลุม lag 5) → ตัดทิ้ง ไม่ impute
- NaN/Inf กลางชุด → ตัดแถวและรายงานวันที่

วันที่ t_prev ห่างเกิน 1 วันทำการ (จ.–ศ.) — **รายงานเท่านั้น ไม่ตัดทิ้ง**
ส่วนใหญ่เป็นวันหลังวันหยุดตลาด · ที่เกิดจากวันที่ถูกตัด: 2025-03-31 และ 2026-04-21 (ทั้งสองหุ้น), 2025-09-17 (ADVANC)

- KBANK (46 วัน): 2023-10-16, 2023-10-24, 2023-12-06, 2023-12-12, 2024-01-03, 2024-02-27, 2024-04-09,
  2024-04-17, 2024-05-02, 2024-05-07, 2024-05-23, 2024-06-04, 2024-07-23, 2024-07-30, 2024-08-13, 2024-10-15,
  2024-10-24, 2024-12-06, 2024-12-11, 2025-01-02, 2025-02-13, 2025-03-31, 2025-04-08, 2025-04-16, 2025-05-02,
  2025-05-06, 2025-05-13, 2025-06-04, 2025-07-11, 2025-07-29, 2025-08-13, 2025-10-14, 2025-10-24, 2025-12-08,
  2025-12-11, 2026-01-05, 2026-02-03, 2026-03-04, 2026-04-07, 2026-04-16, 2026-04-21, 2026-05-05, 2026-06-02,
  2026-06-04, 2026-07-30, 2026-08-13
- ADVANC (47 วัน): เหมือน KBANK + 2025-09-17

## 6. Baselines

- Naive: ŷ = 0 → close_bar16 = C15
- Mean Return: mean ของ y ใน train fold เท่านั้น
- baselines เป็น reference ไม่ถูกตัดด้วย validity guard

## 7. Models

ANN (MLP), Random Forest, XGBoost · fit แยกต่อหุ้น · `TransformedTargetRegressor(StandardScaler)`
+ preprocessing pipeline เดียวกับ daily `models.py` (`_scaled` = imputer + scaler สำหรับ ANN ·
`_unscaled` = imputer สำหรับ tree) · **ไม่มี recency weighting**

ค่าคงที่ที่ไม่อยู่ใน grid:

| | |
|---|---|
| ANN | activation relu, adam, learning_rate_init 1e-3, max_iter 3000, n_iter_no_change 20, early_stopping False |
| RF | n_estimators 400, n_jobs −1, random_state 42 |
| XGB | n_estimators 300, colsample_bytree 0.8, reg_lambda 1.0, random_state 42, n_jobs −1 |

ANN seeds (ล็อก):

| ensemble | seeds | ใช้ทำอะไร |
|---|---|---|
| search / evaluation | [0, 1, 2] | เลือก config + ตัดสิน validity |
| live / dry-run | [0, 1, 2] | โมเดลที่ deploy = โมเดลที่ผ่าน selection/guard |
| sensitivity | range(10) | refit ผู้ชนะบน 6 evaluation เดิม รายงานแยก — ไม่เปลี่ยน winner / guard / live seeds |

ทุก seed fit บน train fold เดียวกันแล้ว **เฉลี่ย predictions** (`SeedAveragedRegressor`) —
ห้ามเฉลี่ย metric ของแต่ละ seed แทน metric ของ averaged prediction

## 8. Grid (24 searched configs) · ชุดเดียวต่อ model family ใช้ทั้งสองหุ้น

```python
ANN 8: alpha [1, 3, 7, 20] × hidden [(8,), (8, 4)]
RF  8: max_depth [3, 5] × min_samples_leaf [15, 30] × max_features [0.5, 1.0]
XGB 8: max_depth [2, 3] × learning_rate [0.01, 0.05] × subsample [0.8, 1.0]
```

## 9. Walk-forward

`splits.walk_forward_splits(X, n_splits=3, min_train=360)` บนแถวของแต่ละหุ้น (ไม่แก้ splitter)
fold_size = (n − 360) // 3 → แถวเศษท้ายชุดไม่ถูกประเมิน

| หุ้น | fold | train | eval |
|---|---|---|---|
| KBANK (n=700, fold 113) | 1 | 360 แถว 2023-11-01 → 2025-04-25 | 113 แถว 2025-04-28 → 2025-10-15 |
| | 2 | 473 แถว 2023-11-01 → 2025-10-15 | 113 แถว 2025-10-16 → 2026-04-02 |
| | 3 | 586 แถว 2023-11-01 → 2026-04-02 | 113 แถว 2026-04-03 → 2026-09-24 |
| ADVANC (n=699, fold 113) | 1 | 360 แถว 2023-11-01 → 2025-04-25 | 113 แถว 2025-04-28 → 2025-10-16 |
| | 2 | 473 แถว 2023-11-01 → 2025-10-16 | 113 แถว 2025-10-17 → 2026-04-03 |
| | 3 | 586 แถว 2023-11-01 → 2026-04-03 | 113 แถว 2026-04-07 → 2026-09-25 |

**Omitted evaluation dates:** KBANK 2026-09-25 · ADVANC ไม่มี
(ไม่ใช่ holdout/test · ใช้ใน live training ได้เพราะเป็นอดีตที่มี label แล้ว) · `tune_1600.py` พิมพ์ทุกครั้ง

- ในแต่ละ fold: fit model + feature scaler + target scaler จาก train fold เท่านั้น
- คง fitted model ตลอด eval fold · ไม่มี refit ภายใน fold · ไม่ shuffle
- การประเมิน = sequential daily prediction: X(t) ใช้ข้อมูลที่ทราบก่อน cutoff ของวัน t
  ซึ่งรวม outcome ของวันก่อนหน้าใน eval fold ได้ (ผ่าน lag/gap) · ห้ามใช้ outcome ของวัน t หรืออนาคต

## 10. Primary score

```
relMAE = MAE_return(model) / MAE_return(Naive)     บน eval window เดียวกัน
primary_score = mean relMAE ของ 6 evaluation (2 หุ้น × 3 folds) น้ำหนักเท่ากัน · ต่ำสุดชนะ
```

เหตุผล: naive MAE ของ ADVANC สูงกว่า KBANK ~25% ถ้าเฉลี่ย MAE ดิบ ADVANC จะคุมผลการเลือก
· รายงาน mean MAE_return ดิบคู่กันเสมอ

## 11. Validity guard (ไม่ใช่ ranking)

StdRatio finite และ ≥ 0.05 ทุก 6 evaluation · ห้ามลดเกณฑ์หลังเห็นผล
ถ้าทุก config ของ family ใดไม่ผ่าน → family นั้น "ไม่มี valid config" รายงานเป็นข้อค้นพบ
→ ไม่มี winner / ไม่มี live model ของ family นั้น · family อื่นทำต่อ

## 12. Tie rule

ในกลุ่มที่ผ่าน guard, |score − best| < 1e-3 (หน่วย relMAE = 0.1% ของ Naive MAE —
เทียบเท่าเชิงสัดส่วนกับ 1e-5 / 0.0092 ของ Phase 1C) = เสมอ

- ANN: จำนวน neuron รวมน้อยกว่า แล้ว alpha สูงกว่า
- RF / XGB: max_depth ตื้นกว่า แล้ว score ต่ำกว่า

## 13. NaN / Inf / zero-denominator policy (ล็อกก่อนรัน)

| กรณี | ผล |
|---|---|
| Naive MAE = 0 ใน evaluation ใด | relMAE นิยามไม่ได้ → **หยุด ranking ทั้งหมด** และรายงาน (dataset-level undefined score) · ห้ามเติม epsilon / ข้าม fold |
| config มี evaluation ไม่ครบ 6 หรือ relMAE non-finite | ห้าม nanmean · ขาด evaluation = bug → STOP |
| prediction non-finite แม้แถวเดียว | config invalid (รายงาน) |
| std(y_true) = 0 | StdRatio = NaN → guard ไม่ผ่าน (ไม่แทนด้วย 0) |
| MSE_Naive = 0 | R2_OOS = NaN |
| truth หรือ prediction คงที่ | Rho = NaN |
| truth คงที่ | R2_return = NaN (ใช้ policy เดียวทุก model/baseline) |

→ `metrics_1600.py` (`RankingUndefined`, `IncompleteEvaluations`, `passes_guard`, `primary_score`)

## 14. Metrics

- main: MAE_return, RMSE_return, R2_return, MAE_baht, RMSE_baht (บาทคำนวณจาก C15)
- skill: relMAE vs Naive, R2_OOS = 1 − MSE/MSE_Naive
- diagnostics: StdRatio, Bias, Rho, สัดส่วนวันที่ y = 0 ใน eval window

**ข้อความบังคับในรายงาน:** ~43–49% ของวันราคาไม่ขยับเลย ⇒ MAE เอื้อ prediction ที่ใกล้ 0 ·
StdRatio ต่ำของผู้ชนะเป็นสิ่งที่คาดได้ ไม่ใช่หลักฐานของ skill · **ห้ามวางตัวเลข Phase 1D ในตารางเดียวกับ daily model**

## 15. Economic diagnostics (two-sided magnitude diagnostic — ไม่ใช่ backtest, ไม่ใช่จำนวนโอกาส long-only)

cost = `cost_round_trip()` (สัดส่วน) · tick = `tick_size(C15)` (บาท) · ต่อวัน:

| | นิยาม |
|---|---|
| actual return magnitude | abs(y) |
| predicted return magnitude | abs(ŷ) |
| actual move (บาท) | abs(y) × C15 |
| predicted move (บาท) | abs(ŷ) × C15 |
| actual ≥ cost | abs(y) ≥ cost |
| actual ≥ 1 tick | abs(y) × C15 ≥ tick |
| predicted > cost | abs(ŷ) > cost |

- ห้ามเทียบ return กับ tick (บาท) โดยตรง
- รายงานสัดส่วน (0–1 ใน CSV; % ในรายงานพร้อมระบุหน่วย) ของสามเงื่อนไข
- signal_days = จำนวนวัน predicted > cost · sign hit rate = mean(sign(ŷ) == sign(y)) บน signal_days
  (y = 0 ไม่นับเป็น hit) · รายงานจำนวน signal days ที่ y = 0 แยก · signal_days = 0 → hit rate = NaN
- ห้ามใช้ `signal_economics()` (ออกแบบให้ถือข้ามวันต่อเนื่อง → นับต้นทุนต่ำไปสำหรับ round-trip ภายในวัน)
- ห้ามคำนวณ PnL ที่ราคา C15 (ราคาที่ซื้อได้จริงเกิดหลัง prediction timestamp + ความล่าช้า + spread)
- แยก forecast quality ออกจาก executability / profitability เสมอ

## 16. Live pipeline (dry-run เท่านั้นจนกว่า human freeze)

`intraday_1600.build_1600_live_feature(bars, target_date)` → X_live, close_bar15, time fields, train_days (+ ชุดเทรน)

1. ตัดข้อมูลทุกวันหลัง target_date ทิ้ง
2. ตัดแท่งของ target_date ที่ timestamp ≥ 16:00 ทิ้ง
3. ต้องมีแท่ง 10, 11, 12, 14, 15 ของ target_date ไม่งั้น raise
4. สร้าง features ด้วยนิยามเดียวกับ historical
5. fit ด้วย labeled feature rows ที่พร้อมใช้ (หลังตัด warm-up/invalid) และวันที่ < target_date เท่านั้น ·
   train_days = จำนวนแถวที่ใช้ fit จริง · ANN ใช้ seeds [0, 1, 2] (`tune_1600.make_live_model`)

- live data ในอนาคต: snapshot ใหม่เก็บใน `raw_data_intraday_live/` พร้อม SHA-256 + `snapshot_downloaded_at`
- ห้ามแก้ไฟล์ `raw_data_intraday/` ที่ล็อก SHA · ตรวจว่าแท่งเก่าใน snapshot ตรงกับไฟล์ล็อก (รายงานถ้าต่าง)
- ต้องออก prediction ก่อน 16:30 (pre-close) และบันทึก `generated_at` จริง

## 17. Logging

- official `same_day_1600` **ยังไม่เปิด** ใน phase นี้ (`predict_live.py` ยังมี `choices=["next_day"]`) ·
  dry-run เขียน `results/dryrun/` เท่านั้น
- เมื่อเปิดในอนาคต: ใช้ schema `prediction_log.csv` เดิม · `prediction_type = same_day_1600` ·
  `prev_close := close_bar15` · `data_cutoff := feature_window_end` ("YYYY-MM-DDT16:00:00+07:00") ·
  `dataset_sha256 :=` SHA-256 เต็มของ live snapshot · `code_commit :=` HEAD ที่ tree สะอาด
- `generated_at` มีช่องอยู่แล้ว · `snapshot_downloaded_at` **ไม่มีช่องใน schema เดิม** →
  เก็บใน dry-run/sidecar และให้คนตัดสินใจ (ห้ามแก้ schema ของ production log เอง)

## 18. Prospective evaluation

ยังไม่เปิดใน Session 1–2 · เริ่มได้หลัง human freeze commit **และ** หลังพิสูจน์ real-time availability แล้วเท่านั้น
· outcome `close_bar16` จาก Yahoo download ภายหลัง พร้อมเวลาดาวน์โหลดและ SHA ·
freeze candidate ไม่ใช่หลักฐานว่า real-time pipeline พร้อมใช้งาน

## 19. Freeze

หลัง Session 2 ห้ามเปลี่ยน feature / grid / rule / metric · การเปลี่ยนใด ๆ = "รอบที่สอง" ต้องมีแผนใหม่ + commit ก่อน

## 20. Implementation ที่ล็อกพร้อมแผนนี้

| ไฟล์ | หน้าที่ |
|---|---|
| `intraday_1600.py` | อ่าน/ตรวจ SHA · eligible days · 21 features · target · warm-up · live builder · causal check |
| `metrics_1600.py` | main/skill metrics · shape · guard · primary score · NaN policy · economic diagnostics |
| `tune_1600.py` | model factory · walk-forward · search 24 configs · tie rule · sensitivity 10 seeds · outputs (`results/phase1d/`) · `--smoke` (สังเคราะห์) · `--run` (ต้อง commit ก่อน) |
| `tests/test_phase1d.py` | 21 tests (ข้อมูลสังเคราะห์) |
| import ที่พึ่ง | `models.py` (SeedAveragedRegressor, _scaled, _unscaled), `splits.py` (walk_forward_splits), `trading_costs.py` (+ `config.py`), `weighting.py` (import โดย models.py) |
