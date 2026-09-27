# Phase 1D (Mode A 16:00) — development / walk-forward — not a test

> สร้างโดย `python report_1600.py` จากผลของ `python tune_1600.py --run` (รันครั้งเดียว)
> ข้อมูลทั้งหมดเป็น development data ที่เคย probe แล้ว · **ไม่มี historical test** · prospective evaluation ยังไม่เปิด
> **ห้ามเทียบในตารางเดียวกับ daily model** (คนละโจทย์ คนละแหล่งข้อมูล คนละ horizon)

**ข้อความบังคับ:** ~43–49% ของวันราคาไม่ขยับเลย (close_bar16 = close_bar15) ⇒ MAE เอื้อ prediction ที่ใกล้ 0 (shrinkage) · StdRatio ต่ำของผู้ชนะเป็นสิ่งที่คาดได้ **ไม่ใช่หลักฐานของ skill**

## 1. Fold boundaries และ omitted evaluation dates

| ticker | fold | train_rows | train_start | train_end | eval_rows | eval_start | eval_end |
|---|---|---|---|---|---|---|---|
| KBANK.BK | 1 | 360 | 2023-11-01 | 2025-04-25 | 113 | 2025-04-28 | 2025-10-15 |
| KBANK.BK | 2 | 473 | 2023-11-01 | 2025-10-15 | 113 | 2025-10-16 | 2026-04-02 |
| KBANK.BK | 3 | 586 | 2023-11-01 | 2026-04-02 | 113 | 2026-04-03 | 2026-09-24 |
| ADVANC.BK | 1 | 360 | 2023-11-01 | 2025-04-25 | 113 | 2025-04-28 | 2025-10-16 |
| ADVANC.BK | 2 | 473 | 2023-11-01 | 2025-10-16 | 113 | 2025-10-17 | 2026-04-03 |
| ADVANC.BK | 3 | 586 | 2023-11-01 | 2026-04-03 | 113 | 2026-04-07 | 2026-09-25 |

Omitted evaluation dates: KBANK.BK 2026-09-25 · ADVANC.BK ไม่มี (ไม่ใช่ holdout/test · ใช้เทรน live ได้)

## 2. Selection (development / walk-forward — not a test)

primary_score = mean relMAE (MAE_model / MAE_Naive) ของ 6 evaluation · ANN จัดอันดับด้วย 3 seeds · guard = StdRatio finite และ >= 0.05 ทุก evaluation · tie = ต่างจากค่าต่ำสุด < 1e-3

- **ANN (MLP)** ผู้ชนะ `{'alpha': 20, 'hidden_layer_sizes': (8, 4)}` · primary_score 1.057462 · mean MAE_return 0.002229 · min StdRatio 0.1309
  - ค่าที่แพ้ใกล้สุด `{'alpha': 20, 'hidden_layer_sizes': (8,)}` = 1.066243 (ต่าง +0.008782 — เกิน tie 1e-3 จึงไม่ใช่เสมอ)
- **Random Forest** ผู้ชนะ `{'max_depth': 3, 'min_samples_leaf': 15, 'max_features': 0.5}` · primary_score 1.051282 · mean MAE_return 0.002218 · min StdRatio 0.1997
  - ค่าที่แพ้ใกล้สุด `{'max_depth': 3, 'min_samples_leaf': 30, 'max_features': 0.5}` = 1.053607 (ต่าง +0.002326 — เกิน tie 1e-3 จึงไม่ใช่เสมอ)
- **XGBoost** ผู้ชนะ `{'max_depth': 2, 'learning_rate': 0.01, 'subsample': 1.0}` · primary_score 1.054307 · mean MAE_return 0.002222 · min StdRatio 0.1853
  - ค่าที่แพ้ใกล้สุด `{'max_depth': 2, 'learning_rate': 0.01, 'subsample': 0.8}` = 1.056248 (ต่าง +0.001941 — เกิน tie 1e-3 จึงไม่ใช่เสมอ)

- searched configs: 24 · valid (prediction finite ทุกแถว): 24 · ผ่าน guard: 24 · config ที่ primary_score < 1 (ดีกว่า Naive โดยเฉลี่ย): **0**
- NaN/Inf events: ไม่มี prediction non-finite · Naive MAE > 0 ทุก evaluation (ต่ำสุด 0.001789) · ทุก config มีครบ 6 evaluations

## 3. Development metrics ต่อหุ้น × fold (development / walk-forward — not a test)

### ADVANC.BK (development / walk-forward — not a test)

| model | ticker | fold | MAE_return | RMSE_return | R2_return | MAE_baht | RMSE_baht | relMAE | R2_OOS | StdRatio | Bias | Rho | frac_y_zero |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ANN (MLP) (winner) | ADVANC.BK | 1 | 0.002663 | 0.003596 | 0.037290 | 0.776023 | 1.051321 | 1.030754 | 0.038112 | 0.207855 | 0.000404 | 0.222800 | 41.59% |
| Baseline: Mean Return | ADVANC.BK | 1 | 0.002767 | 0.003682 | -0.009266 | 0.805824 | 1.074055 | 1.070876 | -0.008405 | 0.000000 | 0.000353 | NaN | 41.59% |
| Baseline: Naive | ADVANC.BK | 1 | 0.002584 | 0.003666 | -0.000854 | 0.752212 | 1.068454 | 1.000000 | 0.000000 | 0.000000 | -0.000107 | NaN | 41.59% |
| Random Forest (winner) | ADVANC.BK | 1 | 0.002684 | 0.003623 | 0.022762 | 0.782036 | 1.058717 | 1.038798 | 0.023596 | 0.201303 | 0.000241 | 0.167920 | 41.59% |
| XGBoost (winner) | ADVANC.BK | 1 | 0.002654 | 0.003600 | 0.035206 | 0.773321 | 1.051735 | 1.027114 | 0.036029 | 0.185280 | 0.000194 | 0.195212 | 41.59% |
| ANN (MLP) (winner) | ADVANC.BK | 2 | 0.002272 | 0.003064 | 0.006732 | 0.765385 | 1.033177 | 1.053509 | 0.012948 | 0.130912 | 0.000073 | 0.093324 | 42.48% |
| Baseline: Mean Return | ADVANC.BK | 2 | 0.002293 | 0.003077 | -0.001833 | 0.772063 | 1.035332 | 1.063188 | 0.004436 | 0.000000 | 0.000132 | NaN | 42.48% |
| Baseline: Naive | ADVANC.BK | 2 | 0.002157 | 0.003084 | -0.006297 | 0.725664 | 1.039060 | 1.000000 | 0.000000 | 0.000000 | -0.000244 | NaN | 42.48% |
| Random Forest (winner) | ADVANC.BK | 2 | 0.002318 | 0.003090 | -0.010056 | 0.782566 | 1.044126 | 1.074623 | -0.003735 | 0.210807 | -0.000103 | 0.084201 | 42.48% |
| XGBoost (winner) | ADVANC.BK | 2 | 0.002335 | 0.003184 | -0.072671 | 0.787393 | 1.073503 | 1.082892 | -0.065959 | 0.258476 | -0.000056 | -0.010695 | 42.48% |
| ANN (MLP) (winner) | ADVANC.BK | 3 | 0.001995 | 0.002697 | 0.034750 | 0.718298 | 0.971074 | 1.039953 | 0.049461 | 0.188274 | 0.000606 | 0.315944 | 45.13% |
| Baseline: Mean Return | ADVANC.BK | 3 | 0.002095 | 0.002831 | -0.063495 | 0.754144 | 1.019109 | 1.092060 | -0.047288 | 0.000000 | 0.000692 | NaN | 45.13% |
| Baseline: Naive | ADVANC.BK | 3 | 0.001919 | 0.002766 | -0.015476 | 0.690265 | 0.995565 | 1.000000 | 0.000000 | 0.000000 | 0.000341 | NaN | 45.13% |
| Random Forest (winner) | ADVANC.BK | 3 | 0.001977 | 0.002702 | 0.030807 | 0.711738 | 0.973246 | 1.030395 | 0.045578 | 0.199714 | 0.000429 | 0.238094 | 45.13% |
| XGBoost (winner) | ADVANC.BK | 3 | 0.001970 | 0.002703 | 0.029988 | 0.709151 | 0.973695 | 1.026588 | 0.044771 | 0.212949 | 0.000474 | 0.246981 | 45.13% |

### KBANK.BK (development / walk-forward — not a test)

| model | ticker | fold | MAE_return | RMSE_return | R2_return | MAE_baht | RMSE_baht | relMAE | R2_OOS | StdRatio | Bias | Rho | frac_y_zero |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ANN (MLP) (winner) | KBANK.BK | 1 | 0.002174 | 0.002902 | 0.035916 | 0.351528 | 0.469378 | 1.075342 | 0.037004 | 0.285153 | 0.000157 | 0.210522 | 44.25% |
| Baseline: Mean Return | KBANK.BK | 1 | 0.002166 | 0.002964 | -0.005537 | 0.350634 | 0.480387 | 1.071265 | -0.004402 | 0.000000 | 0.000220 | NaN | 44.25% |
| Baseline: Naive | KBANK.BK | 1 | 0.002022 | 0.002957 | -0.001130 | 0.327434 | 0.479675 | 1.000000 | 0.000000 | 0.000000 | -0.000099 | NaN | 44.25% |
| Random Forest (winner) | KBANK.BK | 1 | 0.002091 | 0.002910 | 0.030799 | 0.338516 | 0.472091 | 1.034155 | 0.031893 | 0.222408 | -0.000052 | 0.181144 | 44.25% |
| XGBoost (winner) | KBANK.BK | 1 | 0.002069 | 0.002871 | 0.056043 | 0.334887 | 0.465612 | 1.023311 | 0.057108 | 0.232964 | -0.000048 | 0.237336 | 44.25% |
| ANN (MLP) (winner) | KBANK.BK | 2 | 0.001920 | 0.002551 | 0.037429 | 0.364999 | 0.487735 | 1.073499 | 0.043365 | 0.200871 | 0.000007 | 0.193617 | 45.13% |
| Baseline: Mean Return | KBANK.BK | 2 | 0.001900 | 0.002601 | -0.000566 | 0.361652 | 0.498399 | 1.062026 | 0.005604 | 0.000000 | 0.000062 | NaN | 45.13% |
| Baseline: Naive | KBANK.BK | 2 | 0.001789 | 0.002608 | -0.006205 | 0.340708 | 0.500000 | 1.000000 | 0.000000 | 0.000000 | -0.000205 | NaN | 45.13% |
| Random Forest (winner) | KBANK.BK | 2 | 0.001908 | 0.002528 | 0.054884 | 0.362688 | 0.483603 | 1.066659 | 0.060713 | 0.281148 | -0.000115 | 0.241655 | 45.13% |
| XGBoost (winner) | KBANK.BK | 2 | 0.001968 | 0.002561 | 0.030019 | 0.374014 | 0.490040 | 1.100163 | 0.036001 | 0.373370 | 0.000019 | 0.226956 | 45.13% |
| ANN (MLP) (winner) | KBANK.BK | 3 | 0.002349 | 0.003325 | 0.063338 | 0.527626 | 0.759289 | 1.071714 | 0.069579 | 0.190819 | -0.000189 | 0.269270 | 50.44% |
| Baseline: Mean Return | KBANK.BK | 3 | 0.002307 | 0.003436 | -0.000060 | 0.517294 | 0.783906 | 1.052461 | 0.006603 | 0.000000 | -0.000027 | NaN | 50.44% |
| Baseline: Naive | KBANK.BK | 3 | 0.002192 | 0.003447 | -0.006707 | 0.491150 | 0.785657 | 1.000000 | 0.000000 | 0.000000 | -0.000281 | NaN | 50.44% |
| Random Forest (winner) | KBANK.BK | 3 | 0.002330 | 0.003284 | 0.086225 | 0.524373 | 0.750143 | 1.063059 | 0.092313 | 0.240358 | -0.000199 | 0.306546 | 50.44% |
| XGBoost (winner) | KBANK.BK | 3 | 0.002336 | 0.003300 | 0.077401 | 0.524223 | 0.751356 | 1.065775 | 0.083548 | 0.254356 | -0.000185 | 0.285018 | 50.44% |

### ค่าเฉลี่ย 6 evaluation (development / walk-forward — not a test)

| model | MAE_return | relMAE | R2_OOS | StdRatio |
|---|---|---|---|---|
| ANN (MLP) (winner) | 0.002229 | 1.057462 | 0.041745 | 0.200647 |
| Baseline: Mean Return | 0.002255 | 1.068646 | -0.007242 | 0.000000 |
| Baseline: Naive | 0.002110 | 1.000000 | 0.000000 | 0.000000 |
| Random Forest (winner) | 0.002218 | 1.051282 | 0.041726 | 0.225956 |
| XGBoost (winner) | 0.002222 | 1.054307 | 0.031916 | 0.252899 |

relMAE > 1 = แย่กว่า Naive (close_bar16 = close_bar15) · R2_OOS < 0 = MSE แย่กว่า Naive

## 4. ANN 3 seeds vs 10 seeds (sensitivity — ไม่เปลี่ยน winner / guard / live seeds) (development / walk-forward — not a test)

| ticker | fold | relMAE_3 | relMAE_10 | StdRatio_3 | StdRatio_10 |
|---|---|---|---|---|---|
| KBANK.BK | 1 | 1.075342 | 1.073900 | 0.285153 | 0.279113 |
| KBANK.BK | 2 | 1.073499 | 1.066565 | 0.200871 | 0.178949 |
| KBANK.BK | 3 | 1.071714 | 1.067201 | 0.190819 | 0.184113 |
| ADVANC.BK | 1 | 1.030754 | 1.027578 | 0.207855 | 0.185361 |
| ADVANC.BK | 2 | 1.053509 | 1.052183 | 0.130912 | 0.108133 |
| ADVANC.BK | 3 | 1.039953 | 1.040469 | 0.188274 | 0.168411 |

- mean relMAE: 3 seeds 1.057462 · 10 seeds 1.054649 · min StdRatio: 3 seeds 0.1309 · 10 seeds 0.1081

## 5. Economic diagnostics — two-sided magnitude diagnostic, ไม่ใช่ backtest (development / walk-forward — not a test)

cost = cost_round_trip() (สัดส่วน) · tick = tick_size(C15) (บาท) · สัดส่วนแสดงเป็น % ของวันใน eval window ·
hit rate = sign(ŷ)=sign(y) บน signal days (y = 0 ไม่นับ hit) · ไม่มีการคำนวณ PnL ที่ C15

| model | ticker | fold | frac_actual_ge_cost | frac_actual_ge_1tick | frac_pred_gt_cost | signal_days | sign_hit_rate_on_signal | signal_days_y_zero | mean_actual_move_baht | mean_pred_move_baht |
|---|---|---|---|---|---|---|---|---|---|---|
| ANN (MLP) (winner) | ADVANC.BK | 1 | 54.87% | 38.05% | 0.00% | 0 | NaN | 0 | 0.752212 | 0.214653 |
| Baseline: Mean Return | ADVANC.BK | 1 | 54.87% | 38.05% | 0.00% | 0 | NaN | 0 | 0.752212 | 0.133756 |
| Baseline: Naive | ADVANC.BK | 1 | 54.87% | 38.05% | 0.00% | 0 | NaN | 0 | 0.752212 | 0.000000 |
| Random Forest (winner) | ADVANC.BK | 1 | 54.87% | 38.05% | 0.00% | 0 | NaN | 0 | 0.752212 | 0.176604 |
| XGBoost (winner) | ADVANC.BK | 1 | 54.87% | 38.05% | 0.00% | 0 | NaN | 0 | 0.752212 | 0.167787 |
| ANN (MLP) (winner) | ADVANC.BK | 2 | 12.39% | 31.86% | 0.00% | 0 | NaN | 0 | 0.725664 | 0.135039 |
| Baseline: Mean Return | ADVANC.BK | 2 | 12.39% | 31.86% | 0.00% | 0 | NaN | 0 | 0.725664 | 0.127484 |
| Baseline: Naive | ADVANC.BK | 2 | 12.39% | 31.86% | 0.00% | 0 | NaN | 0 | 0.725664 | 0.000000 |
| Random Forest (winner) | ADVANC.BK | 2 | 12.39% | 31.86% | 0.00% | 0 | NaN | 0 | 0.725664 | 0.168655 |
| XGBoost (winner) | ADVANC.BK | 2 | 12.39% | 31.86% | 0.88% | 1 | 0.00% | 0 | 0.725664 | 0.177334 |
| ANN (MLP) (winner) | ADVANC.BK | 3 | 13.27% | 30.09% | 0.00% | 0 | NaN | 0 | 0.690265 | 0.166484 |
| Baseline: Mean Return | ADVANC.BK | 3 | 13.27% | 30.09% | 0.00% | 0 | NaN | 0 | 0.690265 | 0.126294 |
| Baseline: Naive | ADVANC.BK | 3 | 13.27% | 30.09% | 0.00% | 0 | NaN | 0 | 0.690265 | 0.000000 |
| Random Forest (winner) | ADVANC.BK | 3 | 13.27% | 30.09% | 0.00% | 0 | NaN | 0 | 0.690265 | 0.151604 |
| XGBoost (winner) | ADVANC.BK | 3 | 13.27% | 30.09% | 0.00% | 0 | NaN | 0 | 0.690265 | 0.165656 |
| ANN (MLP) (winner) | KBANK.BK | 1 | 7.08% | 26.55% | 0.00% | 0 | NaN | 0 | 0.327434 | 0.115943 |
| Baseline: Mean Return | KBANK.BK | 1 | 7.08% | 26.55% | 0.00% | 0 | NaN | 0 | 0.327434 | 0.051673 |
| Baseline: Naive | KBANK.BK | 1 | 7.08% | 26.55% | 0.00% | 0 | NaN | 0 | 0.327434 | 0.000000 |
| Random Forest (winner) | KBANK.BK | 1 | 7.08% | 26.55% | 0.00% | 0 | NaN | 0 | 0.327434 | 0.079632 |
| XGBoost (winner) | KBANK.BK | 1 | 7.08% | 26.55% | 0.00% | 0 | NaN | 0 | 0.327434 | 0.082725 |
| ANN (MLP) (winner) | KBANK.BK | 2 | 11.50% | 31.86% | 0.00% | 0 | NaN | 0 | 0.340708 | 0.085117 |
| Baseline: Mean Return | KBANK.BK | 2 | 11.50% | 31.86% | 0.00% | 0 | NaN | 0 | 0.340708 | 0.050609 |
| Baseline: Naive | KBANK.BK | 2 | 11.50% | 31.86% | 0.00% | 0 | NaN | 0 | 0.340708 | 0.000000 |
| Random Forest (winner) | KBANK.BK | 2 | 11.50% | 31.86% | 0.00% | 0 | NaN | 0 | 0.340708 | 0.103248 |
| XGBoost (winner) | KBANK.BK | 2 | 11.50% | 31.86% | 0.88% | 1 | 100.00% | 0 | 0.340708 | 0.133532 |
| ANN (MLP) (winner) | KBANK.BK | 3 | 36.28% | 27.43% | 0.00% | 0 | NaN | 0 | 0.491150 | 0.122308 |
| Baseline: Mean Return | KBANK.BK | 3 | 36.28% | 27.43% | 0.00% | 0 | NaN | 0 | 0.491150 | 0.056661 |
| Baseline: Naive | KBANK.BK | 3 | 36.28% | 27.43% | 0.00% | 0 | NaN | 0 | 0.491150 | 0.000000 |
| Random Forest (winner) | KBANK.BK | 3 | 36.28% | 27.43% | 0.00% | 0 | NaN | 0 | 0.491150 | 0.140562 |
| XGBoost (winner) | KBANK.BK | 3 | 36.28% | 27.43% | 0.00% | 0 | NaN | 0 | 0.491150 | 0.141048 |

- forecast quality ≠ executability / profitability: ราคาที่ซื้อได้จริงเกิดหลัง prediction timestamp

## 6. ข้อจำกัด

- ไม่มี historical test — ตัวเลขทั้งหมด = development / walk-forward บนข้อมูลที่เคย probe แล้ว
- prospective evaluation ยังไม่เปิด (ต้องมี human freeze + พิสูจน์ real-time availability ก่อน)
- close_bar16 ≠ official SET close (ตรงกับ Investing close เป๊ะแค่ 45.6% / 40.5% ของวัน)
- time semantics: bar semantics = working assumption · price alignment ยังไม่ตรวจ · real-time availability ยังไม่ได้พิสูจน์
- ไม่มี executable backtest
