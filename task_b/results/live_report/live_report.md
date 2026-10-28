# Live test report — official predictions

> ผลของ **prospective prediction** เท่านั้น · ห้ามรวมกับผล validation / walk-forward เดิม
> แยกตาม prediction_type × หุ้น × โมเดล · ผลจริงจาก raw_data/ (investing.com + yahoo ที่ตรวจ OHLC เป๊ะแล้ว) ไฟล์เดียวกับที่เทรน
> **n < 20 วัน = ผลยังแกว่งมาก อย่าสรุปว่าชนะหรือแพ้**

อ่านอย่างไร: relMAE < 1 = ดีกว่า Naive (ทายว่าราคาไม่เปลี่ยน) · R2_OOS > 0 = MSE ดีกว่า Naive · win_rate = สัดส่วนวันที่พลาดน้อยกว่า Naive (ไม่นับวันเสมอ) · dir_hit_rate นับเฉพาะวันที่ราคาขยับจริง · long_signal = วันที่ทำนาย return เกินต้นทุนไป-กลับ (diagnostic ไม่ใช่กำไร)

## next_day

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 12 | 2026-10-01 | 2026-10-27 | 0.004928 | 0.004938 | 0.998050 | 0.009755 | 0.006014 | 1.663951 | 4 | 8 | 0 | 0.333333 |
| ADVANC.BK | Random Forest | 12 | 2026-10-01 | 2026-10-27 | 0.004994 | 0.004938 | 1.011426 | 0.000179 | 0.006043 | 1.685119 | 5 | 7 | 0 | 0.416667 |
| ADVANC.BK | XGBoost | 12 | 2026-10-01 | 2026-10-27 | 0.005044 | 0.004938 | 1.021549 | -0.006702 | 0.006064 | 1.702280 | 4 | 8 | 0 | 0.333333 |
| KBANK.BK | ANN (MLP) | 12 | 2026-10-01 | 2026-10-27 | 0.009304 | 0.009198 | 1.011511 | -0.004173 | 0.011008 | 2.190824 | 6 | 6 | 0 | 0.500000 |
| KBANK.BK | Random Forest | 12 | 2026-10-01 | 2026-10-27 | 0.009555 | 0.009198 | 1.038903 | -0.036950 | 0.011186 | 2.250371 | 3 | 9 | 0 | 0.250000 |
| KBANK.BK | XGBoost | 12 | 2026-10-01 | 2026-10-27 | 0.009321 | 0.009198 | 1.013443 | 0.003521 | 0.010966 | 2.195652 | 5 | 7 | 0 | 0.416667 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.000140 | 0.102624 | 0.064949 | 3 | 0.444444 | 0 | NaN | 0 | False |
| ADVANC.BK | Random Forest | -0.000329 | 0.093751 | 0.023407 | 3 | 0.555556 | 0 | NaN | 0 | False |
| ADVANC.BK | XGBoost | -0.000257 | 0.117928 | 0.005959 | 3 | 0.444444 | 0 | NaN | 0 | False |
| KBANK.BK | ANN (MLP) | -0.004843 | 0.064254 | -0.104250 | 2 | 0.600000 | 0 | NaN | 0 | False |
| KBANK.BK | Random Forest | -0.004744 | 0.095382 | -0.312057 | 2 | 0.300000 | 0 | NaN | 0 | False |
| KBANK.BK | XGBoost | -0.004622 | 0.037448 | -0.376950 | 2 | 0.500000 | 0 | NaN | 0 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 12 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| ADVANC.BK | Random Forest | 12 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| ADVANC.BK | XGBoost | 12 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | ANN (MLP) | 12 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | Random Forest | 12 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | XGBoost | 12 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |

## same_day_1600

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 7 | 2026-09-30 | 2026-10-15 | 0.002778 | 0.002991 | 0.928716 | 0.200422 | 0.003667 | 0.930041 | 4 | 3 | 0 | 0.571429 |
| ADVANC.BK | Random Forest | 7 | 2026-09-30 | 2026-10-15 | 0.002749 | 0.002991 | 0.919071 | 0.178302 | 0.003718 | 0.919750 | 5 | 2 | 0 | 0.714286 |
| ADVANC.BK | XGBoost | 7 | 2026-09-30 | 2026-10-15 | 0.002767 | 0.002991 | 0.925084 | 0.156531 | 0.003767 | 0.925483 | 5 | 2 | 0 | 0.714286 |
| KBANK.BK | ANN (MLP) | 7 | 2026-09-30 | 2026-10-15 | 0.002455 | 0.002442 | 1.005130 | 0.150164 | 0.002979 | 0.574556 | 3 | 4 | 0 | 0.428571 |
| KBANK.BK | Random Forest | 7 | 2026-09-30 | 2026-10-15 | 0.002619 | 0.002442 | 1.072198 | 0.171372 | 0.002941 | 0.612460 | 4 | 3 | 0 | 0.571429 |
| KBANK.BK | XGBoost | 7 | 2026-09-30 | 2026-10-15 | 0.002718 | 0.002442 | 1.112926 | 0.116712 | 0.003037 | 0.636308 | 3 | 4 | 0 | 0.428571 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.002526 | 0.113967 | 0.506681 | 2 | 0.800000 | 0 | NaN | 3 | False |
| ADVANC.BK | Random Forest | -0.002620 | 0.121423 | 0.541360 | 2 | 1.000000 | 0 | NaN | 3 | False |
| ADVANC.BK | XGBoost | -0.002586 | 0.111471 | 0.267343 | 2 | 1.000000 | 0 | NaN | 3 | False |
| KBANK.BK | ANN (MLP) | 0.000222 | 0.196697 | 0.491922 | 3 | 0.750000 | 0 | NaN | 3 | False |
| KBANK.BK | Random Forest | 0.000191 | 0.329690 | 0.429985 | 3 | 1.000000 | 0 | NaN | 3 | False |
| KBANK.BK | XGBoost | 0.000074 | 0.396072 | 0.345994 | 3 | 0.750000 | 0 | NaN | 3 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 7 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| ADVANC.BK | Random Forest | 7 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| ADVANC.BK | XGBoost | 7 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| KBANK.BK | ANN (MLP) | 7 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| KBANK.BK | Random Forest | 7 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| KBANK.BK | XGBoost | 7 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |

## same_day_close

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 4 | 2026-10-19 | 2026-10-26 | 0.002208 | 0.002234 | 0.988390 | 0.036053 | 0.002533 | 0.741237 | 2 | 2 | 0 | 0.500000 |
| ADVANC.BK | Random Forest | 4 | 2026-10-19 | 2026-10-26 | 0.002042 | 0.002234 | 0.913896 | 0.194012 | 0.002316 | 0.685276 | 3 | 1 | 0 | 0.750000 |
| ADVANC.BK | XGBoost | 4 | 2026-10-19 | 2026-10-26 | 0.002034 | 0.002234 | 0.910286 | 0.248267 | 0.002237 | 0.682275 | 3 | 1 | 0 | 0.750000 |
| KBANK.BK | ANN (MLP) | 4 | 2026-10-19 | 2026-10-26 | 0.004257 | 0.004147 | 1.026639 | -0.028054 | 0.005117 | 1.025723 | 1 | 3 | 0 | 0.250000 |
| KBANK.BK | Random Forest | 4 | 2026-10-19 | 2026-10-26 | 0.004350 | 0.004147 | 1.049090 | 0.048950 | 0.004921 | 1.047292 | 2 | 2 | 0 | 0.500000 |
| KBANK.BK | XGBoost | 4 | 2026-10-19 | 2026-10-26 | 0.004238 | 0.004147 | 1.022091 | 0.080047 | 0.004840 | 1.020442 | 2 | 2 | 0 | 0.500000 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.000623 | 0.032895 | 0.210090 | 1 | 0.666667 | 0 | NaN | 2 | False |
| ADVANC.BK | Random Forest | -0.000651 | 0.109492 | 0.926330 | 1 | 1.000000 | 0 | NaN | 2 | False |
| ADVANC.BK | XGBoost | -0.000635 | 0.174471 | 0.793935 | 1 | 1.000000 | 0 | NaN | 2 | False |
| KBANK.BK | ANN (MLP) | 0.004252 | 0.072072 | 0.176648 | 1 | 0.333333 | 0 | NaN | 2 | False |
| KBANK.BK | Random Forest | 0.004350 | 0.248184 | 0.849186 | 1 | 0.666667 | 0 | NaN | 2 | False |
| KBANK.BK | XGBoost | 0.004238 | 0.222046 | 0.875258 | 1 | 0.666667 | 0 | NaN | 2 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 5 | 1 | 2026-10-21 | 0 | 0 |
| ADVANC.BK | Random Forest | 5 | 1 | 2026-10-21 | 0 | 0 |
| ADVANC.BK | XGBoost | 5 | 1 | 2026-10-21 | 0 | 0 |
| KBANK.BK | ANN (MLP) | 5 | 1 | 2026-10-21 | 0 | 0 |
| KBANK.BK | Random Forest | 5 | 1 | 2026-10-21 | 0 | 0 |
| KBANK.BK | XGBoost | 5 | 1 | 2026-10-21 | 0 | 0 |
