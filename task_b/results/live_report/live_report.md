# Live test report — official predictions

> ผลของ **prospective prediction** เท่านั้น · ห้ามรวมกับผล validation / walk-forward เดิม
> แยกตาม prediction_type × หุ้น × โมเดล · ผลจริงจาก raw_data/ (investing.com + yahoo ที่ตรวจ OHLC เป๊ะแล้ว) ไฟล์เดียวกับที่เทรน
> **n < 20 วัน = ผลยังแกว่งมาก อย่าสรุปว่าชนะหรือแพ้**

อ่านอย่างไร: relMAE < 1 = ดีกว่า Naive (ทายว่าราคาไม่เปลี่ยน) · R2_OOS > 0 = MSE ดีกว่า Naive · win_rate = สัดส่วนวันที่พลาดน้อยกว่า Naive (ไม่นับวันเสมอ) · dir_hit_rate นับเฉพาะวันที่ราคาขยับจริง · long_signal = วันที่ทำนาย return เกินต้นทุนไป-กลับ (diagnostic ไม่ใช่กำไร)

## next_day

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 8 | 2026-10-01 | 2026-10-20 | 0.004034 | 0.004050 | 0.995819 | 0.000490 | 0.005402 | 1.369592 | 3 | 5 | 0 | 0.375000 |
| ADVANC.BK | Random Forest | 8 | 2026-10-01 | 2026-10-20 | 0.004096 | 0.004050 | 1.011290 | 0.022507 | 0.005342 | 1.389396 | 3 | 5 | 0 | 0.375000 |
| ADVANC.BK | XGBoost | 8 | 2026-10-01 | 2026-10-20 | 0.004106 | 0.004050 | 1.013676 | 0.030887 | 0.005319 | 1.393164 | 3 | 5 | 0 | 0.375000 |
| KBANK.BK | ANN (MLP) | 8 | 2026-10-01 | 2026-10-20 | 0.008382 | 0.008014 | 1.045978 | -0.041234 | 0.010130 | 1.961898 | 3 | 5 | 0 | 0.375000 |
| KBANK.BK | Random Forest | 8 | 2026-10-01 | 2026-10-20 | 0.008632 | 0.008014 | 1.077170 | -0.087373 | 0.010352 | 2.020266 | 1 | 7 | 0 | 0.125000 |
| KBANK.BK | XGBoost | 8 | 2026-10-01 | 2026-10-20 | 0.008346 | 0.008014 | 1.041478 | -0.031641 | 0.010083 | 1.953548 | 2 | 6 | 0 | 0.250000 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 0.000866 | 0.114387 | 0.153626 | 3 | 0.600000 | 0 | NaN | 0 | False |
| ADVANC.BK | Random Forest | 0.000594 | 0.123520 | 0.185214 | 3 | 0.600000 | 0 | NaN | 0 | False |
| ADVANC.BK | XGBoost | 0.000725 | 0.150195 | 0.224447 | 3 | 0.600000 | 0 | NaN | 0 | False |
| KBANK.BK | ANN (MLP) | -0.001691 | 0.060922 | -0.309323 | 2 | 0.500000 | 0 | NaN | 0 | False |
| KBANK.BK | Random Forest | -0.001414 | 0.113160 | -0.375776 | 2 | 0.166667 | 0 | NaN | 0 | False |
| KBANK.BK | XGBoost | -0.001297 | 0.043175 | -0.485988 | 2 | 0.333333 | 0 | NaN | 0 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 8 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| ADVANC.BK | Random Forest | 8 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| ADVANC.BK | XGBoost | 8 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | ANN (MLP) | 8 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | Random Forest | 8 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | XGBoost | 8 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |

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
| ADVANC.BK | ANN (MLP) | 1 | 2026-10-19 | 2026-10-19 | 0.003100 | 0.002985 | 1.038409 | -0.078293 | 0.003100 | 1.038409 | 0 | 1 | 0 | 0.000000 |
| ADVANC.BK | Random Forest | 1 | 2026-10-19 | 2026-10-19 | 0.002781 | 0.002985 | 0.931691 | 0.131951 | 0.002781 | 0.931691 | 1 | 0 | 0 | 1.000000 |
| ADVANC.BK | XGBoost | 1 | 2026-10-19 | 2026-10-19 | 0.002797 | 0.002985 | 0.937134 | 0.121779 | 0.002797 | 0.937134 | 1 | 0 | 0 | 1.000000 |
| KBANK.BK | ANN (MLP) | 1 | 2026-10-19 | 2026-10-19 | 0.004339 | 0.004202 | 1.032683 | -0.066433 | 0.004339 | 1.032683 | 0 | 1 | 0 | 0.000000 |
| KBANK.BK | Random Forest | 1 | 2026-10-19 | 2026-10-19 | 0.004004 | 0.004202 | 0.952990 | 0.091810 | 0.004004 | 0.952990 | 1 | 0 | 0 | 1.000000 |
| KBANK.BK | XGBoost | 1 | 2026-10-19 | 2026-10-19 | 0.004041 | 0.004202 | 0.961806 | 0.074929 | 0.004041 | 0.961806 | 1 | 0 | 0 | 1.000000 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 0.003100 | NaN | NaN | 0 | 0.000000 | 0 | NaN | 0 | False |
| ADVANC.BK | Random Forest | 0.002781 | NaN | NaN | 0 | 1.000000 | 0 | NaN | 0 | False |
| ADVANC.BK | XGBoost | 0.002797 | NaN | NaN | 0 | 1.000000 | 0 | NaN | 0 | False |
| KBANK.BK | ANN (MLP) | 0.004339 | NaN | NaN | 0 | 0.000000 | 0 | NaN | 0 | False |
| KBANK.BK | Random Forest | 0.004004 | NaN | NaN | 0 | 1.000000 | 0 | NaN | 0 | False |
| KBANK.BK | XGBoost | 0.004041 | NaN | NaN | 0 | 1.000000 | 0 | NaN | 0 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 2 | 0 |  | 0 | 0 |
| ADVANC.BK | Random Forest | 2 | 0 |  | 0 | 0 |
| ADVANC.BK | XGBoost | 2 | 0 |  | 0 | 0 |
| KBANK.BK | ANN (MLP) | 2 | 0 |  | 0 | 0 |
| KBANK.BK | Random Forest | 2 | 0 |  | 0 | 0 |
| KBANK.BK | XGBoost | 2 | 0 |  | 0 | 0 |
