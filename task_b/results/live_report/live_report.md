# Live test report — official predictions

> ผลของ **prospective prediction** เท่านั้น · ห้ามรวมกับผล validation / walk-forward เดิม
> แยกตาม prediction_type × หุ้น × โมเดล · ผลจริงจาก raw_data/ (investing.com + yahoo ที่ตรวจ OHLC เป๊ะแล้ว) ไฟล์เดียวกับที่เทรน
> **n < 20 วัน = ผลยังแกว่งมาก อย่าสรุปว่าชนะหรือแพ้**

อ่านอย่างไร: relMAE < 1 = ดีกว่า Naive (ทายว่าราคาไม่เปลี่ยน) · R2_OOS > 0 = MSE ดีกว่า Naive · win_rate = สัดส่วนวันที่พลาดน้อยกว่า Naive (ไม่นับวันเสมอ) · dir_hit_rate นับเฉพาะวันที่ราคาขยับจริง · long_signal = วันที่ทำนาย return เกินต้นทุนไป-กลับ (diagnostic ไม่ใช่กำไร)

## next_day

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 9 | 2026-10-01 | 2026-10-21 | 0.004487 | 0.004604 | 0.974399 | 0.050705 | 0.005766 | 1.516611 | 4 | 5 | 0 | 0.444444 |
| ADVANC.BK | Random Forest | 9 | 2026-10-01 | 2026-10-21 | 0.004620 | 0.004604 | 1.003401 | 0.029411 | 0.005830 | 1.560055 | 4 | 5 | 0 | 0.444444 |
| ADVANC.BK | XGBoost | 9 | 2026-10-01 | 2026-10-21 | 0.004621 | 0.004604 | 1.003689 | 0.039263 | 0.005801 | 1.560992 | 4 | 5 | 0 | 0.444444 |
| KBANK.BK | ANN (MLP) | 9 | 2026-10-01 | 2026-10-21 | 0.009337 | 0.009023 | 1.034886 | -0.026464 | 0.011101 | 2.185369 | 4 | 5 | 0 | 0.444444 |
| KBANK.BK | Random Forest | 9 | 2026-10-01 | 2026-10-21 | 0.009594 | 0.009023 | 1.063388 | -0.070111 | 0.011335 | 2.245437 | 1 | 8 | 0 | 0.111111 |
| KBANK.BK | XGBoost | 9 | 2026-10-01 | 2026-10-21 | 0.009281 | 0.009023 | 1.028630 | -0.012611 | 0.011026 | 2.172240 | 3 | 6 | 0 | 0.333333 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.000131 | 0.101338 | 0.238444 | 3 | 0.666667 | 0 | NaN | 0 | False |
| ADVANC.BK | Random Forest | -0.000451 | 0.106836 | 0.155706 | 3 | 0.666667 | 0 | NaN | 0 | False |
| ADVANC.BK | XGBoost | -0.000327 | 0.129979 | 0.176765 | 3 | 0.666667 | 0 | NaN | 0 | False |
| KBANK.BK | ANN (MLP) | -0.003390 | 0.054101 | -0.235596 | 2 | 0.571429 | 0 | NaN | 0 | False |
| KBANK.BK | Random Forest | -0.003179 | 0.101073 | -0.391792 | 2 | 0.142857 | 0 | NaN | 0 | False |
| KBANK.BK | XGBoost | -0.003015 | 0.038228 | -0.441316 | 2 | 0.428571 | 0 | NaN | 0 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 9 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| ADVANC.BK | Random Forest | 9 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| ADVANC.BK | XGBoost | 9 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | ANN (MLP) | 9 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | Random Forest | 9 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | XGBoost | 9 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |

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
