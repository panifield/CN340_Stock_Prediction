# Live test report — official predictions

> ผลของ **prospective prediction** เท่านั้น · ห้ามรวมกับผล validation / walk-forward เดิม
> แยกตาม prediction_type × หุ้น × โมเดล · ผลจริงจาก raw_data/ (investing.com + yahoo ที่ตรวจ OHLC เป๊ะแล้ว) ไฟล์เดียวกับที่เทรน
> **n < 20 วัน = ผลยังแกว่งมาก อย่าสรุปว่าชนะหรือแพ้**

อ่านอย่างไร: relMAE < 1 = ดีกว่า Naive (ทายว่าราคาไม่เปลี่ยน) · R2_OOS > 0 = MSE ดีกว่า Naive · win_rate = สัดส่วนวันที่พลาดน้อยกว่า Naive (ไม่นับวันเสมอ) · dir_hit_rate นับเฉพาะวันที่ราคาขยับจริง · long_signal = วันที่ทำนาย return เกินต้นทุนไป-กลับ (diagnostic ไม่ใช่กำไร)

## next_day

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 11 | 2026-10-01 | 2026-10-26 | 0.004546 | 0.004580 | 0.992662 | 0.024618 | 0.005646 | 1.534684 | 4 | 7 | 0 | 0.363636 |
| ADVANC.BK | Random Forest | 11 | 2026-10-01 | 2026-10-26 | 0.004609 | 0.004580 | 1.006441 | 0.017969 | 0.005666 | 1.554752 | 5 | 6 | 0 | 0.454545 |
| ADVANC.BK | XGBoost | 11 | 2026-10-01 | 2026-10-26 | 0.004648 | 0.004580 | 1.014964 | 0.018412 | 0.005664 | 1.568235 | 4 | 7 | 0 | 0.363636 |
| KBANK.BK | ANN (MLP) | 11 | 2026-10-01 | 2026-10-26 | 0.009894 | 0.009661 | 1.024116 | -0.010456 | 0.011467 | 2.327749 | 5 | 6 | 0 | 0.454545 |
| KBANK.BK | Random Forest | 11 | 2026-10-01 | 2026-10-26 | 0.010087 | 0.009661 | 1.044096 | -0.039524 | 0.011630 | 2.372747 | 2 | 9 | 0 | 0.181818 |
| KBANK.BK | XGBoost | 11 | 2026-10-01 | 2026-10-26 | 0.009832 | 0.009661 | 1.017680 | 0.001409 | 0.011399 | 2.313113 | 4 | 7 | 0 | 0.363636 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.000983 | 0.116081 | 0.039886 | 3 | 0.500000 | 0 | NaN | 0 | False |
| ADVANC.BK | Random Forest | -0.001198 | 0.105880 | 0.071780 | 3 | 0.625000 | 0 | NaN | 0 | False |
| ADVANC.BK | XGBoost | -0.001135 | 0.132868 | 0.065159 | 3 | 0.500000 | 0 | NaN | 0 | False |
| KBANK.BK | ANN (MLP) | -0.005028 | 0.053199 | -0.107952 | 2 | 0.555556 | 0 | NaN | 0 | False |
| KBANK.BK | Random Forest | -0.004838 | 0.095267 | -0.311167 | 2 | 0.222222 | 0 | NaN | 0 | False |
| KBANK.BK | XGBoost | -0.004706 | 0.037425 | -0.376274 | 2 | 0.444444 | 0 | NaN | 0 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 11 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| ADVANC.BK | Random Forest | 11 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| ADVANC.BK | XGBoost | 11 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | ANN (MLP) | 11 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | Random Forest | 11 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | XGBoost | 11 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |

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
| ADVANC.BK | ANN (MLP) | 3 | 2026-10-19 | 2026-10-22 | 0.001968 | 0.001990 | 0.988989 | 0.041284 | 0.002386 | 0.659255 | 1 | 2 | 0 | 0.333333 |
| ADVANC.BK | Random Forest | 3 | 2026-10-19 | 2026-10-22 | 0.001840 | 0.001990 | 0.924522 | 0.189367 | 0.002194 | 0.616196 | 2 | 1 | 0 | 0.666667 |
| ADVANC.BK | XGBoost | 3 | 2026-10-19 | 2026-10-22 | 0.001884 | 0.001990 | 0.946894 | 0.222686 | 0.002149 | 0.630822 | 2 | 1 | 0 | 0.666667 |
| KBANK.BK | ANN (MLP) | 3 | 2026-10-19 | 2026-10-22 | 0.003010 | 0.002819 | 1.067682 | -0.138773 | 0.003684 | 0.711667 | 0 | 3 | 0 | 0.000000 |
| KBANK.BK | Random Forest | 3 | 2026-10-19 | 2026-10-22 | 0.003344 | 0.002819 | 1.186274 | -0.190414 | 0.003767 | 0.792129 | 1 | 2 | 0 | 0.333333 |
| KBANK.BK | XGBoost | 3 | 2026-10-19 | 2026-10-22 | 0.003220 | 0.002819 | 1.142109 | -0.132322 | 0.003674 | 0.762425 | 1 | 2 | 0 | 0.333333 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 0.000146 | 0.031536 | 0.727041 | 1 | 0.500000 | 0 | NaN | 2 | False |
| ADVANC.BK | Random Forest | 0.000014 | 0.111913 | 0.902157 | 1 | 1.000000 | 0 | NaN | 2 | False |
| ADVANC.BK | XGBoost | -0.000019 | 0.176131 | 0.720407 | 1 | 1.000000 | 0 | NaN | 2 | False |
| KBANK.BK | ANN (MLP) | 0.003003 | 0.090594 | -0.768349 | 1 | 0.000000 | 0 | NaN | 1 | False |
| KBANK.BK | Random Forest | 0.003344 | 0.258521 | 0.599621 | 1 | 0.500000 | 0 | NaN | 1 | False |
| KBANK.BK | XGBoost | 0.003220 | 0.201720 | 0.625888 | 1 | 0.500000 | 0 | NaN | 1 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 4 | 1 | 2026-10-21 | 0 | 0 |
| ADVANC.BK | Random Forest | 4 | 1 | 2026-10-21 | 0 | 0 |
| ADVANC.BK | XGBoost | 4 | 1 | 2026-10-21 | 0 | 0 |
| KBANK.BK | ANN (MLP) | 4 | 1 | 2026-10-21 | 0 | 0 |
| KBANK.BK | Random Forest | 4 | 1 | 2026-10-21 | 0 | 0 |
| KBANK.BK | XGBoost | 4 | 1 | 2026-10-21 | 0 | 0 |
