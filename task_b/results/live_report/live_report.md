# Live test report — official predictions

> ผลของ **prospective prediction** เท่านั้น · ห้ามรวมกับผล validation / walk-forward เดิม
> แยกตาม prediction_type × หุ้น × โมเดล · ผลจริงจาก raw_data/ (investing.com + yahoo ที่ตรวจ OHLC เป๊ะแล้ว) ไฟล์เดียวกับที่เทรน
> **n < 20 วัน = ผลยังแกว่งมาก อย่าสรุปว่าชนะหรือแพ้**

อ่านอย่างไร: relMAE < 1 = ดีกว่า Naive (ทายว่าราคาไม่เปลี่ยน) · R2_OOS > 0 = MSE ดีกว่า Naive · win_rate = สัดส่วนวันที่พลาดน้อยกว่า Naive (ไม่นับวันเสมอ) · dir_hit_rate นับเฉพาะวันที่ราคาขยับจริง · long_signal = วันที่ทำนาย return เกินต้นทุนไป-กลับ (diagnostic ไม่ใช่กำไร)

## next_day

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 7 | 2026-10-01 | 2026-10-19 | 0.003589 | 0.003774 | 0.950977 | 0.077620 | 0.005104 | 1.224210 | 3 | 4 | 0 | 0.428571 |
| ADVANC.BK | Random Forest | 7 | 2026-10-01 | 2026-10-19 | 0.003705 | 0.003774 | 0.981697 | 0.081689 | 0.005093 | 1.261642 | 3 | 4 | 0 | 0.428571 |
| ADVANC.BK | XGBoost | 7 | 2026-10-01 | 2026-10-19 | 0.003683 | 0.003774 | 0.976047 | 0.107521 | 0.005020 | 1.255138 | 3 | 4 | 0 | 0.428571 |
| KBANK.BK | ANN (MLP) | 7 | 2026-10-01 | 2026-10-19 | 0.007640 | 0.007350 | 1.039448 | -0.013412 | 0.009536 | 1.782517 | 3 | 4 | 0 | 0.428571 |
| KBANK.BK | Random Forest | 7 | 2026-10-01 | 2026-10-19 | 0.007990 | 0.007350 | 1.087069 | -0.090470 | 0.009892 | 1.864475 | 1 | 6 | 0 | 0.142857 |
| KBANK.BK | XGBoost | 7 | 2026-10-01 | 2026-10-19 | 0.007662 | 0.007350 | 1.042475 | -0.020260 | 0.009568 | 1.788014 | 2 | 5 | 0 | 0.285714 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.000031 | 0.114392 | 0.367005 | 3 | 0.750000 | 0 | NaN | 0 | False |
| ADVANC.BK | Random Forest | -0.000298 | 0.126243 | 0.372466 | 3 | 0.750000 | 0 | NaN | 0 | False |
| ADVANC.BK | XGBoost | -0.000180 | 0.154472 | 0.407797 | 3 | 0.750000 | 0 | NaN | 0 | False |
| KBANK.BK | ANN (MLP) | -0.003872 | 0.058474 | 0.034127 | 2 | 0.600000 | 0 | NaN | 0 | False |
| KBANK.BK | Random Forest | -0.003491 | 0.135439 | -0.402383 | 2 | 0.200000 | 0 | NaN | 0 | False |
| KBANK.BK | XGBoost | -0.003358 | 0.051574 | -0.521191 | 2 | 0.400000 | 0 | NaN | 0 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 7 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| ADVANC.BK | Random Forest | 7 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| ADVANC.BK | XGBoost | 7 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | ANN (MLP) | 7 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | Random Forest | 7 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | XGBoost | 7 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |

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
