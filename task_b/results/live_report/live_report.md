# Live test report — official predictions

> ผลของ **prospective prediction** เท่านั้น · ห้ามรวมกับผล validation / walk-forward เดิม
> แยกตาม prediction_type × หุ้น × โมเดล · ผลจริงจาก raw_data/ (investing.com + yahoo ที่ตรวจ OHLC เป๊ะแล้ว) ไฟล์เดียวกับที่เทรน
> **n < 20 วัน = ผลยังแกว่งมาก อย่าสรุปว่าชนะหรือแพ้**

อ่านอย่างไร: relMAE < 1 = ดีกว่า Naive (ทายว่าราคาไม่เปลี่ยน) · R2_OOS > 0 = MSE ดีกว่า Naive · win_rate = สัดส่วนวันที่พลาดน้อยกว่า Naive (ไม่นับวันเสมอ) · dir_hit_rate นับเฉพาะวันที่ราคาขยับจริง · long_signal = วันที่ทำนาย return เกินต้นทุนไป-กลับ (diagnostic ไม่ใช่กำไร)

## next_day

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 6 | 2026-10-01 | 2026-10-15 | 0.004154 | 0.004403 | 0.943561 | 0.077814 | 0.005512 | 1.417339 | 3 | 3 | 0 | 0.500000 |
| ADVANC.BK | Random Forest | 6 | 2026-10-01 | 2026-10-15 | 0.004276 | 0.004403 | 0.971167 | 0.082080 | 0.005499 | 1.456431 | 3 | 3 | 0 | 0.500000 |
| ADVANC.BK | XGBoost | 6 | 2026-10-01 | 2026-10-15 | 0.004231 | 0.004403 | 0.960917 | 0.108329 | 0.005420 | 1.442079 | 3 | 3 | 0 | 0.500000 |
| KBANK.BK | ANN (MLP) | 6 | 2026-10-01 | 2026-10-15 | 0.007323 | 0.007157 | 1.023186 | 0.018435 | 0.009535 | 1.705773 | 3 | 3 | 0 | 0.500000 |
| KBANK.BK | Random Forest | 6 | 2026-10-01 | 2026-10-15 | 0.007695 | 0.007157 | 1.075187 | -0.061120 | 0.009914 | 1.792881 | 1 | 5 | 0 | 0.166667 |
| KBANK.BK | XGBoost | 6 | 2026-10-01 | 2026-10-15 | 0.007457 | 0.007157 | 1.041911 | -0.010819 | 0.009676 | 1.737577 | 2 | 4 | 0 | 0.333333 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.000003 | 0.104034 | 0.387977 | 2 | 0.750000 | 0 | NaN | 0 | False |
| ADVANC.BK | Random Forest | -0.000301 | 0.121797 | 0.376906 | 2 | 0.750000 | 0 | NaN | 0 | False |
| ADVANC.BK | XGBoost | -0.000144 | 0.145581 | 0.420746 | 2 | 0.750000 | 0 | NaN | 0 | False |
| KBANK.BK | ANN (MLP) | -0.002927 | 0.043068 | 0.272817 | 2 | 0.750000 | 0 | NaN | 0 | False |
| KBANK.BK | Random Forest | -0.002446 | 0.119771 | -0.343304 | 2 | 0.250000 | 0 | NaN | 0 | False |
| KBANK.BK | XGBoost | -0.002435 | 0.039410 | -0.511433 | 2 | 0.500000 | 0 | NaN | 0 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 6 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| ADVANC.BK | Random Forest | 6 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| ADVANC.BK | XGBoost | 6 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | ANN (MLP) | 6 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | Random Forest | 6 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | XGBoost | 6 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |

## same_day_1600

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 6 | 2026-09-30 | 2026-10-14 | 0.002779 | 0.002988 | 0.930302 | 0.204219 | 0.003796 | 0.931836 | 3 | 3 | 0 | 0.500000 |
| ADVANC.BK | Random Forest | 6 | 2026-09-30 | 2026-10-14 | 0.002719 | 0.002988 | 0.910097 | 0.188655 | 0.003833 | 0.910962 | 4 | 2 | 0 | 0.666667 |
| ADVANC.BK | XGBoost | 6 | 2026-09-30 | 2026-10-14 | 0.002733 | 0.002988 | 0.914637 | 0.167485 | 0.003883 | 0.915188 | 4 | 2 | 0 | 0.666667 |
| KBANK.BK | ANN (MLP) | 6 | 2026-09-30 | 2026-10-14 | 0.002844 | 0.002849 | 0.998162 | 0.150359 | 0.003217 | 0.665651 | 3 | 3 | 0 | 0.500000 |
| KBANK.BK | Random Forest | 6 | 2026-09-30 | 2026-10-14 | 0.002937 | 0.002849 | 1.030618 | 0.178287 | 0.003164 | 0.686695 | 4 | 2 | 0 | 0.666667 |
| KBANK.BK | XGBoost | 6 | 2026-09-30 | 2026-10-14 | 0.003095 | 0.002849 | 1.086154 | 0.119579 | 0.003275 | 0.724432 | 3 | 3 | 0 | 0.500000 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.002486 | 0.109312 | 0.529155 | 2 | 0.750000 | 0 | NaN | 3 | False |
| ADVANC.BK | Random Forest | -0.002569 | 0.113976 | 0.577850 | 2 | 1.000000 | 0 | NaN | 3 | False |
| ADVANC.BK | XGBoost | -0.002521 | 0.097841 | 0.306249 | 2 | 1.000000 | 0 | NaN | 3 | False |
| KBANK.BK | ANN (MLP) | 0.000278 | 0.191323 | 0.505117 | 2 | 0.750000 | 0 | NaN | 2 | False |
| KBANK.BK | Random Forest | 0.000341 | 0.308395 | 0.458685 | 2 | 1.000000 | 0 | NaN | 2 | False |
| KBANK.BK | XGBoost | 0.000163 | 0.389866 | 0.351034 | 2 | 0.750000 | 0 | NaN | 2 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 7 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| ADVANC.BK | Random Forest | 7 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| ADVANC.BK | XGBoost | 7 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| KBANK.BK | ANN (MLP) | 7 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| KBANK.BK | Random Forest | 7 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| KBANK.BK | XGBoost | 7 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
