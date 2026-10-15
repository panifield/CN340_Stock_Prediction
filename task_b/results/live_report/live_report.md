# Live test report — official predictions

> ผลของ **prospective prediction** เท่านั้น · ห้ามรวมกับผล validation / walk-forward เดิม
> แยกตาม prediction_type × หุ้น × โมเดล · ผลจริงจาก raw_data/ (investing.com + yahoo ที่ตรวจ OHLC เป๊ะแล้ว) ไฟล์เดียวกับที่เทรน
> **n < 20 วัน = ผลยังแกว่งมาก อย่าสรุปว่าชนะหรือแพ้**

อ่านอย่างไร: relMAE < 1 = ดีกว่า Naive (ทายว่าราคาไม่เปลี่ยน) · R2_OOS > 0 = MSE ดีกว่า Naive · win_rate = สัดส่วนวันที่พลาดน้อยกว่า Naive (ไม่นับวันเสมอ) · dir_hit_rate นับเฉพาะวันที่ราคาขยับจริง · long_signal = วันที่ทำนาย return เกินต้นทุนไป-กลับ (diagnostic ไม่ใช่กำไร)

## next_day

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 5 | 2026-10-01 | 2026-10-14 | 0.004985 | 0.005283 | 0.943497 | 0.077814 | 0.006038 | 1.700695 | 3 | 2 | 0 | 0.600000 |
| ADVANC.BK | Random Forest | 5 | 2026-10-01 | 2026-10-14 | 0.005078 | 0.005283 | 0.961119 | 0.082437 | 0.006023 | 1.729988 | 3 | 2 | 0 | 0.600000 |
| ADVANC.BK | XGBoost | 5 | 2026-10-01 | 2026-10-14 | 0.004964 | 0.005283 | 0.939682 | 0.109921 | 0.005932 | 1.693025 | 3 | 2 | 0 | 0.600000 |
| KBANK.BK | ANN (MLP) | 5 | 2026-10-01 | 2026-10-14 | 0.005402 | 0.005125 | 1.054042 | -0.012043 | 0.007196 | 1.264906 | 2 | 3 | 0 | 0.400000 |
| KBANK.BK | Random Forest | 5 | 2026-10-01 | 2026-10-14 | 0.005595 | 0.005125 | 1.091761 | -0.011001 | 0.007193 | 1.310929 | 1 | 4 | 0 | 0.200000 |
| KBANK.BK | XGBoost | 5 | 2026-10-01 | 2026-10-14 | 0.005540 | 0.005125 | 1.081064 | -0.060758 | 0.007368 | 1.297913 | 1 | 4 | 0 | 0.200000 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.000005 | 0.095688 | 0.404755 | 1 | 0.750000 | 0 | NaN | 0 | False |
| ADVANC.BK | Random Forest | -0.000308 | 0.115604 | 0.383910 | 1 | 0.750000 | 0 | NaN | 0 | False |
| ADVANC.BK | XGBoost | -0.000060 | 0.125438 | 0.464646 | 1 | 0.750000 | 0 | NaN | 0 | False |
| KBANK.BK | ANN (MLP) | -0.000127 | 0.053641 | -0.082629 | 2 | 0.666667 | 0 | NaN | 0 | False |
| KBANK.BK | Random Forest | 0.000703 | 0.141116 | 0.065771 | 2 | 0.333333 | 0 | NaN | 0 | False |
| KBANK.BK | XGBoost | 0.000485 | 0.053741 | -0.495705 | 2 | 0.333333 | 0 | NaN | 0 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 5 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| ADVANC.BK | Random Forest | 5 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| ADVANC.BK | XGBoost | 5 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | ANN (MLP) | 5 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | Random Forest | 5 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |
| KBANK.BK | XGBoost | 5 | 4 | 2026-10-07 2026-10-08 2026-10-09 2026-10-12 | 0 | 0 |

## same_day_1600

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 5 | 2026-09-30 | 2026-10-12 | 0.001709 | 0.001767 | 0.967066 | 0.218438 | 0.002017 | 0.581460 | 2 | 3 | 0 | 0.400000 |
| ADVANC.BK | Random Forest | 5 | 2026-09-30 | 2026-10-12 | 0.001599 | 0.001767 | 0.904654 | 0.273071 | 0.001945 | 0.543922 | 3 | 2 | 0 | 0.600000 |
| ADVANC.BK | XGBoost | 5 | 2026-09-30 | 2026-10-12 | 0.001570 | 0.001767 | 0.888317 | 0.330980 | 0.001866 | 0.534096 | 3 | 2 | 0 | 0.600000 |
| KBANK.BK | ANN (MLP) | 5 | 2026-09-30 | 2026-10-12 | 0.002619 | 0.002553 | 1.025732 | 0.147247 | 0.003044 | 0.615410 | 2 | 3 | 0 | 0.400000 |
| KBANK.BK | Random Forest | 5 | 2026-09-30 | 2026-10-12 | 0.002702 | 0.002553 | 1.058145 | 0.205779 | 0.002938 | 0.634146 | 3 | 2 | 0 | 0.600000 |
| KBANK.BK | XGBoost | 5 | 2026-09-30 | 2026-10-12 | 0.002850 | 0.002553 | 1.116286 | 0.158952 | 0.003023 | 0.669861 | 2 | 3 | 0 | 0.400000 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.001357 | 0.197961 | -0.076268 | 2 | 0.666667 | 0 | NaN | 2 | False |
| ADVANC.BK | Random Forest | -0.001419 | 0.233616 | 0.436597 | 2 | 1.000000 | 0 | NaN | 2 | False |
| ADVANC.BK | XGBoost | -0.001316 | 0.223543 | 0.467770 | 2 | 1.000000 | 0 | NaN | 2 | False |
| KBANK.BK | ANN (MLP) | 0.001128 | 0.229211 | 0.580608 | 2 | 0.666667 | 0 | NaN | 2 | False |
| KBANK.BK | Random Forest | 0.001231 | 0.369056 | 0.591303 | 2 | 1.000000 | 0 | NaN | 2 | False |
| KBANK.BK | XGBoost | 0.001059 | 0.466641 | 0.459777 | 2 | 0.666667 | 0 | NaN | 2 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 6 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| ADVANC.BK | Random Forest | 6 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| ADVANC.BK | XGBoost | 6 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| KBANK.BK | ANN (MLP) | 6 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| KBANK.BK | Random Forest | 6 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| KBANK.BK | XGBoost | 6 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
