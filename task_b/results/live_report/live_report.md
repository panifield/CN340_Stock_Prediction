# Live test report — official predictions

> ผลของ **prospective prediction** เท่านั้น · ห้ามรวมกับผล validation / walk-forward เดิม
> แยกตาม prediction_type × หุ้น × โมเดล · ผลจริงจาก raw_data/ (investing.com + yahoo ที่ตรวจ OHLC เป๊ะแล้ว) ไฟล์เดียวกับที่เทรน
> **n < 20 วัน = ผลยังแกว่งมาก อย่าสรุปว่าชนะหรือแพ้**

อ่านอย่างไร: relMAE < 1 = ดีกว่า Naive (ทายว่าราคาไม่เปลี่ยน) · R2_OOS > 0 = MSE ดีกว่า Naive · win_rate = สัดส่วนวันที่พลาดน้อยกว่า Naive (ไม่นับวันเสมอ) · dir_hit_rate นับเฉพาะวันที่ราคาขยับจริง · long_signal = วันที่ทำนาย return เกินต้นทุนไป-กลับ (diagnostic ไม่ใช่กำไร)

## next_day

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 4 | 2026-10-01 | 2026-10-06 | 0.005030 | 0.005098 | 0.986676 | 0.013412 | 0.006309 | 1.727199 | 2 | 2 | 0 | 0.500000 |
| ADVANC.BK | Random Forest | 4 | 2026-10-01 | 2026-10-06 | 0.004842 | 0.005098 | 0.949826 | 0.100678 | 0.006024 | 1.662812 | 2 | 2 | 0 | 0.500000 |
| ADVANC.BK | XGBoost | 4 | 2026-10-01 | 2026-10-06 | 0.004900 | 0.005098 | 0.961186 | 0.078763 | 0.006097 | 1.682835 | 2 | 2 | 0 | 0.500000 |
| KBANK.BK | ANN (MLP) | 4 | 2026-10-01 | 2026-10-06 | 0.006677 | 0.006406 | 1.042304 | -0.011690 | 0.008044 | 1.563762 | 2 | 2 | 0 | 0.500000 |
| KBANK.BK | Random Forest | 4 | 2026-10-01 | 2026-10-06 | 0.006770 | 0.006406 | 1.056791 | -0.007863 | 0.008029 | 1.586914 | 1 | 3 | 0 | 0.250000 |
| KBANK.BK | XGBoost | 4 | 2026-10-01 | 2026-10-06 | 0.006897 | 0.006406 | 1.076677 | -0.060709 | 0.008237 | 1.615900 | 1 | 3 | 0 | 0.250000 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 0.001195 | 0.092254 | 0.245111 | 1 | 0.666667 | 0 | NaN | 0 | False |
| ADVANC.BK | Random Forest | 0.001120 | 0.124853 | 0.545867 | 1 | 0.666667 | 0 | NaN | 0 | False |
| ADVANC.BK | XGBoost | 0.001230 | 0.137531 | 0.450603 | 1 | 0.666667 | 0 | NaN | 0 | False |
| KBANK.BK | ANN (MLP) | -0.000083 | 0.051798 | -0.086049 | 1 | 0.666667 | 0 | NaN | 0 | False |
| KBANK.BK | Random Forest | 0.001103 | 0.083769 | 0.108382 | 1 | 0.333333 | 0 | NaN | 0 | False |
| KBANK.BK | XGBoost | 0.000635 | 0.031470 | -0.848981 | 1 | 0.333333 | 0 | NaN | 0 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 4 | 0 |  | 0 | 0 |
| ADVANC.BK | Random Forest | 4 | 0 |  | 0 | 0 |
| ADVANC.BK | XGBoost | 4 | 0 |  | 0 | 0 |
| KBANK.BK | ANN (MLP) | 4 | 0 |  | 0 | 0 |
| KBANK.BK | Random Forest | 4 | 0 |  | 0 | 0 |
| KBANK.BK | XGBoost | 4 | 0 |  | 0 | 0 |

## same_day_1600

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 4 | 2026-09-30 | 2026-10-05 | 0.001505 | 0.001454 | 1.035622 | 0.173015 | 0.001869 | 0.518041 | 1 | 3 | 0 | 0.250000 |
| ADVANC.BK | Random Forest | 4 | 2026-09-30 | 2026-10-05 | 0.001408 | 0.001454 | 0.968580 | 0.210439 | 0.001827 | 0.484495 | 2 | 2 | 0 | 0.500000 |
| ADVANC.BK | XGBoost | 4 | 2026-09-30 | 2026-10-05 | 0.001387 | 0.001454 | 0.953956 | 0.283254 | 0.001740 | 0.477120 | 2 | 2 | 0 | 0.500000 |
| KBANK.BK | ANN (MLP) | 4 | 2026-09-30 | 2026-10-05 | 0.003061 | 0.003192 | 0.958938 | 0.160630 | 0.003377 | 0.720229 | 2 | 2 | 0 | 0.500000 |
| KBANK.BK | Random Forest | 4 | 2026-09-30 | 2026-10-05 | 0.002916 | 0.003192 | 0.913585 | 0.268467 | 0.003152 | 0.686561 | 3 | 1 | 0 | 0.750000 |
| KBANK.BK | XGBoost | 4 | 2026-09-30 | 2026-10-05 | 0.003117 | 0.003192 | 0.976545 | 0.217529 | 0.003260 | 0.734742 | 2 | 2 | 0 | 0.500000 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.001065 | 0.217177 | -0.160671 | 2 | 0.500000 | 0 | NaN | 2 | False |
| ADVANC.BK | Random Forest | -0.001183 | 0.230107 | 0.295508 | 2 | 1.000000 | 0 | NaN | 2 | False |
| ADVANC.BK | XGBoost | -0.001069 | 0.225829 | 0.350880 | 2 | 1.000000 | 0 | NaN | 2 | False |
| KBANK.BK | ANN (MLP) | 0.001197 | 0.213460 | 0.579291 | 1 | 0.666667 | 0 | NaN | 1 | False |
| KBANK.BK | Random Forest | 0.001078 | 0.293129 | 0.653897 | 1 | 1.000000 | 0 | NaN | 1 | False |
| KBANK.BK | XGBoost | 0.000878 | 0.400680 | 0.463255 | 1 | 0.666667 | 0 | NaN | 1 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 5 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| ADVANC.BK | Random Forest | 5 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| ADVANC.BK | XGBoost | 5 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| KBANK.BK | ANN (MLP) | 5 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| KBANK.BK | Random Forest | 5 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
| KBANK.BK | XGBoost | 5 | 4 | 2026-10-06 2026-10-07 2026-10-08 2026-10-09 | 0 | 0 |
