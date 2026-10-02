# Live test report — official predictions

> ผลของ **prospective prediction** เท่านั้น · ห้ามรวมกับผล validation / walk-forward เดิม
> แยกตาม prediction_type × หุ้น × โมเดล · ผลจริงจาก raw_data/ (investing.com + yahoo ที่ตรวจ OHLC เป๊ะแล้ว) ไฟล์เดียวกับที่เทรน
> **n < 20 วัน = ผลยังแกว่งมาก อย่าสรุปว่าชนะหรือแพ้**

อ่านอย่างไร: relMAE < 1 = ดีกว่า Naive (ทายว่าราคาไม่เปลี่ยน) · R2_OOS > 0 = MSE ดีกว่า Naive · win_rate = สัดส่วนวันที่พลาดน้อยกว่า Naive (ไม่นับวันเสมอ) · dir_hit_rate นับเฉพาะวันที่ราคาขยับจริง · long_signal = วันที่ทำนาย return เกินต้นทุนไป-กลับ (diagnostic ไม่ใช่กำไร)

## next_day

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 1 | 2026-10-01 | 2026-10-01 | 0.007608 | 0.008772 | 0.867323 | 0.247751 | 0.007608 | 2.601969 | 1 | 0 | 0 | 1.000000 |
| ADVANC.BK | Random Forest | 1 | 2026-10-01 | 2026-10-01 | 0.007159 | 0.008772 | 0.816167 | 0.333872 | 0.007159 | 2.448501 | 1 | 0 | 0 | 1.000000 |
| ADVANC.BK | XGBoost | 1 | 2026-10-01 | 2026-10-01 | 0.007008 | 0.008772 | 0.798899 | 0.361760 | 0.007008 | 2.396698 | 1 | 0 | 0 | 1.000000 |
| KBANK.BK | ANN (MLP) | 1 | 2026-10-01 | 2026-10-01 | 0.004733 | 0.004274 | 1.107578 | -0.226729 | 0.004733 | 1.107578 | 0 | 1 | 0 | 0.000000 |
| KBANK.BK | Random Forest | 1 | 2026-10-01 | 2026-10-01 | 0.004471 | 0.004274 | 1.046173 | -0.094477 | 0.004471 | 1.046173 | 0 | 1 | 0 | 0.000000 |
| KBANK.BK | XGBoost | 1 | 2026-10-01 | 2026-10-01 | 0.004840 | 0.004274 | 1.132668 | -0.282937 | 0.004840 | 1.132668 | 0 | 1 | 0 | 0.000000 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.007608 | NaN | NaN | 0 | 1.000000 | 0 | NaN | 0 | False |
| ADVANC.BK | Random Forest | -0.007159 | NaN | NaN | 0 | 1.000000 | 0 | NaN | 0 | False |
| ADVANC.BK | XGBoost | -0.007008 | NaN | NaN | 0 | 1.000000 | 0 | NaN | 0 | False |
| KBANK.BK | ANN (MLP) | 0.004733 | NaN | NaN | 0 | 0.000000 | 0 | NaN | 0 | False |
| KBANK.BK | Random Forest | 0.004471 | NaN | NaN | 0 | 0.000000 | 0 | NaN | 0 | False |
| KBANK.BK | XGBoost | 0.004840 | NaN | NaN | 0 | 0.000000 | 0 | NaN | 0 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 1 | 0 |  | 0 | 0 |
| ADVANC.BK | Random Forest | 1 | 0 |  | 0 | 0 |
| ADVANC.BK | XGBoost | 1 | 0 |  | 0 | 0 |
| KBANK.BK | ANN (MLP) | 1 | 0 |  | 0 | 0 |
| KBANK.BK | Random Forest | 1 | 0 |  | 0 | 0 |
| KBANK.BK | XGBoost | 1 | 0 |  | 0 | 0 |

## same_day_1600

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 1 | 2026-09-30 | 2026-09-30 | 0.000688 | 0.000000 | NaN | NaN | 0.000688 | 0.236766 | 0 | 1 | 0 | 0.000000 |
| ADVANC.BK | Random Forest | 1 | 2026-09-30 | 2026-09-30 | 0.000450 | 0.000000 | NaN | NaN | 0.000450 | 0.154844 | 0 | 1 | 0 | 0.000000 |
| ADVANC.BK | XGBoost | 1 | 2026-09-30 | 2026-09-30 | 0.000635 | 0.000000 | NaN | NaN | 0.000635 | 0.218452 | 0 | 1 | 0 | 0.000000 |
| KBANK.BK | ANN (MLP) | 1 | 2026-09-30 | 2026-09-30 | 0.002960 | 0.004292 | 0.689669 | 0.524357 | 0.002960 | 0.689669 | 1 | 0 | 0 | 1.000000 |
| KBANK.BK | Random Forest | 1 | 2026-09-30 | 2026-09-30 | 0.002575 | 0.004292 | 0.599888 | 0.640134 | 0.002575 | 0.599888 | 1 | 0 | 0 | 1.000000 |
| KBANK.BK | XGBoost | 1 | 2026-09-30 | 2026-09-30 | 0.002456 | 0.004292 | 0.572203 | 0.672584 | 0.002456 | 0.572203 | 1 | 0 | 0 | 1.000000 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 0.000688 | NaN | NaN | 1 | NaN | 0 | NaN | 0 | False |
| ADVANC.BK | Random Forest | 0.000450 | NaN | NaN | 1 | NaN | 0 | NaN | 0 | False |
| ADVANC.BK | XGBoost | 0.000635 | NaN | NaN | 1 | NaN | 0 | NaN | 0 | False |
| KBANK.BK | ANN (MLP) | -0.002960 | NaN | NaN | 0 | 1.000000 | 0 | NaN | 0 | False |
| KBANK.BK | Random Forest | -0.002575 | NaN | NaN | 0 | 1.000000 | 0 | NaN | 0 | False |
| KBANK.BK | XGBoost | -0.002456 | NaN | NaN | 0 | 1.000000 | 0 | NaN | 0 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 2 | 0 |  | 0 | 0 |
| ADVANC.BK | Random Forest | 2 | 0 |  | 0 | 0 |
| ADVANC.BK | XGBoost | 2 | 0 |  | 0 | 0 |
| KBANK.BK | ANN (MLP) | 2 | 0 |  | 0 | 0 |
| KBANK.BK | Random Forest | 2 | 0 |  | 0 | 0 |
| KBANK.BK | XGBoost | 2 | 0 |  | 0 | 0 |
