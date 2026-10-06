# Live test report — official predictions

> ผลของ **prospective prediction** เท่านั้น · ห้ามรวมกับผล validation / walk-forward เดิม
> แยกตาม prediction_type × หุ้น × โมเดล · ผลจริงจาก raw_data/ (investing.com + yahoo ที่ตรวจ OHLC เป๊ะแล้ว) ไฟล์เดียวกับที่เทรน
> **n < 20 วัน = ผลยังแกว่งมาก อย่าสรุปว่าชนะหรือแพ้**

อ่านอย่างไร: relMAE < 1 = ดีกว่า Naive (ทายว่าราคาไม่เปลี่ยน) · R2_OOS > 0 = MSE ดีกว่า Naive · win_rate = สัดส่วนวันที่พลาดน้อยกว่า Naive (ไม่นับวันเสมอ) · dir_hit_rate นับเฉพาะวันที่ราคาขยับจริง · long_signal = วันที่ทำนาย return เกินต้นทุนไป-กลับ (diagnostic ไม่ใช่กำไร)

## next_day

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 3 | 2026-10-01 | 2026-10-05 | 0.003481 | 0.003890 | 0.894695 | 0.231727 | 0.004675 | 1.193167 | 2 | 1 | 0 | 0.666667 |
| ADVANC.BK | Random Forest | 3 | 2026-10-01 | 2026-10-05 | 0.003345 | 0.003890 | 0.859903 | 0.319823 | 0.004399 | 1.146921 | 2 | 1 | 0 | 0.666667 |
| ADVANC.BK | XGBoost | 3 | 2026-10-01 | 2026-10-05 | 0.003328 | 0.003890 | 0.855428 | 0.341409 | 0.004329 | 1.141066 | 2 | 1 | 0 | 0.666667 |
| KBANK.BK | ANN (MLP) | 3 | 2026-10-01 | 2026-10-05 | 0.006085 | 0.005716 | 1.064417 | -0.018157 | 0.007903 | 1.419979 | 1 | 2 | 0 | 0.333333 |
| KBANK.BK | Random Forest | 3 | 2026-10-01 | 2026-10-05 | 0.005551 | 0.005716 | 0.971015 | 0.189567 | 0.007051 | 1.295642 | 1 | 2 | 0 | 0.333333 |
| KBANK.BK | XGBoost | 3 | 2026-10-01 | 2026-10-05 | 0.006023 | 0.005716 | 1.053695 | 0.017662 | 0.007763 | 1.405743 | 1 | 2 | 0 | 0.333333 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.001633 | 0.119629 | 0.980850 | 1 | 1.000000 | 0 | NaN | 0 | False |
| ADVANC.BK | Random Forest | -0.001618 | 0.181360 | 0.973519 | 1 | 1.000000 | 0 | NaN | 0 | False |
| ADVANC.BK | XGBoost | -0.001565 | 0.195571 | 0.962927 | 1 | 1.000000 | 0 | NaN | 0 | False |
| KBANK.BK | ANN (MLP) | -0.002929 | 0.065569 | -0.075730 | 1 | 0.500000 | 0 | NaN | 0 | False |
| KBANK.BK | Random Forest | -0.002005 | 0.075335 | 0.965613 | 1 | 0.500000 | 0 | NaN | 0 | False |
| KBANK.BK | XGBoost | -0.002326 | 0.019980 | -0.802981 | 1 | 0.500000 | 0 | NaN | 0 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 3 | 0 |  | 0 | 0 |
| ADVANC.BK | Random Forest | 3 | 0 |  | 0 | 0 |
| ADVANC.BK | XGBoost | 3 | 0 |  | 0 | 0 |
| KBANK.BK | ANN (MLP) | 3 | 0 |  | 0 | 0 |
| KBANK.BK | Random Forest | 3 | 0 |  | 0 | 0 |
| KBANK.BK | XGBoost | 3 | 0 |  | 0 | 0 |

## same_day_1600

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 3 | 2026-09-30 | 2026-10-02 | 0.001269 | 0.000966 | 1.313090 | -0.079651 | 0.001739 | 0.437467 | 0 | 3 | 0 | 0.000000 |
| ADVANC.BK | Random Forest | 3 | 2026-09-30 | 2026-10-02 | 0.001147 | 0.000966 | 1.187439 | -0.017775 | 0.001688 | 0.395663 | 1 | 2 | 0 | 0.333333 |
| ADVANC.BK | XGBoost | 3 | 2026-09-30 | 2026-10-02 | 0.001122 | 0.000966 | 1.161410 | 0.123724 | 0.001567 | 0.386925 | 1 | 2 | 0 | 0.333333 |
| KBANK.BK | ANN (MLP) | 3 | 2026-09-30 | 2026-10-02 | 0.003825 | 0.004256 | 0.898749 | 0.171498 | 0.003874 | 0.899598 | 2 | 1 | 0 | 0.666667 |
| KBANK.BK | Random Forest | 3 | 2026-09-30 | 2026-10-02 | 0.003521 | 0.004256 | 0.827311 | 0.290795 | 0.003584 | 0.828398 | 3 | 0 | 0 | 1.000000 |
| KBANK.BK | XGBoost | 3 | 2026-09-30 | 2026-10-02 | 0.003482 | 0.004256 | 0.818127 | 0.292812 | 0.003579 | 0.819876 | 2 | 1 | 0 | 0.666667 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.000682 | 0.219054 | -0.736086 | 2 | 0.000000 | 0 | NaN | 1 | False |
| ADVANC.BK | Random Forest | -0.000847 | 0.174797 | -0.319442 | 2 | 1.000000 | 0 | NaN | 1 | False |
| ADVANC.BK | XGBoost | -0.000699 | 0.218172 | -0.012099 | 2 | 1.000000 | 0 | NaN | 1 | False |
| KBANK.BK | ANN (MLP) | 0.001851 | 0.153991 | 0.998740 | 0 | 0.666667 | 0 | NaN | 1 | False |
| KBANK.BK | Random Forest | 0.001804 | 0.230623 | 0.997416 | 0 | 1.000000 | 0 | NaN | 1 | False |
| KBANK.BK | XGBoost | 0.001844 | 0.266155 | 0.918680 | 0 | 0.666667 | 0 | NaN | 1 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 4 | 0 |  | 0 | 0 |
| ADVANC.BK | Random Forest | 4 | 0 |  | 0 | 0 |
| ADVANC.BK | XGBoost | 4 | 0 |  | 0 | 0 |
| KBANK.BK | ANN (MLP) | 4 | 0 |  | 0 | 0 |
| KBANK.BK | Random Forest | 4 | 0 |  | 0 | 0 |
| KBANK.BK | XGBoost | 4 | 0 |  | 0 | 0 |
