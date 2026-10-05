# Live test report — official predictions

> ผลของ **prospective prediction** เท่านั้น · ห้ามรวมกับผล validation / walk-forward เดิม
> แยกตาม prediction_type × หุ้น × โมเดล · ผลจริงจาก raw_data/ (investing.com + yahoo ที่ตรวจ OHLC เป๊ะแล้ว) ไฟล์เดียวกับที่เทรน
> **n < 20 วัน = ผลยังแกว่งมาก อย่าสรุปว่าชนะหรือแพ้**

อ่านอย่างไร: relMAE < 1 = ดีกว่า Naive (ทายว่าราคาไม่เปลี่ยน) · R2_OOS > 0 = MSE ดีกว่า Naive · win_rate = สัดส่วนวันที่พลาดน้อยกว่า Naive (ไม่นับวันเสมอ) · dir_hit_rate นับเฉพาะวันที่ราคาขยับจริง · long_signal = วันที่ทำนาย return เกินต้นทุนไป-กลับ (diagnostic ไม่ใช่กำไร)

## next_day

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 2 | 2026-10-01 | 2026-10-02 | 0.003835 | 0.004386 | 0.874326 | 0.247702 | 0.005380 | 1.311581 | 1 | 1 | 0 | 0.500000 |
| ADVANC.BK | Random Forest | 2 | 2026-10-01 | 2026-10-02 | 0.003722 | 0.004386 | 0.848647 | 0.332817 | 0.005066 | 1.273398 | 1 | 1 | 0 | 0.500000 |
| ADVANC.BK | XGBoost | 2 | 2026-10-01 | 2026-10-02 | 0.003670 | 0.004386 | 0.836709 | 0.360330 | 0.004961 | 1.255562 | 1 | 1 | 0 | 0.500000 |
| KBANK.BK | ANN (MLP) | 2 | 2026-10-01 | 2026-10-02 | 0.008779 | 0.008575 | 1.023900 | -0.015533 | 0.009667 | 2.047978 | 1 | 1 | 0 | 0.500000 |
| KBANK.BK | Random Forest | 2 | 2026-10-01 | 2026-10-02 | 0.007902 | 0.008575 | 0.921583 | 0.193472 | 0.008615 | 1.843433 | 1 | 1 | 0 | 0.500000 |
| KBANK.BK | XGBoost | 2 | 2026-10-01 | 2026-10-02 | 0.008683 | 0.008575 | 1.012595 | 0.020361 | 0.009495 | 2.025447 | 1 | 1 | 0 | 0.500000 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.003835 | 0.139680 | NaN | 1 | 1.000000 | 0 | NaN | 0 | False |
| ADVANC.BK | Random Forest | -0.003722 | 0.216313 | NaN | 1 | 1.000000 | 0 | NaN | 0 | False |
| ADVANC.BK | XGBoost | -0.003670 | 0.238911 | NaN | 1 | 1.000000 | 0 | NaN | 0 | False |
| KBANK.BK | ANN (MLP) | -0.004046 | 0.023900 | NaN | 0 | 0.500000 | 0 | NaN | 0 | False |
| KBANK.BK | Random Forest | -0.003431 | 0.078417 | NaN | 0 | 0.500000 | 0 | NaN | 0 | False |
| KBANK.BK | XGBoost | -0.003842 | 0.012595 | NaN | 0 | 0.500000 | 0 | NaN | 0 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 2 | 0 |  | 0 | 0 |
| ADVANC.BK | Random Forest | 2 | 0 |  | 0 | 0 |
| ADVANC.BK | XGBoost | 2 | 0 |  | 0 | 0 |
| KBANK.BK | ANN (MLP) | 2 | 0 |  | 0 | 0 |
| KBANK.BK | Random Forest | 2 | 0 |  | 0 | 0 |
| KBANK.BK | XGBoost | 2 | 0 |  | 0 | 0 |

## same_day_1600

### ความแม่นเทียบ Naive

| ticker | model | n | first_date | last_date | MAE_return | MAE_naive | relMAE | R2_OOS | RMSE_return | MAE_baht | wins | losses | ties | win_rate_ex_ties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 2 | 2026-09-30 | 2026-10-01 | 0.001807 | 0.001449 | 1.246849 | -0.075264 | 0.002125 | 0.623081 | 0 | 2 | 0 | 0.000000 |
| ADVANC.BK | Random Forest | 2 | 2026-09-30 | 2026-10-01 | 0.001669 | 0.001449 | 1.151469 | -0.016481 | 0.002066 | 0.575510 | 1 | 1 | 0 | 0.500000 |
| ADVANC.BK | XGBoost | 2 | 2026-09-30 | 2026-10-01 | 0.001636 | 0.001449 | 1.128592 | 0.124801 | 0.001917 | 0.563979 | 1 | 1 | 0 | 0.500000 |
| KBANK.BK | ANN (MLP) | 2 | 2026-09-30 | 2026-10-01 | 0.003643 | 0.004283 | 0.850590 | 0.251078 | 0.003706 | 0.850935 | 1 | 1 | 0 | 0.500000 |
| KBANK.BK | Random Forest | 2 | 2026-09-30 | 2026-10-01 | 0.003257 | 0.004283 | 0.760567 | 0.396133 | 0.003328 | 0.760912 | 2 | 0 | 0 | 1.000000 |
| KBANK.BK | XGBoost | 2 | 2026-09-30 | 2026-10-01 | 0.002980 | 0.004283 | 0.695808 | 0.500876 | 0.003026 | 0.696074 | 2 | 0 | 0 | 1.000000 |

### รูปร่างคำทำนาย ทิศทาง และสัญญาณ

| ticker | model | Bias | StdRatio | Rho | zero_move_days | dir_hit_rate_moved_days | long_signal_days | long_signal_up_rate | prev_close_mismatch | enough_data |
|---|---|---|---|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | -0.001119 | 0.246849 | NaN | 1 | 0.000000 | 0 | NaN | 1 | False |
| ADVANC.BK | Random Forest | -0.001219 | 0.151469 | NaN | 1 | 1.000000 | 0 | NaN | 1 | False |
| ADVANC.BK | XGBoost | -0.001001 | 0.128592 | NaN | 1 | 1.000000 | 0 | NaN | 1 | False |
| KBANK.BK | ANN (MLP) | 0.000683 | 0.149410 | NaN | 0 | 0.500000 | 0 | NaN | 0 | False |
| KBANK.BK | Random Forest | 0.000683 | 0.239433 | NaN | 0 | 1.000000 | 0 | NaN | 0 | False |
| KBANK.BK | XGBoost | 0.000524 | 0.304192 | NaN | 0 | 1.000000 | 0 | NaN | 0 | False |

### ความครบและตรงเวลา

| ticker | model | predictions | missing_trading_days | missing_dates | early_predictions | late_predictions |
|---|---|---|---|---|---|---|
| ADVANC.BK | ANN (MLP) | 3 | 0 |  | 0 | 0 |
| ADVANC.BK | Random Forest | 3 | 0 |  | 0 | 0 |
| ADVANC.BK | XGBoost | 3 | 0 |  | 0 | 0 |
| KBANK.BK | ANN (MLP) | 3 | 0 |  | 0 | 0 |
| KBANK.BK | Random Forest | 3 | 0 |  | 0 | 0 |
| KBANK.BK | XGBoost | 3 | 0 |  | 0 | 0 |
