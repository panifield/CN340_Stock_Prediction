# split_profile — distribution ของ train / val / test (feedback ข้อ 5)

> สร้างโดย `split_profile.py` · ทุกตัวเลขในไฟล์นี้มาจาก `results/split_profile.csv`
> และ `results/split_profile_tests.csv` ที่รันในรอบเดียวกัน

## ขอบเขตของไฟล์นี้ (อ่านก่อน)

ไฟล์นี้อ่านค่า return ของช่วง historical test เพื่ออธิบายข้อมูลเท่านั้น **ไม่มีโมเดลใดถูกประเมินบน test** (ราคาช่วงนี้เป็นข้อมูลสาธารณะและถูกใช้เทรนโมเดล live อยู่แล้ว)

- `split_profile.py` ไม่เรียก `build_features` ไม่ fit ไม่ predict อะไรเลย
- ไม่มีการเลือกโมเดล hyperparameter หรือ feature จากตัวเลขในไฟล์นี้
- `PRE_TEST_LOCK.md` ยังไม่มีในโปรเจกต์ -> historical test **ยังไม่ถูกเปิด** เป็นผลการประเมินของโมเดล และการเปิดเป็นการตัดสินใจของผู้ใช้

## 1. คำถามของอาจารย์: test set ควร balance / ทำไมให้เท่า ๆ กัน

### (1) ทำไม val กับ test มีจำนวนแถวเท่ากัน

เพราะ **ตั้งใจเลือกวันที่ให้เท่ากัน เพื่อให้เทียบผลกับเวอร์ชันก่อนหน้าได้** ไม่ใช่เพราะต้อง balance อะไร · คอมเมนต์ของ `SPLIT_BY_DATE` ใน `config.py` เขียนไว้ตรง ๆ ว่า:

> *วันที่ชุดนี้ทำให้ train/val/test มีขนาดเท่าเวอร์ชันก่อนหน้า (1,688 / 362 / 362) จึงเทียบผลได้ว่าอะไรเปลี่ยนเพราะการแก้บั๊ก ไม่ใช่เพราะ split เลื่อน*

การแบ่งจึง freeze ด้วย **วันที่** (`train_end` 2023-09-01, `val_end` 2025-02-25, ขอบปลาย test `HISTORICAL_TEST_END` 2026-08-28) ไม่ใช่ด้วยสัดส่วน (`TRAIN_RATIO` / `VAL_RATIO` ถูกตั้งเป็น `None` เลิกใช้แล้ว) และห้าม shuffle

### (2) งาน B เป็น regression ไม่มี class ให้ balance

คำว่า balance ใช้กับงานจำแนกประเภทที่มี class ไม่เท่ากัน (เช่น Task A / C ที่ทำนายขึ้น/ลง แล้วต้องดู base rate) แต่งาน B ทำนาย **ค่าต่อเนื่อง** (return รายวัน) จึงไม่มี class ให้นับ

สิ่งที่เทียบกันได้จริงสำหรับ regression และรายงานไว้ข้างล่างคือ:

- **volatility** (`ret_sd`) — ถ้า test ผันผวนกว่า train มาก MAE ทุกตัวจะสูงขึ้นทั้งแถว
- **สัดส่วนวันราคานิ่ง** (`frac_zero_move`) — ยิ่งเยอะ Naive (ทาย 0) ยิ่งได้เปรียบ
- **ทิศทาง** (`frac_up` / `frac_down`) — ใกล้ 50/50 แค่ไหน
- **ระดับราคาและ tick** — ราคาต่างระดับมี tick ต่างกัน -> ขยับ 1 tick คิดเป็น return ไม่เท่ากัน

### (3) สามชุดต่างกันแค่ไหน และผลต่อการตีความ MAE vs Naive

**KBANK.BK**

- ขนาด: train 1,708 / val 362 / test 362 แถว
- SD ของ return: train 0.018943 · val 0.011252 (0.59x ของ train) · test 0.013015 (0.69x ของ train)
- วันราคานิ่งเป๊ะ (return = 0): train 11.4% · val 14.1% · test 15.5%
- Naive MAE_return: train 0.012579 · val 0.008375 · test 0.009226
- ราคาเฉลี่ย: train 159.05 · val 136.95 · test 184.16 บาท · tick ที่พบบ่อยสุด 0.5 / 0.5 / 0.5
- วันที่ราคาขยับน้อยกว่า 1 tick: train 22.1% · val 33.1% · test 27.9%
- KS train vs val: D = 0.1155 (p = 0.000628) · train vs test: D = 0.1230 (p = 0.000211)
- Levene train vs val: p = 4.61e-08 · train vs test: p = 1.69e-05

**ADVANC.BK**

- ขนาด: train 1,708 / val 362 / test 362 แถว
- SD ของ return: train 0.012743 · val 0.013587 (1.07x ของ train) · test 0.013614 (1.07x ของ train)
- วันราคานิ่งเป๊ะ (return = 0): train 13.9% · val 18.5% · test 13.0%
- Naive MAE_return: train 0.009139 · val 0.009441 · test 0.010070
- ราคาเฉลี่ย: train 192.56 · val 237.14 · test 323.07 บาท · tick ที่พบบ่อยสุด 0.5 / 1 / 1
- วันที่ราคาขยับน้อยกว่า 1 tick: train 28.7% · val 31.2% · test 26.2%
- KS train vs val: D = 0.0710 (p = 0.0929) · train vs test: D = 0.0626 (p = 0.184)
- Levene train vs val: p = 0.564 · train vs test: p = 0.072

**ผลต่อการตีความ MAE vs Naive**

1. `naive_mae_return` ของแต่ละชุด **ไม่เท่ากัน** -> ห้ามเทียบ MAE ข้ามชุด (เช่น เอา MAE บน test ไปเทียบกับ MAE บน val แล้วสรุปว่าโมเดลดีขึ้น/แย่ลง) ต้องเทียบกับ Naive **ของชุดเดียวกัน** เสมอ ซึ่งเป็นสิ่งที่ `evaluate.compare_to_baseline` บังคับอยู่แล้ว (baseline อยู่ในตารางเดียวกับโมเดล)
2. ชุดที่ `frac_zero_move` สูงกว่า จะเอื้อ Naive มากกว่าโดยอัตโนมัติ เพราะ Naive ทายค่า 0 แล้วถูกเป๊ะในวันเหล่านั้น (error = 0) ส่วนโมเดล ML ที่ทายค่าไม่เป็นศูนย์ จะเสีย error ทุกวันนิ่ง -- ดู `results/feedback/WHY_NAIVE_WINS.md`
3. ชุดที่ SD สูงกว่าจะมี MAE สูงกว่าทั้งแถว (ทั้งโมเดลและ Naive) -> ตัวเลขที่ ความหมายคงที่ข้ามชุดคือ **relMAE = MAE_model / MAE_naive** ไม่ใช่ MAE ดิบ

## 2. ตารางเต็ม (ต่อหุ้น ต่อชุด)

| ticker | split | n | date_start | date_end | ret_mean | ret_sd | ret_skew | ret_kurtosis_excess | frac_zero_move | frac_up | frac_down | naive_mae_return | price_mean | tick_mode | frac_move_lt_1_tick | frac_move_gt_cost_rt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KBANK.BK | train | 1,708 | 2016-08-29 | 2023-09-01 | -0.000070 | 0.018943 | 0.4456 | 11.4839 | 11.4% | 43.1% | 45.5% | 0.012579 | 159.05 | 0.5 | 22.1% | 80.5% |
| KBANK.BK | val | 362 | 2023-09-04 | 2025-02-25 | 0.000551 | 0.011252 | -0.0001 | 1.1597 | 14.1% | 45.0% | 40.9% | 0.008375 | 136.95 | 0.5 | 33.1% | 77.6% |
| KBANK.BK | test | 362 | 2025-02-26 | 2026-08-28 | 0.001382 | 0.013015 | -0.3020 | 6.6206 | 15.5% | 46.4% | 38.1% | 0.009226 | 184.16 | 0.5 | 27.9% | 67.7% |
| ADVANC.BK | train | 1,708 | 2016-08-29 | 2023-09-01 | 0.000215 | 0.012743 | 0.0202 | 2.4345 | 13.9% | 42.9% | 43.1% | 0.009139 | 192.56 | 0.5 | 28.7% | 70.8% |
| ADVANC.BK | val | 362 | 2023-09-04 | 2025-02-25 | 0.000792 | 0.013587 | 0.7304 | 5.0269 | 18.5% | 43.1% | 38.4% | 0.009441 | 237.14 | 1 | 31.2% | 81.2% |
| ADVANC.BK | test | 362 | 2025-02-26 | 2026-08-28 | 0.000755 | 0.013614 | -0.1392 | 1.4736 | 13.0% | 44.8% | 42.3% | 0.010070 | 323.07 | 1 | 26.2% | 72.1% |

- `frac_move_gt_cost_rt` ใช้ต้นทุนไป-กลับ 33.6 bps (commission 0.1500% ต่อขา -- ดู `REFERENCES.md`)
- `tick_mode` = tick size ที่พบบ่อยสุดในชุดนั้น (จาก `trading_costs.tick_size`) · `frac_move_lt_1_tick` คิด tick รายวันจาก `prev_close` ของวันนั้นก่อนเฉลี่ย
- `ret_kurtosis_excess` เป็น excess kurtosis (การแจกแจงปกติ = 0)
- คอลัมน์ที่ไม่ได้แสดงในตารางนี้แต่มีใน CSV: `ret_min`, `ret_max`, `price_min`, `price_max`, `tick_mode_frac`

## 3. การทดสอบความต่างของ distribution (descriptive)

> **ป้ายกำกับ: descriptive เท่านั้น** ไม่ได้ประกาศล่วงหน้าในแผนใด และไม่ได้ใช้ตัดสินใจอะไร
> return รายวันมี volatility clustering (ไม่ i.i.d.) ซึ่งละเมิดสมมติฐานของทั้ง KS และ Levene -> อ่าน p-value ได้แบบหยาบว่าเล็ก/ไม่เล็กเท่านั้น ห้ามรายงานเป็นหลักฐานทางสถิติที่เข้มงวด

- **Kolmogorov-Smirnov 2 กลุ่ม** (`scipy.stats.ks_2samp`): ทดสอบว่า distribution ทั้งก้อนต่างกันไหม
- **Levene** (`scipy.stats.levene` + `center=median` = Brown-Forsythe ซึ่งทนหางหนา): ทดสอบว่าความแปรปรวนต่างกันไหม

| ticker | comparison | n_train | n_other | ks_stat | ks_pvalue | levene_stat | levene_pvalue | sd_train | sd_other | sd_ratio_other_over_train | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| KBANK.BK | train vs val | 1708 | 362 | 0.1155 | 0.000628 | 30.0981 | 4.61e-08 | 0.018943 | 0.011252 | 0.5940 | descriptive |
| KBANK.BK | train vs test | 1708 | 362 | 0.1230 | 0.000211 | 18.5985 | 1.69e-05 | 0.018943 | 0.013015 | 0.6870 | descriptive |
| ADVANC.BK | train vs val | 1708 | 362 | 0.0710 | 0.0929 | 0.3322 | 0.564 | 0.012743 | 0.013587 | 1.0662 | descriptive |
| ADVANC.BK | train vs test | 1708 | 362 | 0.0626 | 0.184 | 3.2411 | 0.072 | 0.012743 | 0.013614 | 1.0684 | descriptive |

## 4. ข้อจำกัด

- ตัวเลขทั้งหมดเป็น descriptive ของข้อมูลราคา ไม่ใช่ผลของโมเดล
- ชุด train ที่รายงานที่นี่มีทุกแถวที่คำนวณ return ได้ ส่วน train ที่โมเดลเห็นจริง สั้นกว่าเล็กน้อย เพราะ `splits.prepare_xy` ตัดแถว warm-up ของ rolling indicator ทิ้ง (เหลือ 1,688 แถว) -- ขอบวันที่เป็นชุดเดียวกัน ทำให้ val/test ได้แถวเท่ากันเป๊ะ
- ชุด test ถูกตัดที่ `HISTORICAL_TEST_END` จึงเป็นช่วง 362 แถวเดิมเสมอ ไม่เลื่อนตามข้อมูลที่ append เข้ามาใหม่
