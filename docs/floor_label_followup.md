# เปลี่ยน label คู่/คี่เป็น floor และงานต่อเนื่อง (A / A2)

วันที่ 24 ต.ค. 2026 · branch `floor-label` (ยังไม่ merge เข้า main และยังไม่ deploy ขึ้น Railway)

## สรุปสั้น

- **label ใหม่:** คู่/คี่ของส่วนจำนวนเต็มบาท (floor) เช่น 162.50 → 162 = คู่ คำนวณที่ `rounding.parity_label()` จุดเดียว ทุกสคริปต์เรียกฟังก์ชันนี้
- **ผลหลังเปลี่ยน label ไม่ต่างจากเดิม:** ไม่มีโมเดลไหนชนะ persistence อย่างมีนัยสำคัญ ทั้งงาน A และ A2 ทั้งชุดพัฒนาและ test
- **persistence ของ A2 KBANK ลดลงเมื่อใช้ floor:** ชุดพัฒนาจาก 66.8% เหลือ 62.9% (n = 410) ส่วนช่วงราคา 100–200 บาทได้ 62.5% (n = 424) ซึ่งต่ำกว่า H1 ของอีก repo (66.4%, n = 244) เล็กน้อย
- **กฎ H1 อยู่ในระบบ live แล้ว แต่จะ abstain ทุกวันในตอนนี้:** ราคาทั้งสองหุ้นอยู่เหนือ 200 บาท (KBANK 242, ADVANC 336) ซึ่งเป็นช่วง tick 1 บาท
- **โมเดล Δtick ไม่ชนะ persistence:** ใน KBANK ช่วง 200–400 บาทดีกว่า persistence (53.0% เทียบ 47.0%, n = 83) แต่ไม่มีนัยสำคัญ (p = 0.42)
- **ข้อมูล Settrade 1 นาทีมีแค่ 15 วัน:** สร้าง pipeline และ leak test ไว้แล้ว ยังไม่สรุปผล ซ่อมตัวเก็บข้อมูลแล้ว

---

## 1. เปลี่ยน label เป็น floor

### กฎ

```
parity_label(price) = (round(price × 100) // 100) % 2      0 = คู่, 1 = คี่
```

แปลงเป็นสตางค์ก่อนเพื่อกันเศษ float ดังนั้น 162.99999999 จึงนับเป็น 163 = คี่ ทีมตัดสินใจเรื่องนี้เมื่อ 24 ต.ค. เพราะเทสในสเปกเดิมคาดว่าเป็นคู่ ซึ่งขัดกับสูตร

| ราคา | 162.0 | 162.5 | 163.0 | 199.5 | 162.99999999 |
|---|---|---|---|---|---|
| ผล | คู่ | คู่ | คี่ | คี่ | คี่ |

### สคริปต์ที่เปลี่ยน

| กลุ่ม | ไฟล์ | สิ่งที่เปลี่ยน |
|---|---|---|
| แกนกลาง | `task_a/rounding.py` | เพิ่ม `parity_label()` และ `floor_baht()` · `to_int_baht()` เดิมเก็บไว้เทียบเท่านั้น |
| | `task_a/config.py` | อธิบายกฎใหม่ · เพิ่ม `TASK_A_OUTPUT_DIR` |
| | `task_a/targets.py` | `y_parity` และ `int_price` ใช้ floor → label "พลิก/ไม่พลิก" และ Persistence เปลี่ยนตาม |
| | `task_a2/features_a2.py` | label ของราคาปิดและราคา 16:00 ใช้ floor · `dist_to_boundary_1600` วัดระยะถึง x.00 (เดิม x.50) |
| live | `task_a/predict_live.py` | เพิ่มคอลัมน์ `label_rule` |
| | `task_a2/predict_live.py` | label floor · เพิ่มคอลัมน์ `label_rule`, `tick_1600`, `h1_status`, `h1_predicted_parity` |
| | `task_a/evaluate_live.py` | รายงานทั้งกฎเดิมและ floor · เพิ่มสรุป H1 |
| | `task_a/log_schema.py` (ใหม่) | เพิ่มคอลัมน์ให้ log โดยต่อ "," ท้ายบรรทัด แถวเก่าไม่เปลี่ยนแม้แต่ byte เดียว |
| วิจัย | `experiment_matrix.py`, `five_analyses.py`, `dev_report.py`, `regime_baseline.py`, `run_permutation.py`, `run_permutation_fixed.py`, `improve_task_a.py`, `scan_universe.py`, `tick_utils.py`, `task_a2/build_price_at_1600.py` | เรียก `parity_label` · สคริปต์ที่มีตัวเลือกกฎปัดใช้ `floor` เป็นค่าเริ่มต้น (gt/gte ยังเลือกได้) |
| | `main.py`, `main_a2.py` | เพิ่มเฉพาะการเลือกโฟลเดอร์ผลลัพธ์ และบันทึกตาราง walk-forward เป็น CSV |
| backfill | `compare_train_window.py`, `rolling_backtest.py` | เรียก `parity_label` · `run_one.py` ได้ label ใหม่ผ่าน `predict_one_ticker` |

งาน A ไม่มี feature "ระยะห่างจากจุดปัด" `dist_to_boundary` ของงาน A วัดระยะถึงขอบช่วง tick (100/200/400 บาท) จึงไม่เปลี่ยน

### ผลเดิมเทียบกับใหม่

ทุกชุดใช้ข้อมูลและการแบ่งแบบเดียวกัน
- **ผลเดิม:** รันโค้ด commit `c87a7f2` ใหม่ใน worktree แยก
- **ผลใหม่:** อยู่ที่ `task_a/results_floor/` และ `task_a2/results_floor/`

**งาน A: walk-forward OOF (n = 1,815) และ test (n = 365)**

| หุ้น | | กฎเดิม | floor |
|---|---|---|---|
| KBANK | persistence (OOF) | 48.9% | 50.1% |
| | โมเดลที่เลือก (OOF) | Logistic Regression 50.2% | LightGBM 49.6% |
| | test: โมเดลที่เลือก / persistence | 50.1% / 52.1% | 50.1% / 49.9% |
| ADVANC | persistence (OOF) | 50.9% | 50.9% |
| | โมเดลที่เลือก (OOF) | ANN 52.3% | LightGBM 51.3% |
| | test: โมเดลที่เลือก / persistence / majority | 57.3% / 50.4% / 54.0% | 48.5% / 50.4% / 54.0% |

**งาน A2: walk-forward (n = 410) และ test (n = 140)**

| หุ้น | | กฎเดิม | floor |
|---|---|---|---|
| KBANK | persistence-1600 (CV) | 66.8% | 62.9% |
| | LightGBM (CV) | 66.1% | 59.0% |
| | test: โมเดล / persistence | 50.0% / 50.0% | 54.3% / 55.0% |
| ADVANC | persistence-1600 (CV) | 52.4% | 52.4% |
| | โมเดลที่เลือก (CV) | ANN 51.0% | LightGBM 49.8% |
| | test: โมเดล / persistence | 41.4% / 53.6% | 45.0% / 53.6% |

ในช่วงข้อมูลของ A2 (ต.ค. 2023 เป็นต้นไป) ADVANC อยู่เหนือ 200 บาทเกือบทุกวัน ราคาจึงเป็นจำนวนเต็ม กฎปัดกับ floor ให้คู่/คี่ตรงกันทุกวัน ตัวเลขของ persistence จึงไม่เปลี่ยน ส่วนที่ต่างกันมาจากโมเดลเท่านั้น

**การวิเคราะห์อื่น**

| การวิเคราะห์ | กฎเดิม | floor |
|---|---|---|
| Experiment matrix (96 การทดลอง, FDR) | มีนัยสำคัญ 2 ตัว คือ oracle ทั้งสองตัว · feature จริงดีสุด 53.2% | มีนัยสำคัญ 2 ตัว คือ oracle ทั้งสองตัว (65–71%) · feature จริงดีสุด 53.1% (p หลัง Bonferroni = 0.94) |
| AUC ของ P(พลิก) | 0.48–0.57 | 0.47–0.53 |
| Threshold sweep | บน holdout ไม่ช่วย | บน holdout ไม่ช่วย |
| Pretrain → fine-tune | ไม่ชนะ persistence (p ≥ 0.16) | ไม่ชนะ persistence · ดีสุดคือ KBANK LightGBM +50 ต้น 52.5% เทียบ 48.5% (p = 0.32) |
| Optuna (59 trial ต่อชุด, Bonferroni × 4) | ไม่ผ่านทั้ง 4 ชุด | ไม่ผ่านทั้ง 4 ชุด (p บน holdout ก่อนปรับ ≥ 0.49) |
| Backfill งาน A (12 วัน-หุ้น) | persistence 8/12 · โมเดล 4–7/12 | persistence 8/12 · โมเดล 5–6/12 |
| Backfill งาน A2 (14 วัน-หุ้น) | persistence 6/14 · โมเดล 8–9/14 | persistence 6/14 · โมเดล 8–10/14 |

เปลี่ยนไปใช้ floor แล้วข้อสรุปไม่เปลี่ยน ยังไม่มีหลักฐานว่าโมเดลทายได้ดีกว่า baseline

### ระบบ live

- **ยังไม่ได้ deploy:** โค้ดอยู่บน branch ในเครื่อง Railway ยังรันโค้ดเดิม
- **log เพิ่มคอลัมน์เมื่อรันครั้งแรก:** `prediction_log.csv` จะได้คอลัมน์ใหม่ต่อท้ายตอนที่โค้ดใหม่รันครั้งแรก แถวเก่าเป็นค่าว่าง เทสตรวจกับ log จริงแล้วว่าค่าเดิมไม่เปลี่ยน
- **`outcomes.csv` (ทดสอบแล้ว):** คอลัมน์เดิมอยู่ลำดับเดิม ค่าทุกแถวเหมือนเดิม คอลัมน์ใหม่ `actual_parity_floor`, `correct_floor` และ `baseline_correct_floor` ต่อท้ายและเป็นค่าว่างในแถวเก่า
  - `actual_parity` (กฎเดิม) ยังคำนวณทุกแถวไว้เทียบ
  - `correct` ตามกฎเดิมคำนวณเฉพาะแถวเดิม
- **`evaluate_live.py`:** รายงานแยก 2 ส่วน คือ "label เดิม" และ "label floor"

## 2. กฎ H1 ในระบบ live

- **H1 (รอบ 16:00):** ถ้า tick ของราคา ณ 16:00 เท่ากับ 0.50 (ช่วง 100–200 บาท) จะทายว่าคู่/คี่เท่ากับ `parity_label(ราคา 16:00)` ถ้าไม่ใช่จะบันทึก `abstain` กฎนี้อยู่ใน `task_a2/predict_live.py` ฟังก์ชัน `h1_rule()` และห้ามปรับตามผล log
- **`evaluate_live.py`:** รายงาน coverage (วัน-หุ้นที่ไม่ abstain) และ accuracy ของ H1 แยกต่างหาก
- **สถานะตอนนี้:** dry-run วันที่ 22 ต.ค. ได้ abstain ทั้ง 2 หุ้น เพราะ KBANK อยู่ที่ 242 และ ADVANC อยู่ที่ 336 coverage จะเป็น 0% จนกว่าราคาจะกลับเข้าช่วง 100–200 บาท
- **ข้อมูลย้อนหลังเทียบกับ H1:** งานนี้ไม่ได้ใช้ทดสอบ H1 ใหม่ แต่ persistence-1600 ของ KBANK ช่วง 100–200 บาทตาม floor ได้ 62.5% บนชุดพัฒนา (n = 424) และ 63.9% บน holdout (n = 61)
- **H2 (16:30) บันทึกเป็นแผนไว้ก่อน:** ระบบตอนนี้ทายแค่รอบ 16:02 ถ้าจะเพิ่มรอบ 16:30 ต้องแยก log และตัวกันเวลาออกจากรอบเดิม ต้องถามทีมก่อนเพิ่ม แท่ง 1 นาทีของ Settrade มีถึง 16:29 จึงพอเป็นข้อมูลให้ H2 ได้

## 3. โมเดลทายจำนวน tick ที่ขยับ (เฉพาะ A2)

`task_a2/dtick_model.py` ผลอยู่ที่ `task_a2/results_floor/dtick/`

- **target:** Δtick = ตำแหน่งบน tick grid ของราคาปิดทางการ ลบด้วยของราคา ณ 16:00 ตัดเป็น 5 คลาส {≤−2, −1, 0, +1, ≥+2}
  - นับข้ามช่วงราคาได้ถูก เช่น 199.50 → 201 = +2
  - ราคาปิดมาจาก `verified_daily`
- **P(คี่):** แปลงแต่ละคลาสเป็นราคาปิดที่เป็นไปได้แล้วใช้ `parity_label` ส่วนคลาสปลายเปิดใช้สัดส่วน Δtick จริงจากชุด train ของ fold นั้นเท่านั้น
- **ตาราง tick:** ตรวจกับหน้า SET "Price bands and tick sizes" ได้ 25–100 = 0.25, 100–200 = 0.50, 200–400 = 1.00, ≥ 400 = 2.00 และราคา OHLC รายวันกับแท่ง 1 ชั่วโมงทุกแถวตั้งแต่ 2016 อยู่บน grid นี้ มีข่าวว่า SET จะลด tick ของช่วง 5–50 บาท แต่ไม่กระทบสองหุ้นนี้
- **feature:** ทุกตัวเป็นหน่วย tick เช่น ระยะถึง x.00 การขยับ 1 ชั่วโมงและ 2 ชั่วโมงล่าสุด ช่วงราคาของวัน gap ตอนเปิด และ feature รายวันของงาน A ไม่ใช้ข้อมูลตั้งแต่แท่ง 16:00
- **การแบ่งข้อมูล:** แถวและการแบ่งเดียวกับ A2 เดิม คือชุดพัฒนา 576 วันและ holdout 144 วัน เลือกโมเดลจาก OOF แล้วดู holdout ครั้งเดียว Bonferroni × 4 (ต่อหุ้นมี Δtick ที่เลือกกับ flip ที่เลือก)

**การกระจายของ Δtick**

| หุ้น / ช่วงราคา | n | ≤−2 | −1 | 0 | +1 | ≥+2 | อัตราการพลิก |
|---|---|---|---|---|---|---|---|
| KBANK 100–200 | 635 | 4% | 23% | 41% | 24% | 8% | 35.6% |
| KBANK 200–400 | 85 | 4% | 31% | 44% | 20% | 2% | 51.8% |
| ADVANC 200–400 | 708 | 8% | 24% | 38% | 21% | 9% | 47.3% |

ขนาดการขยับของสองช่วงราคาใกล้กัน แต่เมื่อ tick เป็น 1 บาท ทุก tick ที่ขยับทำให้คู่/คี่พลิก ส่วนเมื่อ tick เป็น 0.50 การขยับครึ่งหนึ่งไม่ทำให้พลิก persistence จึงเหลือราว 50% ในช่วง 200–400 บาท

**ผลบน holdout (โมเดลที่เลือกจากชุดพัฒนา)**

| หุ้น | ช่วงราคา | n | Persistence-1600 | Δtick | Flip (เดิม) |
|---|---|---|---|---|---|
| KBANK | ทั้งหมด | 144 | 54.2% | LightGBM 55.6% (p = 0.85, หลังปรับ 1.00) | LightGBM 54.9% |
| | 100–200 | 61 | 63.9% | 59.0% | 65.6% |
| | 200–400 | 83 | 47.0% | 53.0% (p = 0.42) | 47.0% |
| ADVANC | ทั้งหมด | 144 | 51.4% | LogReg 55.6% (p = 0.54, หลังปรับ 1.00) | ANN 43.8% |
| | 200–400 | 142 | 50.7% | 54.9% | 43.0% |

บนชุดพัฒนา Δtick ก็ไม่ชนะ persistence (KBANK 61.9% เทียบ 62.6% · ADVANC 53.2% เทียบ 52.9%)

**อ่านผล:**
- Δtick ลดความเสียหายช่วง regime shift ของ KBANK ได้ในเชิงตัวเลข แต่ยังไม่มีนัยสำคัญ (n = 83)
- KBANK เพิ่งขึ้นเกิน 200 บาทในช่วง holdout ใน OOF ของชุดพัฒนามีวันที่ tick 1 บาทแค่ 1 วัน โมเดลจึงแทบไม่เคยเห็น regime นี้ตอนเทรน
- ถ้าจะลองต่อ ควรเทรนรวมข้อมูล ADVANC ซึ่งอยู่ช่วง tick 1 ตลอด ต้องวางแผนและแก้ p ใหม่ก่อน

## 4. ข้อมูล 1 นาทีของ Settrade

**สำรวจข้อมูล**
- **ที่อยู่ไฟล์:** ไฟล์อยู่ใน `group_proj` เก่า คัดลอกมาไว้ที่ `task_a2/data/settrade_log/` แล้ว
- **จำนวนวัน:** แท่ง 1 นาทีหุ้นละ **15 วัน** ตั้งแต่ 28 ก.ย. ถึง 22 ต.ค. ขาด 8, 15 และ 16 ต.ค.
- **คอลัมน์:** time, open, high, low, close, volume, dt
- **ไม่มี bid/ask และไม่มีรายการซื้อขายทีละ tick**
- **แท่ง 16:35 = ATC:** ตรงกับราคาปิดทางการ

**ตัวเก็บข้อมูลหยุดตั้งแต่ 21 ต.ค. แก้แล้ว**

| สาเหตุ | วิธีแก้ |
|---|---|
| scheduled task ชี้ไป `group_proj` | ชี้มาที่ repo นี้ |
| ไม่รันตอนใช้แบต และถูกฆ่าเมื่อถอดสายชาร์จ | เปิดให้รันบนแบต |
| เครื่อง sleep ตอน 17:30 | เปิด WakeToRun |
| process ถูก Ctrl+C ฆ่ากลางทาง (0xC000013A) | รันผ่าน `pythonw.exe` ที่ไม่มีหน้าต่าง console และเขียน log เอง ทดสอบรันแล้วจบปกติ (exit 0) |

- **เติมวันที่ขาด:** `collect_settrade_daily.py` เติมวันที่ขาดย้อนหลัง 14 วันได้ ดึง 21–22 ต.ค. คืนมาได้ แต่ 15–16 ต.ค. ได้แค่แท่ง 5 นาที เพราะ Settrade เก็บแท่ง 1 นาทีไว้ราว 1 สัปดาห์
- **ข้อควรระวัง:** ถ้า checkout กลับไป `main` ก่อน merge ตัวเก็บจะยังรันได้ แต่จะไม่มีความสามารถเติมวันที่ขาด

**pipeline:** อยู่ที่ `task_a2/settrade_features.py` ใช้เฉพาะแท่งที่ dt < 16:00
- `last_tick_dir`: uptick / downtick / เท่าเดิม ประมาณจากแท่ง 1 นาทีสุดท้าย
- `last_change_dir`: ทิศของการเปลี่ยนราคาครั้งล่าสุด (tick test)
- `dticks_5m`, `dticks_15m`, `dticks_30m`: จำนวน tick ที่ขยับในช่วงนั้น
- `close_at_bar_low/high`: เป็นแค่ **ตัวแทน** bid/ask ไม่ใช่ข้อมูลจริง ราคาเด้งระหว่าง 2 ระดับทุกนาที

**leak test (`tests/test_settrade_features.py`):**
- วางยาแท่งตั้งแต่ 16:00 แล้ว feature ไม่เปลี่ยน ทดสอบทั้งข้อมูลสังเคราะห์และไฟล์จริงทุกวัน
- วางยาแท่ง 15:59 แล้ว feature เปลี่ยน

**ยังไม่สรุปผล:** มีแค่ 15 วัน ต่ำกว่าเกณฑ์ 200 วันมาก (power analysis ต้องใช้ 194 วันถึงจะยืนยันความต่าง 10 จุดได้) ถ้าเก็บได้ทุกวันทำการจะครบ 200 วันราวเดือนสิงหาคม 2027

## 5. เทส

| ชุด | ผล |
|---|---|
| task_a/tests | 26 ผ่าน (เพิ่ม `test_parity_label.py` และ `test_floor_live.py`) |
| task_a2/tests | 28 ผ่าน (เพิ่ม `test_floor_h1.py`, `test_dtick_model.py` และ `test_settrade_features.py`) |
| task_c/tests | 15 ผ่าน |
| automation/tests | 9 ผ่าน |
| backfill leak test (เทียบกับ backfill ชุด floor, 29 ก.ย. / 7 ต.ค. / 21 ต.ค.) | **ผ่าน** · rerun noise = 0 ทุกชุด · future 5/5 ผลเหมือนเดิมเป๊ะ (ต่าง 0) · control 5/5 ผลเปลี่ยน |

หมายเหตุเรื่อง leak test: รอบแรกรันเทียบกับ `backfill/results/` (ชุดกฎเดิม) แม้แต่ rerun ที่ไม่วางยาก็ต่างไป 1.0
ผลรอบนั้นจึงใช้ไม่ได้ เก็บไว้ที่ `task_a/results_floor/logs/leak_test_report_vs_old_backfill_INVALID.csv`
รอบที่ใช้ได้รันด้วย `BACKFILL_RESULTS_DIR=backfill/results_floor/results` อยู่ที่ `task_a/results_floor/logs/leak_test_report_floor.csv`
ไฟล์ `backfill/leak_test/leak_test_report.csv` ของทีมคืนเป็นเวอร์ชันเดิมแล้ว

## ก่อน deploy ขึ้น Railway

1. ทีมรีวิวผลในเอกสารนี้ แล้ว merge `floor-label` เข้า `main` (Railway deploy ใหม่ทุกครั้งที่ push)
2. รอบแรกหลัง deploy ตรวจว่า `prediction_log.csv` ทั้งสองไฟล์ได้คอลัมน์ใหม่ และแถวเก่ายังเหมือนเดิม
3. ข้อมูล Settrade: ตัดสินใจแล้วว่าไม่เก็บใน repo · `task_a2/data/settrade_log/` และ `task_a2/data/*_1m_atc_check.csv` อยู่ใน `.gitignore` แล้ว · ไฟล์ `*_1m_atc_check.csv` กับ `scheduler_run.log` ที่เคย commit ไว้ (`1b71e78`) ถูกเลิก track ใน branch นี้ (`git rm --cached`, ไฟล์ยังอยู่ในเครื่อง) แต่ยังอยู่ในประวัติ git เพราะไม่แก้ประวัติ

## ตรวจความพร้อมก่อน merge (24 ต.ค. 2026)

| รายการ | ผล |
|---|---|
| เทสทั้งหมด | 78 ผ่าน (task_a 26 · task_a2 28 · task_c 15 · automation 9) |
| dry-run A (target 26 ต.ค.) และ A2 (target 22 ต.ค.) | ผ่าน · เขียนลง `results/dryrun/prediction_log_dryrun.csv` เท่านั้น |
| hash ของ log จริงและ outcomes ก่อนและหลัง dry-run / evaluate | ไม่เปลี่ยนทั้ง 4 ไฟล์ |
| คอลัมน์ใหม่ในไฟล์ dry-run | ครบ · A: `label_rule` = floor · A2: `label_rule`, `tick_1600` = 1.0, `h1_status` = abstain, `h1_predicted_parity` = ว่าง |
| จำลองรันครั้งแรกบนสำเนา log จริง | ส่วนแถวเก่าเหมือนเดิมทุก byte (A 74 แถว · A2 66 แถว) |
| `evaluate_live.py` บนสำเนา | รายงานครบ "label เดิม" · "label floor" · "กฎ H1" (coverage 0/2 · ยังไม่มี accuracy เพราะ abstain) |

sha256 ของไฟล์จริงใน working tree (CRLF บน Windows) ตอนตรวจ:

| ไฟล์ | แถว | sha256 |
|---|---|---|
| `task_a/results/prediction_log.csv` | 74 | `994afe7558f47d9bfdb1a9f1a282d85feee52280966a815fef91601db36605f1` |
| `task_a2/results/prediction_log.csv` | 66 | `e1ee0026f935795ce1531f153c283401e3e5b104a79fb7ff6826a9cb57967306` |
| `task_a/results/outcomes.csv` | 74 | `c965322f5e39485e2e658c34780907cb0858d56608c978a4c8a9f7c17be476a9` |
| `task_a2/results/outcomes.csv` | 66 | `027f6b83263a4f781ecb19340dc0ddd4f61da650b8d66bb8867d013684651a58` |

## ตรวจหลัง deploy

ส่วนแถวเก่าของ log วัดด้วย `task_a/log_schema.py` (แปลง CRLF เป็น LF ก่อน ใช้ได้ทั้ง Windows และ Railway)
ค่าที่คาดคือ log ใน `main` ณ 24 ต.ค. (`c87a7f2`):

| log | แถวเก่า | คอลัมน์ใหม่ | sha256 ของส่วนแถวเก่า |
|---|---|---|---|
| A | 74 | 1 | `c8740622137cc7f3710d2e785f24460b292dee5fc2c351e47a830aee42a5c41f` |
| A2 | 66 | 4 | `fe26c867366d3b368b84d8a22a6793ea1453af8c4ba3ab8bcf50f80aeac4e62e` |

ถ้า Railway เขียน log เพิ่มเข้า `main` ก่อน merge (เช่นรอบวันจันทร์) ค่าข้างบนจะใช้ไม่ได้ ให้คำนวณใหม่ทันทีก่อน merge:

```bash
git fetch origin
git show origin/main:task_a/results/prediction_log.csv | tail -n +2 | wc -l    # จำนวนแถวเก่า
git show origin/main:task_a/results/prediction_log.csv | sha256sum            # sha256 ที่คาด
# ทำซ้ำกับ task_a2/results/prediction_log.csv
```

หลัง deploy:

1. **รอบแรกของ A (ราว 08:03)** หลัง pull log ใหม่:
   `python task_a/log_schema.py task_a/results/prediction_log.csv 74 1 c8740622137cc7f3710d2e785f24460b292dee5fc2c351e47a830aee42a5c41f`
   ต้องได้ "OK" · header ลงท้ายด้วย `label_rule` · แถวใหม่ทุกแถว `label_rule` = floor
2. **รอบแรกของ A2 (ราว 16:03)**:
   `python task_a/log_schema.py task_a2/results/prediction_log.csv 66 4 fe26c867366d3b368b84d8a22a6793ea1453af8c4ba3ab8bcf50f80aeac4e62e`
   ต้องได้ "OK" · header ลงท้ายด้วย `label_rule,tick_1600,h1_status,h1_predicted_parity`
3. **H1:** ตราบใดที่ราคายังเกิน 200 บาท `h1_status` ต้องเป็น `abstain` ทั้ง KBANK และ ADVANC และ `tick_1600` = 1.0
4. **ถ้าผิดพลาด** (hash ไม่ตรง, คอลัมน์ไม่ครบ, สคริปต์ล้ม):
   ```bash
   mkdir -p ~/floor_incident
   git show HEAD:task_a/results/prediction_log.csv  > ~/floor_incident/task_a_prediction_log.csv
   git show HEAD:task_a2/results/prediction_log.csv > ~/floor_incident/task_a2_prediction_log.csv
   git revert -m 1 <merge-commit>      # merge-commit = commit ที่ merge floor-label
   git push origin main
   ```
   คัดลอก log ที่ได้รับผลกระทบเก็บไว้ก่อน revert แล้วแจ้งทีม ห้ามแก้แถวใน log ด้วยมือ
