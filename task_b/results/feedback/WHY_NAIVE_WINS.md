# ทำไม Naive ชนะ — คำอธิบายสำหรับสไลด์ (feedback ข้อ 2)

> **ตัวเลขทุกตัวในไฟล์นี้มาจากไฟล์ผลที่รันจริงรอบนี้** ไม่มีเลขจากความจำ:
> - `results/<TICKER>_taskB_val_stats.csv` + `_val_preds.csv` + `_val_shape.csv` (`main.py --dev`)
> - `results/feedback/walkforward_stats.csv` + `walkforward_summary.csv` (`stats_tests.py --walkforward`)
> - `results/split_profile.csv` (`split_profile.py`)
>
> เครื่องมือ: `stats_tests.py` (โมดูลใหม่) · Diebold–Mariano (R8) + Harvey–Leybourne–Newbold
> small-sample correction (R9) — ดู `../../REFERENCES.md` หมวด 3.3

---

## สรุป 1 ย่อหน้า (ใช้พูดได้เลย)

MAE ลงโทษทุกบาทที่ทายผิดเท่ากัน และราคาปิด **ไม่ขยับเลยประมาณ 15% ของวัน** ในวันเหล่านั้น
Naive ที่ทาย "return = 0" มี error **= 0 เป๊ะ** ซึ่งเป็นเพดานล่างที่โมเดลเอาชนะไม่ได้ ได้แค่เสมอ
ส่วนโมเดล ML ที่ทายค่าไม่เป็นศูนย์ เสีย error ทุกวันนิ่ง · ในวันที่ราคาขยับจริง
**โมเดลชนะ Naive ได้จริง** (error ต่ำกว่าทุกโมเดล แทบทุก fold) แต่กำไรฝั่งนั้น
**ไม่พอกลบ** ที่เสียไปในวันนิ่ง -> MAE รวมเลยแพ้ · และเมื่อทดสอบนัยสำคัญด้วย DM test
ส่วนต่างนี้ **แยกจาก 0 ไม่ได้เลยทั้ง 18 evaluation** ทั้งฝั่งที่โมเดลดีกว่าและฝั่งที่ Naive ดีกว่า
-> ข้อสรุปที่ถูกต้องไม่ใช่ "Naive เหนือกว่า ML" แต่คือ **"สัญญาณที่หาได้จากราคาย้อนหลังอ่อนมาก
จนแยกจาก random walk ไม่ออก"** ซึ่งตรงกับสิ่งที่ efficient market hypothesis (R1) ทำนาย

---

## 1. สัดส่วนวันราคานิ่ง — ต้นตอของเรื่องทั้งหมด

| ชุด | KBANK | ADVANC |
|---|---|---|
| train (1,708 แถว) | 11.4% | 13.9% |
| **val (362 แถว)** | **14.1%** | **18.5%** |
| test (362 แถว · ยังไม่เปิด) | 15.5% | 13.0% |

(`results/split_profile.csv` คอลัมน์ `frac_zero_move`)

บน walk-forward 6 evaluation เฉลี่ย **15.04%** (`walkforward_summary.csv` → `mean_frac_zero_move`)
ต่อ evaluation อยู่ระหว่าง **11.65%** (KBANK fold 2) ถึง **22.56%** (ADVANC fold 2)

**ทำไมนิ่งได้เยอะขนาดนี้** — tick size ของหุ้นสองตัวนี้คือ 0.50–1.00 บาทบนราคา 119–404 บาท
ราคาที่ขยับน้อยกว่า 1 tick คือขยับไม่ได้จริง · วันที่ `|return| < 1 tick` มีถึง
**26–33% ของวัน** ในชุด val/test (`split_profile.csv` → `frac_move_lt_1_tick`)

---

## 2. การแยก MAE: Naive ได้เปรียบจากวันนิ่งเท่าไหร่

`stats_tests.mae_decomposition()` แยกส่วนต่าง MAE เป็นสองพจน์ที่รวมกันได้ค่าเดิมเป๊ะ:

```
mae_model − mae_naive  =  contrib_zero_days  +  contrib_moved_days
                          (วันที่ y = 0)       (วันที่ y ≠ 0)
```

**บน fixed val** (`results/<TICKER>_taskB_val_stats.csv`)

| หุ้น | โมเดล | relMAE | เสียเปรียบจากวันนิ่ง | ได้เปรียบจากวันขยับ | รวม |
|---|---|---|---|---|---|
| KBANK | Random Forest | 1.0019 | +0.000135 | −0.000119 | +0.000015 |
| KBANK | XGBoost | 1.0033 | +0.000079 | −0.000052 | +0.000027 |
| KBANK | ANN (MLP) | 1.0106 | +0.000141 | −0.000052 | +0.000088 |
| ADVANC | XGBoost | **0.9983** | +0.000125 | −0.000141 | **−0.000016** |
| ADVANC | ANN (MLP) | **0.9993** | +0.000147 | −0.000154 | **−0.000007** |
| ADVANC | Random Forest | 1.0017 | +0.000159 | −0.000144 | +0.000016 |

**บน walk-forward 18 evaluation** (`walkforward_summary.csv`)

| โมเดล | mean relMAE | เสียเปรียบจากวันนิ่ง (เฉลี่ย) | ได้เปรียบจากวันขยับ (เฉลี่ย) |
|---|---|---|---|
| Random Forest | 1.0025 | +0.000140 | −0.000120 |
| XGBoost | 1.0029 | +0.000097 | −0.000072 |
| ANN (MLP) | 1.0061 | +0.000154 | −0.000100 |

**อ่านอย่างไร**
- `contrib_zero_days` **บวกทุกช่อง ไม่มีข้อยกเว้น** — เพราะ Naive มี error = 0 บนวันนิ่งเสมอ
  (`mae_naive_zero_days = 0.0` ทุกแถวในทุกไฟล์) โมเดลจึงเสียเปรียบตรงนี้แบบหนีไม่ได้
- `contrib_moved_days` **ติดลบ 16 จาก 18 evaluation** — แปลว่าในวันที่ราคาขยับจริง
  โมเดล ML มี MAE **ต่ำกว่า** Naive · ตัวอย่าง ADVANC fold 3: XGBoost 0.011952
  vs Naive 0.012133 (`mae_model_moved_days` / `mae_naive_moved_days`)
- สองพจน์เกือบหักกลบกันพอดี — ที่เหลือคือส่วนต่างระดับ 1e-5 ซึ่งเล็กกว่า noise (ดูข้อ 4)

> **ประโยคสำหรับสไลด์:** โมเดลไม่ได้ "ทำนายไม่ได้" มันแพ้เพราะ metric
> ให้รางวัลกับการทาย 0 ในวันที่ตลาดไม่ขยับ ซึ่งเป็น 15% ของวันทั้งหมด

---

## 3. StdRatio ต่ำ = โมเดล shrink เข้าหา 0 เพราะสัญญาณอ่อน

`results/<TICKER>_taskB_val_shape.csv` คอลัมน์ `StdRatio` = `std(pred) / std(true)`

| โมเดล | KBANK | ADVANC |
|---|---|---|
| ANN (MLP) | 0.0994 | 0.0780 |
| Random Forest | 0.0960 | 0.1044 |
| XGBoost | 0.0634 | 0.0934 |

โมเดลแกว่งกว้างเพียง **6–10% ของความแกว่งจริง** · นี่ไม่ใช่บั๊ก แต่เป็นสิ่งที่ตัวลด loss
แบบ squared/absolute error **ควรทำ** เมื่อสัญญาณอ่อน: ถ้า R² ที่ทำได้จริงมีแค่ ~1%
คำตอบที่ลด error ได้ดีที่สุดคือ "ทายเกือบ 0 ทุกวัน" ซึ่งก็คือเกือบเป็น Naive นั่นเอง

ผลลัพธ์: `mean_pred_move_ticks` = **0.14–0.24 tick** (`results/<TICKER>_taskB_val_costs.csv`)
แปลว่าโมเดลทำนายการขยับที่ **เล็กกว่าขั้นราคาที่ตลาดขยับได้จริง** — ต่อให้ทิศถูก
ก็สั่งซื้อขายตามไม่ได้

---

## 4. Information coefficient: มีสัญญาณจริง แต่เล็กมาก

| | ANN (MLP) | Random Forest | XGBoost |
|---|---|---|---|
| mean IC (Pearson) บน 6 evals | **+0.0933** | **+0.1122** | **+0.0772** |
| mean IC (Spearman) | +0.0996 | +0.1027 | +0.0894 |
| mean hit rate เฉพาะวันที่ราคาขยับ | 55.3% | 54.9% | 54.0% |

(`walkforward_summary.csv`) · บน fixed val: IC Pearson อยู่ระหว่าง +0.047 (ANN KBANK)
ถึง +0.190 (XGBoost ADVANC) · hit rate เฉพาะวันขยับ 50.2–60.0%

**IC เป็นบวก 16 จาก 18 evaluation** — มีสัญญาณอยู่จริง ไม่ใช่ศูนย์
แต่ IC ~0.1 แปลเป็น R² ได้แค่ ~1% ซึ่งเล็กเกินกว่าจะพลิก MAE ให้ชนะ Naive ได้
สองช่องที่ IC ติดลบคือ XGBoost KBANK fold 2 (−0.1190) และ ANN KBANK fold 2 (−0.0220)
→ ช่วง 2022-12-20 → 2024-01-19 ของ KBANK เป็น fold เดียวที่โมเดลทำนายผิดทิศเป็นระบบ
(Random Forest ใน fold เดียวกันยังเป็นบวกเฉียดฉิว +0.0414)

> **หมายเหตุนโยบายวันนิ่ง:** hit rate นับ **เฉพาะวันที่ `y ≠ 0`** เพราะวันที่ราคาไม่ขยับ
> ไม่มี "ทิศที่ถูก" ให้ทาย ถ้านับรวม ตัวเลขจะกลายเป็นผลของนโยบายการให้คะแนนวันนิ่ง
> ไม่ใช่ความสามารถของโมเดล (เขียนไว้ใน `stats_tests.direction_metrics` docstring)

---

## 5. DM test: ต่างจาก Naive อย่างมีนัยสำคัญไหม → **ไม่**

Diebold–Mariano บน absolute loss (primary ของงาน B) + HLN small-sample correction
เทียบกับ t-distribution df = n − 1 · α = 0.05 one-sided (H1: โมเดล MAE ต่ำกว่า Naive)

**walk-forward 6 evaluation ต่อโมเดล** (`walkforward_summary.csv`)

| โมเดล | evals ที่ MAE ต่ำกว่า Naive | ชนะอย่างมีนัยสำคัญ (α = 0.05) | หลัง Bonferroni (α = 0.00833) | **แพ้** อย่างมีนัยสำคัญ | p one-sided ที่เล็กสุด |
|---|---|---|---|---|---|
| ANN (MLP) | 1 / 6 | **0 / 6** | **0 / 6** | 0 / 6 | 0.433 |
| Random Forest | 2 / 6 | **0 / 6** | **0 / 6** | 0 / 6 | 0.410 |
| XGBoost | 2 / 6 | **0 / 6** | **0 / 6** | 0 / 6 | 0.233 |

Bonferroni: α / 6 evaluation = 0.05 / 6 = **0.008333**

**fixed val** (`results/<TICKER>_taskB_val_stats.csv` คอลัมน์ `dm_abs_*`)

| หุ้น | โมเดล | DM (HLN) | p one-sided | p two-sided | สรุป |
|---|---|---|---|---|---|
| ADVANC | XGBoost | −0.240 | 0.405 | 0.811 | MAE ต่ำกว่า Naive แต่ **ไม่มีนัยสำคัญ** |
| ADVANC | ANN (MLP) | −0.123 | 0.451 | 0.902 | MAE ต่ำกว่า Naive แต่ **ไม่มีนัยสำคัญ** |
| ADVANC | Random Forest | +0.210 | 0.583 | 0.834 | แพ้ Naive ไม่มีนัยสำคัญ |
| KBANK | Random Forest | +0.280 | 0.610 | 0.780 | แพ้ Naive ไม่มีนัยสำคัญ |
| KBANK | XGBoost | +0.730 | 0.767 | 0.466 | แพ้ Naive ไม่มีนัยสำคัญ |
| KBANK | ANN (MLP) | +1.511 | 0.934 | 0.132 | แพ้ Naive ไม่มีนัยสำคัญ |

**ข้อสรุปของ DM test:** จาก 18 evaluation walk-forward + 6 แถวบน fixed val
**ไม่มีสักช่องเดียว** ที่ส่วนต่าง MAE ระหว่างโมเดลกับ Naive แยกจาก 0 ได้ที่ α = 0.05
→ `n_sig_better_uncorrected = 0` และ `n_sig_worse_uncorrected = 0` ทุกโมเดล

ค่า `Baseline: Mean Return` เป็นตัวตรวจว่าการทดสอบ **ไวพอ** จริง: มันแพ้ Naive
อย่างมีนัยสำคัญทั้งสองหุ้น (KBANK DM +3.52, p two-sided = 0.00048 ·
ADVANC DM +2.65, p two-sided = 0.0084) → เครื่องมือจับความต่างได้เมื่อมีความต่างจริง

---

## 6. เชื่อมกับ efficient market hypothesis

Naive baseline ของเรา (`baselines.baseline_naive_return` → ทำนาย return = 0)
**คือ random walk แบบตรงตัว** · EMH แบบ weak form (Fama 1970 — R1 ใน
`../../REFERENCES.md`) บอกว่าราคาในอดีตเพียงอย่างเดียวไม่ควรทำนายราคาในอนาคตได้
พอจะทำกำไรหลังหักต้นทุน

สิ่งที่เราวัดได้สอดคล้องกับนั้นพอดี และยัง **ละเอียดกว่าคำว่า "ทำนายไม่ได้"**:
1. มีสัญญาณอยู่จริง (IC เป็นบวก 16/18 evaluation, hit rate เฉลี่ย 54–55% บนวันที่ราคาขยับ)
2. แต่เล็กเกินกว่าจะมีนัยสำคัญทางสถิติ (DM test 0/18)
3. และเล็กกว่าต้นทุนการซื้อขาย: `mean_pred_move_ticks` 0.14–0.24 tick ขณะที่ต้นทุนไป-กลับ
   = **33.598 bps** และโมเดลแนะนำให้ซื้อเพียง **0–1.9% ของวัน** ที่อัตราค่าคอมมิชชันปัจจุบัน
   (`results/<TICKER>_taskB_val_costs.csv` → `pct_pred_above_cost`) ·
   ถ้าลดค่าคอมมิชชันเป็น 0.10% (ต่ำสุดที่ตรวจเจอ) ตัวเลขนี้ขึ้นเป็นเพียง 1.4–3.9%
   (`results/cost_sensitivity.csv`) → **ข้อสรุปไม่เปลี่ยนเมื่อค่าธรรมเนียมเปลี่ยน**

---

## 7. ข้อจำกัด (ต้องพูดด้วย)

1. **val ถูกใช้จูนมาแล้ว** — `TUNING_PLAN.md` fold 2–3 ของ `tune.py` ทับช่วง validation
   ของ `main.py` · ตัวเลขบน fixed val (362 แถว) จึงเป็น in-sample ของการเลือก
   hyperparameter **ไม่ใช่การประเมินอิสระ** และมองในแง่ดีเกินจริง
2. **walk-forward 6 evaluation ก็ไม่อิสระ** — เป็น fold ชุดเดียวกับที่ `tune.py` ใช้เลือก
   config Phase 1C มา ตัวเลขในข้อ 5 จึงเป็น "ผลของผู้ชนะบนสนามที่มันถูกเลือกมา"
3. **n เล็ก** — แต่ละ evaluation มี 266 แถว · HLN correction ช่วยให้ conservative ขึ้น
   (ทำให้ p ใหญ่กว่าการใช้ normal distribution) แต่ไม่ได้แก้ว่าอำนาจการทดสอบต่ำ:
   ส่วนต่างระดับ 1e-5 บน 266 วันตรวจจับไม่ได้ ไม่ว่าจะมีจริงหรือไม่
4. **DM test สมมติ loss differential เป็น stationary** · return รายวันมี volatility clustering
   ซึ่งกระทบ variance ที่ประมาณได้ (h = 1 จึงไม่มีพจน์ autocovariance มาช่วย)
5. **historical test ยังไม่ถูกเปิด** — ไม่มี `PRE_TEST_LOCK.md` ในโปรเจกต์ ตัวเลขทั้งหมด
   ในไฟล์นี้เป็น train/val เท่านั้น การเปิด test เป็นการตัดสินใจของผู้ใช้
6. ตัวเลขทั้งหมดในไฟล์นี้ **ไม่ถูกใช้เลือกอะไรเลย** — การเลือกโมเดลใช้ `MAE_return` เท่านั้น
