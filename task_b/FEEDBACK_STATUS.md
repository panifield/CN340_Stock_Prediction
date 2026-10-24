# FEEDBACK_STATUS — สถานะ feedback ของอาจารย์ (14 ก.ย. 2026) สำหรับ Task B

> **ไฟล์นี้ทำอะไร** — ตารางเดียวที่บอกว่า feedback ทั้ง 12 ข้อ ข้อไหนเกี่ยวกับ Task B
> ข้อไหนทำแล้ว ทำที่ไหน และผลเป็นอย่างไร · อัปเดตล่าสุด **2026-10-24**
>
> Task B = ทำนายราคาปิดแบบ regression ผ่าน return (`main.py --dev`)
> งานคู่/คี่ (classification) เป็นของ Task A / A2 · งานทิศทาง 16:00 เป็นของ Task C
>
> **ตัวเลขทุกตัวในคอลัมน์ "ผลสรุป" มาจากไฟล์ที่อยู่ในคอลัมน์ "ไฟล์หลักฐาน"**

---

## ตารางสถานะ 12 ข้อ

| # | feedback (สรุป) | เกี่ยวกับ Task B? | สถานะ | ไฟล์หลักฐาน | ผลสรุป |
|---|---|---|---|---|---|
| **1** | ใช้ปัญหาคู่/คี่ (parity) | **ไม่เกี่ยว** | งานของ **Task A / A2** | `task_a/`, `task_a2/`, `docs/floor_label_followup.md` | Task B เป็น regression ทำนายค่าต่อเนื่อง ไม่มี label คู่/คี่ให้ตัด |
| **2** | ทำไม Naive ถึงชนะ | **เกี่ยว** | ✅ **ทำรอบนี้** | `stats_tests.py` · `results/<หุ้น>_taskB_val_stats.csv` · `results/<หุ้น>_taskB_val_preds.csv` · `results/feedback/walkforward_stats.csv` + `walkforward_summary.csv` · **`results/feedback/WHY_NAIVE_WINS.md`** · `tests/test_stats.py` | ราคาปิดไม่ขยับเลย **15.0%** ของวัน (เฉลี่ย 6 evaluation) · วันเหล่านั้น Naive มี error = 0 เป๊ะ · โมเดล ML **ชนะ Naive จริงในวันที่ราคาขยับ** (`contrib_moved_days` ติดลบ 16/18 evaluation) แต่ไม่พอกลบที่เสียในวันนิ่ง · **DM test: 0 จาก 18 evaluation ที่ต่างจาก Naive อย่างมีนัยสำคัญ** (ทั้งสองทิศ) |
| **3** | ใช้ HyperOpt เพื่อหา Feature | **เกี่ยว** | ✅ **ทำรอบนี้** | `features_experimental.py` · `optuna_features.py` · `results/feedback/OPTUNA_PLAN.md` (+SHA-256) · `optuna_trials.csv` / `optuna_summary.csv` / `optuna_group_frequency.csv` / `optuna_fold3.csv` / `optuna_ablation.csv` / `optuna_null.csv` · 3 รูป · **`results/feedback/OPTUNA_RESULTS.md`** · `tests/test_features_experimental.py` | Optuna TPE **160 trials** (RF 60 / XGB 60 / ANN 40) ค้น 8 feature group + hyperparameter บน fold 1–2 · ผู้ชนะดีกว่า Phase 1C **0.38–0.40%** บนสนามที่ค้น แต่บน **fold 3 แย่กว่า Phase 1C ทั้งสามโมเดล** และ **ไม่มีใครชนะ Naive อย่างมีนัยสำคัญ (0/6)** → เป็น selection bias ไม่ใช่สัญญาณ · กลุ่มใหม่ `cross_stock` ไม่ช่วย · **ไม่ได้นำผู้ชนะไปใส่ config** |
| **4** | ต้องมี Reference | **เกี่ยว** | ✅ **ทำรอบนี้** | **`REFERENCES.md`** · คอมเมนต์ใน `config.py` (`SET_COSTS`, `SET_TICK_BANDS`) · `trading_costs.cost_sensitivity()` · `results/cost_sensitivity.csv` | 17 reference วิชาการ (R1–R17) + 6 แหล่งค่าธรรมเนียม/tick (S1–S3, B1–B3) แบ่ง 4 หมวด · **DOI ตรวจผ่าน Crossref 9 ชิ้น** · ลิงก์ที่เหลือตรวจว่าเข้าถึงได้จริง (HTTP 200) ทุกอัน ยกเว้น investing.com ที่คืน 403 ให้ request อัตโนมัติ · สิ่งที่ยืนยันไม่ได้ติดป้าย `[ยังไม่ได้ตรวจ]` ไว้ตรง ๆ และสรุปรวมไว้ใน `REFERENCES.md` หมวด 5 (**5 หัวข้อ**) · **tick size ตรงกับ SET ทั้ง 8 ช่วง** · ค่าธรรมเนียมตรงกับตารางโบรกเกอร์จริง 3 ราย · **ไม่ได้แก้ตัวเลขใน config** |
| **5** | test set ควร balance / ทำไมให้เท่า ๆ กัน | **เกี่ยว** | ✅ **ทำรอบนี้** | `split_profile.py` · `results/split_profile.csv` · `results/split_profile_tests.csv` · **`results/split_profile.md`** | val กับ test เท่ากัน (362 แถว) เพราะ **ตั้งใจเลือกวันที่ให้เทียบกับเวอร์ชันก่อนได้** (คอมเมนต์ `SPLIT_BY_DATE`) ไม่ใช่เพราะต้อง balance · regression ไม่มี class ให้ balance → เทียบ volatility / zero-move / ทิศทางแทน · KBANK train ผันผวนกว่า val/test ชัดเจน (KS p = 0.00063 / 0.00021) ส่วน ADVANC ไม่ต่าง (p = 0.093 / 0.18) · **`PRE_TEST_LOCK.md` ยังไม่มี → historical test ยังไม่ถูกเปิด เป็นการตัดสินใจของผู้ใช้** |
| **6** | ทำนายวันถัดไปจริง (live) | **เกี่ยว** | ✅ ทำแล้วก่อนรอบนี้ | `predict_live.py` · `predict_1600.py` · `predict_close.py` · `daily_next_day.py` · `daily_1600.py` · `record_outcomes.py` · `report_live.py` · `PHASE1D_PLAN.md` / `PHASE1D_PLAN_R2.md` · `results/prediction_log.csv` · `results/outcomes.csv` · `results/live_report/` | มี pipeline ทำนายวันข้างหน้าจริงที่รันอัตโนมัติ + บันทึกผลจริงย้อนกลับมาเทียบ · Phase 1D (16:00) pre-registered ไว้สองรอบ · **รอบนี้ไม่แตะไฟล์เหล่านี้เลย** |
| **7** | pretrain ด้วยข้อมูลเก่า แล้ว fine-tune ด้วยข้อมูลใหม่ | **เกี่ยว** | ✅ **ทำรอบนี้** | `finetune_experiment.py` · `results/feedback/FINETUNE_PLAN.md` (+SHA-256) · `finetune_results.csv` (222 แถว) · `finetune_summary.csv` (37 config) · `finetune.png` · **`results/feedback/FINETUNE_RESULTS.md`** | เดิมมีแค่ recency weighting (`weighting.py`) ซึ่งเป็นเหตุผลเชิงทฤษฎีที่ยังไม่เคยทดลอง · รอบนี้ทดลองจริง 37 config × 6 evaluation: **ไม่มี config ใดผ่านกฎตัดสินที่ประกาศไว้ล่วงหน้า** · ANN มี 3 config ที่ MAE ต่ำกว่า `pooled` แต่ **ตก guard StdRatio** (ยุบเข้าหาค่าคงที่) · `recent_only` แย่กว่า `pooled` ทุกตัว (+1.3% ถึง +5.9%) · **7/7 คำทำนายในแผนตรง** |
| **8** | C-Curve | **เกี่ยว** | ✅ ทำแล้วก่อนรอบนี้ | `curves.py` · `results/curves/` (`python main.py --dev --curves`) · REC curve → `REFERENCES.md` R10 | **ตีความ C-Curve = REC curve** (Regression Error Characteristic, Bi & Bennett 2003) = สัดส่วนวันที่ `\|error_return\| ≤ tolerance` ของทุกโมเดล + baseline · **นี่เป็นการตีความ** ไม่ได้ยืนยันกับอาจารย์ว่าหมายถึง curve ตัวนี้จริง · มี ANN loss curve และ ANN alpha curve ด้วย |
| **9** | Threshold ของ Binary | **ไม่เกี่ยว** | งานของ **Task A / C** | `task_a/results/five_analyses/2_threshold_choice.csv` + `2_threshold_sweep.csv` · `task_c/` | Task B เป็น regression **ไม่มี probability ให้ตัด threshold** · **หมายเหตุสำคัญ:** `trading_costs.signal_economics(threshold=cost_rt)` ใน Task B ไม่ใช่ข้อ 9 — มันเป็น **เกณฑ์ต้นทุน** ของข้อ 10 ("ทำนายกำไรเกินค่าธรรมเนียมไหม") ไม่ใช่ threshold ของ classifier |
| **10** | Breakeven / commission + ตาราง diagnostic | **เกี่ยว** | ✅ ทำแล้ว + **เซฟไฟล์รอบนี้** | `trading_costs.py` · `main.py` (`save_results`) · `results/<หุ้น>_taskB_val_shape.csv` · `results/<หุ้น>_taskB_val_costs.csv` · `results/cost_sensitivity.csv` · `tests/test_pipeline.py` (2 tests ใหม่) | ตารางที่ 2 (รูปร่างการทำนาย) และตารางที่ 3 (ต้นทุน) เดิม **print อย่างเดียว อ้างอิงในรายงานไม่ได้** → รอบนี้เซฟลง CSV แล้ว (ตารางหลักไม่เปลี่ยนแม้แต่ byte เดียว) · ต้นทุนไป-กลับ **33.598 bps** · โมเดลทำนายการขยับเฉลี่ย **0.14–0.24 tick** (เล็กกว่าขั้นราคาที่ตลาดขยับได้) · แนะนำซื้อเพียง **0–1.9% ของวัน** · ลดค่าคอมมิชชันเป็น 0.10% ก็ยังแค่ 1.4–3.9% → **ข้อสรุปไม่เปลี่ยนตามค่าธรรมเนียม** |
| **11** | เหตุผลเลือกโมเดล / จูน hyperparameter | **เกี่ยว** | ✅ จูนแล้วก่อนรอบนี้ + **เขียนเหตุผลรอบนี้** | **`MODEL_RATIONALE.md`** (ใหม่) · `TUNING_PLAN.md` · `tune.py` · `results/tuning_summary.md` · `results/tuning_scores.csv` · `results/tuning_stage1_halflife.csv` · `results/tuning_stage2_params.csv` | การจูนทำไปแล้ว 2026-09-27 ตามแผนที่ commit ก่อน (17 configs ต่อโมเดลเท่ากัน) แต่ **ยังไม่มีเอกสารอธิบายว่าทำไมเลือกโมเดลชุดนี้** → เขียนใหม่รอบนี้: ทำไมทำนาย return · แต่ละตระกูลแก้ปัญหาอะไร + จุดอ่อน · ทำไมไม่มี ARIMA/GARCH/linear (**ตอบตรง ๆ ว่าไม่ได้ทำ** เป็นข้อจำกัด) · พารามิเตอร์แต่ละตัวมาจากไหน · ผลจริงเทียบ Naive |
| **12** | ROC ของ Binary | **ไม่เกี่ยว** | งานของ **Task A / C** | `task_a/results/five_analyses/1_roc_auc.csv` · `task_a/results/five_analyses/figures/` | Task B เป็น regression **ไม่มี probability จึงไม่มี ROC/AUC** · สิ่งที่เทียบเคียงได้ในงาน regression คือ **REC curve** (ข้อ 8) ซึ่งทำแล้ว |

---

## สรุปรวม

| | จำนวน |
|---|---|
| เกี่ยวกับ Task B | **9 ข้อ** (2, 3, 4, 5, 6, 7, 8, 10, 11) |
| ไม่เกี่ยว (Task A / A2 / C) | **3 ข้อ** (1, 9, 12) |
| ทำรอบนี้ | **6 ข้อ** (2, 3, 4, 5, 7, 11) + เซฟไฟล์เพิ่มของข้อ 10 |
| ทำแล้วก่อนรอบนี้ | **3 ข้อ** (6, 8, และการจูนของข้อ 11) |
| **ยังไม่ได้ทำ** | **0 ข้อของ Task B** |

---

## สิ่งที่ยังค้างและเป็นการตัดสินใจของผู้ใช้

1. **historical test ยังไม่ถูกเปิด** — ไม่มี `PRE_TEST_LOCK.md` ในโปรเจกต์ ·
   ทุกตัวเลขในรอบนี้เป็น train/val เท่านั้น (ตัดที่ 2025-02-25 ก่อนสร้าง feature) ·
   `split_profile.py` อ่าน **ค่า return** ของช่วง test เพื่ออธิบายข้อมูลอย่างเดียว
   **ไม่มีโมเดลใดถูกประเมินบน test** · การเปิด test เป็นการตัดสินใจของผู้ใช้
   และทำได้ครั้งเดียว
2. **ผู้ชนะ Optuna / ผลการ fine-tune ไม่ได้ถูกนำไปใช้** — `config.py` / `features.py` /
   สคริปต์ live ไม่ถูกแก้เลย · ผลทั้งสองเป็นรายงาน และตัวเลขในรายงานบอกว่า
   **ยังไม่ควรนำไปใช้**
3. **การตีความ C-Curve (ข้อ 8)** ว่าหมายถึง REC curve — ควรยืนยันกับอาจารย์
4. **linear baseline (Ridge / Lasso) ยังไม่มี** — ช่องว่างที่ใหญ่ที่สุดที่เหลืออยู่
   (ดู `MODEL_RATIONALE.md` ข้อ 3.4 และ `results/feedback/OPTUNA_RESULTS.md` ข้อ 7)
