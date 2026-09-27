# งาน B: ทำนายราคาปิด (Regression) — เฟส 1

ทำนายราคาปิดของวันถัดไป โดยทำนายผ่าน **return** (ผลตอบแทน) ก่อน
แล้วค่อยแปลงกลับเป็นราคาบาท — โฟลเดอร์นี้แยกอิสระจาก `task_a/` และ
`task_c/` โดยสมบูรณ์ ไม่ import ไฟล์จากที่อื่นเลย

โมเดล: **ANN (MLP, เฉลี่ย 10 seeds) + Random Forest + XGBoost**
ทั้ง 3 ตัวใช้ค่า conservative ที่กำหนดเอง **ยังไม่ได้ปรับจูน** (จูนในเฟส 2)

---

## วิธีรัน

```bash
pip install -r requirements.txt
python main.py --dev          # พัฒนา: train/val เท่านั้น ไม่แตะ test
python tests/test_pipeline.py # test 5 ข้อ
```

รันจากโฟลเดอร์ไหนก็ได้ (`python task_b/main.py --dev` ได้ผลเหมือนกัน)
ผลบันทึกที่ `results/<หุ้น>_taskB_val.csv`

**`python main.py` (ไม่มี `--dev`) จะเปิด test set** ซึ่งถูกล็อกไว้:
ต้องมี `PRE_TEST_LOCK.md` ที่กรอกครบ 10 หัวข้อก่อน ไม่งั้นจะ error ทันที
— **ห้ามสร้างไฟล์นี้ตลอดเฟส 1**

ถ้าเจอ `UnicodeEncodeError` ตอนพิมพ์ภาษาไทย ให้รันด้วย `PYTHONUTF8=1`

---

## ข้อมูล

- อ่านจาก `raw_data/*_10Y_Cleaned.csv` (investing.com) **แหล่งเดียว**
  ไฟล์หาย = error ทันที ไม่มีการไปดึง Yahoo มาแทน
- ทุกครั้งที่โหลดจะพิมพ์ `sha256` ของไฟล์ ไว้ยืนยันว่าใช้ข้อมูลชุดเดิม
- split ตายตัวด้วยวันที่ (`config.SPLIT_BY_DATE`):

| ชุด | ช่วง | แถว |
|---|---|---:|
| train | 2016-09-26 → 2023-09-01 | 1,688 |
| val | 2023-09-04 → 2025-02-25 | 362 |
| test | 2025-02-26 → 2026-08-28 | 362 |

ในโหมด `--dev` แถวของ test ถูกตัดทิ้งตั้งแต่ก่อนสร้าง feature

---

## โครงสร้างไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `config.py` | path, แหล่งข้อมูล, วันที่ split, hyperparameter, หัวข้อของ lock |
| `data_loader.py` | อ่าน `raw_data/` + fingerprint |
| `features.py` | market 24 ตัว (shift(1), ลงท้าย `_prev`) + calendar 5 ตัว (`dow_*`, ไม่ shift) |
| `targets.py` | `y_return` = close(t) / close(t-1) − 1 |
| `splits.py` | แบ่งตามวันที่ ไม่สร้าง test partition ในโหมด `--dev` |
| `models.py` | ANN / RF / XGBoost + `SeedAveragedRegressor` |
| `baselines.py` | Naive (return = 0) — คู่แข่งหลัก, Mean Return |
| `evaluate.py` | 5 metric + ตาราง + เทียบกับ Naive |
| `diagnostics.py` | วิเคราะห์ข้อมูลก่อนเทรน — **train เท่านั้น** |
| `main.py` | ตัวหลัก + lock gate |
| `tests/test_pipeline.py` | alignment, dow, NaN, lock gate, seed averaging |
| `results_reference/` | ผลของเวอร์ชันก่อนหน้า — backup เท่านั้น ห้ามเป็น input |

---

## อ่านผลยังไง

- ตัดสินด้วย **`MAE_return` เทียบกับ `Baseline: Naive (RW)`**
  (target คือ return และ scale-normalized ทุกวันเทียบกันได้)
- `MAE_baht` / `RMSE_baht` ใช้อธิบายขนาด error เป็นบาท
- `R2_return` < 0 = แย่กว่าการทายค่าเฉลี่ยของชุดนั้น
- ไม่มี `R2_price`: ได้ ~0.98 ทุกโมเดลรวมทั้ง naive — วัดข้อมูล ไม่ได้วัดโมเดล
- ไม่มี `DirAcc` ในเฟส 1 (ต้องกำหนดนโยบายวันราคานิ่งก่อน — เฟส 2)

**จะเห็น `ConvergenceWarning` ของ MLP** — ตั้งใจไม่ซ่อนไว้
(`max_iter=500` ยังไม่พอให้ converge) ดูรายละเอียดในสรุปงานเฟส 1
