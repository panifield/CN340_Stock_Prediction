# งาน B: ทำนายราคาปิด (Regression)

ทำนายราคาปิดของวันถัดไป โดยทำนายผ่าน **return** (ผลตอบแทน) ก่อน
แล้วค่อยแปลงกลับเป็นราคาบาท — โฟลเดอร์นี้แยกอิสระจาก `task_a/` และ
`task_c/` โดยสมบูรณ์ ไม่ import ไฟล์จากที่อื่นเลย แก้อะไรในนี้ไม่กระทบ
โฟลเดอร์อื่น

โมเดล: **ANN (MLP) + Random Forest + XGBoost** และ `Ensemble (1/3 each)`
ซึ่งเป็นค่าเฉลี่ยของทั้งสามด้วยน้ำหนักตายตัว (ไม่ได้หาน้ำหนักจาก val)
ANN เฉลี่ยผลจาก 10 seed เพื่อไม่ให้ผลขึ้นกับการสุ่มค่าเริ่มต้น

ใช้ **29 features** (ตัดกลุ่มราคาดิบออกตาม ablation H1)
ค่าทั้งหมดที่ถูกล็อกไว้ก่อนเปิด test อยู่ใน **[`PRE_TEST_LOCK.md`](PRE_TEST_LOCK.md)**

---

## วิธีรัน

```bash
pip install -r ../requirements.txt
python main.py
```

ข้อมูลอ่านจากไฟล์ในเครื่อง ไม่ต้องต่อเน็ต

**โหมด dev** — ใช้ตอนกำลังปรับ feature/พารามิเตอร์ซ้ำๆ
เทรน+ประเมินบน train/val เท่านั้น ยังไม่แตะ test เลย:

```bash
python main.py --dev
```

ทั้งสองโหมดบันทึกผลลง `results/` เหมือนกัน ต่างกันที่ชื่อไฟล์
(`*_val_*` กับ `*_test_*`) ซึ่งอ่านจากชุดข้อมูลที่ใช้ประเมินจริง ไม่ได้อ่านจาก
flag จึงไม่มีทางที่ผล val จะถูกตั้งชื่อเป็น test หรือกลับกัน

ไฟล์ที่ได้ต่อการรันหนึ่งครั้ง:

| ไฟล์ | เนื้อหา |
|---|---|
| `{หุ้น}_taskB_{stage}_{n}feat_{เวลา}.csv` | ตาราง metric ของทุกโมเดลและ baseline |
| `{หุ้น}_taskB_regime_{stage}_..._{เวลา}.csv` | MAE แยกตามสภาวะตลาด (vol สูง/ต่ำ, วันขึ้น/ลง/นิ่ง) |
| `{หุ้น}_taskB_preds_{stage}_..._{เวลา}.csv` | คำทำนายรายวันของทุกตัวทำนาย (เอาไปทำกราฟได้) |
| `taskB_{stage}_..._{เวลา}_report.txt` | รายงานเต็มแบบเดียวกับที่ขึ้นบนจอ |

**สคริปต์เสริม** (ทุกตัวใช้แค่ train/val ไม่แตะ test):

```bash
python ablation_h1.py                      # ทดสอบ H1: feature ราคาดิบทำให้เอียงไหม
python ann_sweep.py                        # กวาด hyperparameter ANN
python xgb_sweep.py                        # กวาด hyperparameter XGBoost + ตรวจความไว RF
python feature_selection.py --model XGBoost  # backward elimination + permutation guard
```

---

## แหล่งข้อมูล

**ข้อมูลที่ใช้ในรายงานมาจาก investing.com ไม่ใช่ Yahoo Finance**

| | |
|---|---|
| ไฟล์ต้นทาง | `raw_data/KBANK_10Y_Cleaned.csv`, `raw_data/ADVANC_10Y_Cleaned.csv` |
| ช่วงเวลา | 2016-08-26 ถึง 2026-08-28 (อย่างละ 2,433 แถว) |
| คอลัมน์ต้นทาง | `Date` (MM/DD/YYYY), `Price`, `Open`, `High`, `Low`, `Vol. ('000)`, `Change %` |
| การแปลง | `Price` → `Close` / `Vol. ('000)` × 1000 → `Volume` / ทิ้ง `Change %` (ซ้ำกับที่คำนวณเองได้) |

ตั้งค่าที่ `DATA_SOURCE` ใน `config.py`

```python
DATA_SOURCE = "investing"   # อ่านจาก raw_data/ โดยตรง  <-- ค่าที่ใช้จริง
# DATA_SOURCE = "yahoo"     # ดึงจาก Yahoo Finance ผ่าน yfinance แล้ว cache ไว้
```

### ทำไมต้องระบุ `DATA_SOURCE` ให้ชัด

ไฟล์ใน `data_cache/` **ถูกสร้างจาก `raw_data/` มาแต่แรก** ไม่ใช่ cache ของ
Yahoo จริง ๆ (ตรวจสอบแล้วว่า OHLC ตรงกันทุกแถวโดยมีค่าต่างสูงสุด = 0 และ
`Volume` = `Vol. ('000)` × 1000 พอดีทุกแถว) แต่ชื่อไฟล์ใช้ ticker แบบ Yahoo
(`KBANK_BK_...csv`) จึงทำให้เข้าใจผิดได้ง่ายมาก

โค้ดเดิมอ่านจาก `data_cache/` และ **ถ้าไฟล์นั้นหายไปจะไปดึงข้อมูล Yahoo
ของจริงมาแทนเงียบ ๆ โดยไม่มี error** ซึ่งจะได้ข้อมูลคนละชุด ทำให้ตัวเลข
ในรายงาน reproduce ไม่ได้และไม่มีร่องรอยว่าแหล่งข้อมูลเปลี่ยนไปแล้ว

ตอนนี้เมื่อ `DATA_SOURCE = "investing"` แล้วหาไฟล์ใน `raw_data/` ไม่เจอ
โปรแกรมจะ **`raise` ทันที ไม่ fallback ไป Yahoo**
ถ้าไฟล์หายให้กู้คืนด้วย `git checkout -- raw_data/`

**ถ้าอยากทดสอบว่าโค้ดรันผ่านไหมโดยไม่มีไฟล์ข้อมูล** ให้เปิด `config.py`
แล้วตั้ง `USE_SYNTHETIC_DATA = True`
(แต่ห้ามเอาผลจากข้อมูลจำลองไปใส่รายงาน)

---

## โครงสร้างไฟล์

| ไฟล์ | หน้าที่ | แก้เมื่อไหร่ |
|---|---|---|
| `config.py` | ค่าตั้งทั้งหมดของงาน B (รวม `DATA_SOURCE`) | อยากเปลี่ยนหุ้น / แหล่งข้อมูล / พารามิเตอร์โมเดล / สัดส่วน split / feature windows |
| `data_loader.py` | โหลดข้อมูลราคาหุ้น + ทำความสะอาด | เปลี่ยนแหล่งข้อมูล / ใช้ไฟล์ csv เอง |
| `features.py` | สร้าง feature + shift(1) กัน leak | อยากเพิ่ม/ลด indicator |
| `targets.py` | สร้าง target `y_return` | เปลี่ยนนิยาม target |
| `splits.py` | แบ่ง train/val/test ตามเวลา (ห้าม shuffle) | อยากใช้ walk-forward |
| `baselines.py` | Naive (RW) / Mean Return / Always Up / Rolling Mean 5-10-20 วัน | เพิ่ม baseline ใหม่ |
| `models.py` | นิยามโมเดล ANN / RF / XGBoost + seed averaging + ensemble | เปลี่ยนโมเดล / สลับไปใช้ Keras |
| `evaluate.py` | metric ทั้งหมด + DM test + error analysis แยก regime + ตาราง | เพิ่ม metric |
| `diagnostics.py` | วิเคราะห์ข้อมูลก่อนเทรน (ใช้ train เท่านั้น) | — |
| `main.py` | ตัวหลัก เรียกทุกอย่าง (รวม `TransformedTargetRegressor` scale target) | เปลี่ยนขั้นตอนการทดลอง |
| `PRE_TEST_LOCK.md` | ค่าที่ล็อกไว้ก่อนเปิด test + คำทำนายก่อนเห็นผล | **ห้ามแก้หลังล็อก** |
| `ablation_h1.py` | ทดสอบ H1 (feature ราคาดิบทำให้โมเดลเอียง) | — |
| `ann_sweep.py` / `xgb_sweep.py` | กวาด hyperparameter บน val | — |
| `feature_selection.py` | backward elimination + permutation guard | — |
| `plots.py` | วาดกราฟจากไฟล์ผลใน `results/` (ไม่เทรนโมเดลใหม่) | อยากได้กราฟแบบอื่น |

---

## ทำไมทำนาย return ไม่ทำนายราคาดิบ?

1. **Tree model extrapolate ไม่ได้** — Random Forest / XGBoost ทำนาย
   ด้วยค่าเฉลี่ยของ leaf node ถ้าเทรนช่วงราคา 100-150 แล้ว test ช่วง
   150-200 โมเดลจะทำนายตันอยู่ที่ 150
2. **ราคาดิบเป็น non-stationary** (มี trend) ส่วน return เป็น
   stationary → โมเดลเรียนรู้ได้ถูกต้องกว่า
3. **ถ้าทำนายราคาดิบจะได้ R² ~0.99 ซึ่งหลอกมาก** เพราะโมเดลแค่
   เรียนรู้ว่า "พรุ่งนี้ ≈ วันนี้"

พอทำนาย return เสร็จ ค่อยแปลงกลับเป็นราคาด้วย
`Close_pred(t) = Close(t-1) * (1 + return_pred)`

## สิ่งที่ต้องดูก่อนเขียนรายงาน

- **`DM_p` คือตัวตัดสิน** ไม่ใช่การเทียบตัวเลข MAE ด้วยตาเปล่า
  Diebold-Mariano test บอกว่าส่วนต่างจาก `Baseline: Naive (RW)` ใหญ่เกิน
  ความแปรปรวนของมันเองหรือยัง ถ้า `DM_p > 0.05` = เสมอกันในเชิงสถิติ
- **`R2_OOS`** เทียบกับ Naive โดยตรง (`1 − SSE/SSE_naive`) ส่วน `R2_return`
  เทียบกับค่าเฉลี่ยของชุดที่ประเมินเอง ซึ่ง ณ เวลาทำนายยังไม่มีทางรู้
- **อย่าดู `R2_price` เฉยๆ** มันสูงหลอกๆ อยู่แล้วจากธรรมชาติของราคาหุ้น
- **`StdRatio` เทียบกับ `Rho`** ถ้า StdRatio สูงกว่า Rho มาก แปลว่าโมเดล
  ทำนายแกว่งเกินจริง (ปรับเทียบผิด) ไม่ใช่ทายผิดทิศ
- **`DirAcc`** นับเฉพาะวันที่ราคาขยับจริง เทียบกับ `Baseline: Always Up`
  ไม่ใช่เทียบกับ 50%
- เลือกโมเดลที่ดีที่สุดด้วย **`MAE_return`** ไม่ใช่ `MAE_baht` เพราะ
  `MAE_baht` ถ่วงน้ำหนักวันท้ายชุดมากกว่าตามระดับราคาที่สูงขึ้น
- ถ้าโมเดล ML แพ้ `Baseline: Naive (RW)` หรือแพ้ `Baseline: Rolling Mean`
  แปลว่าโมเดลไม่มีค่าเพิ่ม
