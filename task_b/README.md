# งาน B: ทำนายราคาปิด (Regression)

ทำนายราคาปิดของวันถัดไป โดยทำนายผ่าน **return** (ผลตอบแทน) ก่อน
แล้วค่อยแปลงกลับเป็นราคาบาท — โฟลเดอร์นี้แยกอิสระจาก `task_a/` และ
`task_c/` โดยสมบูรณ์ ไม่ import ไฟล์จากที่อื่นเลย แก้อะไรในนี้ไม่กระทบ
โฟลเดอร์อื่น

โมเดล: **ANN (MLP) + Random Forest + XGBoost**

---

## วิธีรัน

```bash
pip install -r ../requirements.txt
python main.py
```

ครั้งแรกจะโหลดข้อมูลจาก Yahoo Finance แล้วเก็บไว้ใน `data_cache/`
(มีไฟล์ cache เตรียมไว้ให้แล้ว ไม่ต้องต่อเน็ตก็รันได้เลย)
ผลลัพธ์จะถูกบันทึกเป็น csv ในโฟลเดอร์ `results/`

**โหมด dev** — ใช้ตอนกำลังปรับ feature/พารามิเตอร์ซ้ำๆ
เทรน+ประเมินบน train/val เท่านั้น ยังไม่แตะ test เลย ไม่บันทึกผล:

```bash
python main.py --dev
```

**ถ้ารันไม่ได้เพราะเน็ต** ให้เปิด `config.py` แล้วตั้ง
`USE_SYNTHETIC_DATA = True` เพื่อทดสอบว่าโค้ดทำงานได้
(แต่ห้ามเอาผลจากข้อมูลจำลองไปใส่รายงาน)

---

## โครงสร้างไฟล์

| ไฟล์ | หน้าที่ | แก้เมื่อไหร่ |
|---|---|---|
| `config.py` | ค่าตั้งทั้งหมดของงาน B | อยากเปลี่ยนหุ้น / พารามิเตอร์โมเดล / สัดส่วน split / feature windows |
| `data_loader.py` | โหลดข้อมูลราคาหุ้น + ทำความสะอาด | เปลี่ยนแหล่งข้อมูล / ใช้ไฟล์ csv เอง |
| `features.py` | สร้าง feature + shift(1) กัน leak | อยากเพิ่ม/ลด indicator |
| `targets.py` | สร้าง target `y_return` | เปลี่ยนนิยาม target |
| `splits.py` | แบ่ง train/val/test ตามเวลา (ห้าม shuffle) | อยากใช้ walk-forward |
| `baselines.py` | Baseline: Naive (RW) / Mean Return | เพิ่ม baseline ใหม่ |
| `models.py` | นิยามโมเดล ANN / Random Forest / XGBoost (regression) | เปลี่ยนโมเดล / สลับไปใช้ Keras |
| `evaluate.py` | คำนวณ metric (MAE, RMSE, R², DirAcc, MAE_baht ฯลฯ) + ตาราง | เพิ่ม metric |
| `diagnostics.py` | วิเคราะห์ข้อมูลก่อนเทรน (ขนาด return, leak check) | — |
| `main.py` | ตัวหลัก เรียกทุกอย่าง (รวม `TransformedTargetRegressor` scale target) | เปลี่ยนขั้นตอนการทดลอง |

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

- **ดู `MAE_baht` เทียบกับ `Baseline: Naive (RW)` เป็นหลัก** อย่าไปดู
  `R2_price` เฉยๆ เพราะมันจะสูงหลอกๆ อยู่แล้วจากธรรมชาติของราคาหุ้น
- **`DirAcc`** (ทายทิศทางถูกกี่ %) มีความหมายในทางปฏิบัติมากกว่า R²
- ถ้าโมเดล ML แพ้ `Baseline: Naive (RW)` แปลว่าโมเดลไม่มีค่าเพิ่ม
