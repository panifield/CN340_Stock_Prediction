# งาน C: ทำนายขึ้น/ลง (Classification)

ทำนายว่าราคาปิดของวันถัดไปจะ**ขึ้น**หรือ**ลง**(หรือเท่าเดิม)จากวันนี้
— โฟลเดอร์นี้แยกอิสระจาก `task_a/` และ `task_b/` โดยสมบูรณ์
ไม่ import ไฟล์จากที่อื่นเลย แก้อะไรในนี้ไม่กระทบโฟลเดอร์อื่น

โมเดล: **ANN (MLP) + Logistic Regression + XGBoost**
(ใช้ Logistic Regression แทน Random Forest เพื่อให้ครบ 3 แนวทาง:
เชิงเส้น / tree-based / neural network)

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
| `config.py` | ค่าตั้งทั้งหมดของงาน C (รวม `SUSPICIOUS_ACCURACY` เตือน leak) | อยากเปลี่ยนหุ้น / พารามิเตอร์โมเดล / สัดส่วน split / feature windows |
| `data_loader.py` | โหลดข้อมูลราคาหุ้น + ทำความสะอาด | เปลี่ยนแหล่งข้อมูล / ใช้ไฟล์ csv เอง |
| `features.py` | สร้าง feature + shift(1) กัน leak | อยากเพิ่ม/ลด indicator |
| `targets.py` | สร้าง target `y_updown` + `drop_flat_days()` | เปลี่ยนนิยาม target |
| `splits.py` | แบ่ง train/val/test ตามเวลา (ห้าม shuffle) | อยากใช้ walk-forward |
| `baselines.py` | Baseline: Always Up / Majority / Persistence / Random | เพิ่ม baseline ใหม่ |
| `models.py` | นิยามโมเดล ANN / Logistic Regression / XGBoost | เปลี่ยนโมเดล / สลับไปใช้ Keras |
| `evaluate.py` | คำนวณ metric (Accuracy, F1, ROC-AUC ฯลฯ) + ตาราง | เพิ่ม metric |
| `diagnostics.py` | วิเคราะห์ข้อมูลก่อนเทรน (base rate, leak check) | — |
| `main.py` | ตัวหลัก เรียกทุกอย่าง | เปลี่ยนขั้นตอนการทดลอง (เช่น `drop_flat=True`) |

---

## สิ่งที่ต้องดูก่อนเขียนรายงาน

- **วันที่ราคานิ่งเป๊ะ** (`is_flat`) — ถ้าเกิน 5% ของข้อมูล ควรพิจารณา
  ตัดออกด้วย `run_task_c(X, targets, drop_flat=True)` ใน `main.py`
  เพราะวันพวกนี้ไม่ใช่ทั้ง "ขึ้น" และ "ลง"
- **Always-Up baseline** — ตลาดขาขึ้นระยะยาวทำให้ทาย "ขึ้น" ตลอด
  ก็ได้ accuracy ราว 50%+ อยู่แล้ว ต้องเทียบโมเดลกับ baseline นี้ให้ชัด
- ถ้า accuracy สูงเกิน `SUSPICIOUS_ACCURACY` (ตั้งไว้ 0.65 ใน
  `config.py`) โปรแกรมจะเตือนอัตโนมัติว่าน่าจะมี data leakage
