# tools/

## check_raw_update.py — ตรวจ raw_data ชุดใหม่ก่อน commit

ใช้ทุกครั้งที่อัปเดต `raw_data/*.csv` จาก investing.com (แหล่งเดิมเท่านั้น)

```bash
cp -r raw_data raw_data_backup_20260828
# ...วางไฟล์ใหม่ลง raw_data/...
python tools/check_raw_update.py raw_data_backup_20260828 raw_data   # ต้องผ่าน
python main.py --dev                                                  # ต้องได้ 1688/362 เท่าเดิม
diff <ผล val CSV เก่า> <ผล val CSV ใหม่>                              # ต้องว่าง
git add raw_data && git commit -m "data: update raw_data ถึง YYYY-MM-DD"
```

ตรวจอะไร:

1. ช่วงวันที่ที่ทับกันระหว่างไฟล์เก่ากับใหม่ ต้องมี Open/High/Low/Close/Volume
   **เท่ากันเป๊ะ** — ต่างแม้แถวเดียวจะพิมพ์แถวนั้นออกมาแล้ว exit 1
2. รายงานจำนวนแถว, วันสุดท้ายเก่า → ใหม่, วันที่เพิ่มทั้งหมด
3. เตือนวันที่ซ้ำ, วันที่ไม่เรียง, วันที่หายไปจากไฟล์ใหม่

ทำไมต้องตรวจช่วงทับ: `predict_live.py` เทรนจากข้อมูล**ทั้งหมด** ถ้า investing.com
แก้ราคาย้อนหลัง โมเดล live จะเปลี่ยนโดยไม่รู้ตัว ส่วน `main.py --dev`
ตรวจได้แค่ถึง validation boundary จึงไม่พอ
