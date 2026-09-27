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

### option `--cross-check raw_data_intraday/` (§6.7)

```bash
python tools/check_raw_update.py raw_data_backup_20260828 raw_data --cross-check raw_data_intraday
```

เทียบราคาปิดของ **วันที่เพิ่มใหม่** กับ `close_bar16` ของไฟล์รายชั่วโมง (Yahoo)
วันไหนต่างเกิน 1 บาท จะพิมพ์ `?? เตือน` ให้ตรวจด้วยตา — **เป็นคำเตือนเท่านั้น
ไม่กระทบ exit code** เพราะสองแหล่งต่างกันเฉลี่ย 0.33–0.69 บาทอยู่แล้ว

ห้ามใช้ไฟล์รายชั่วโมงเติมวันที่ขาดใน `raw_data/` — ใช้ตรวจทานอย่างเดียว

## intraday_probe.py — สำรวจข้อมูลรายชั่วโมงสำหรับโมเดล 16:00 (§6.5)

```bash
python tools/intraday_probe.py
python tools/intraday_probe.py --yahoo-daily KBANK_BK_1d.csv ADVANC_BK_1d.csv   # ถ้ามีไฟล์ Yahoo 1d
```

สร้างตัวเลขใน `INTRADAY_FINDINGS.md` ขึ้นมาใหม่จาก `raw_data_intraday/` (ไม่ใช้อินเทอร์เน็ต):
SHA-256, ช่วงเวลา, แท่งต่อวันแยกตามปี, วันที่ไม่ครบ, partition ตาม `SPLIT_BY_DATE`,
ความต่างของสองแหล่ง, ความยากของโจทย์แนว A / B2 (ใช้ `cost_round_trip()` จาก
`trading_costs.py`), แท่งที่ Volume = 0 → เขียนลง `results/intraday_probe.csv`

ไม่มีโมเดลใดถูกเทรนหรือประเมินในสคริปต์นี้ (โมเดล 16:00 = Phase 1D)
