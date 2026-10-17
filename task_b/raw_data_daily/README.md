# raw_data_daily/ — ราคาปิดทางการรายวันที่ล็อก (Phase 1D รอบ 2)

ใช้โดย `daily_close.py` เป็น target / ฐานของ feature `gap` ของโมเดล 16:00 รอบ 2 (`PHASE1D_PLAN_R2.md` ข้อ 3)
**ห้ามแก้ไฟล์ในโฟลเดอร์นี้** — `daily_close.load_locked()` ตรวจ SHA-256 ทุกครั้ง ไม่ตรง = หยุด

| ไฟล์ | แถว | ช่วงวันที่ | SHA-256 |
|---|---|---|---|
| `KBANK_BK_1d_locked.csv` | 2453 | 2016-08-26 → 2026-09-25 | `ab244d8ff4f3b3c6dff3329aa4d5e8ae94981eeb57bc5cd64473deebe664ec31` |
| `ADVANC_BK_1d_locked.csv` | 2453 | 2016-08-26 → 2026-09-25 | `641dab045cb5f5141bbdefdf41adc066ba4686a6d27d76ab895443a51a0ca800` |

คอลัมน์: `Date` (YYYY-MM-DD), `Close` (ราคาปิดทางการ ไม่ปรับปันผล), `Volume` (หุ้น)

## ที่มา
สร้างครั้งเดียวด้วย `python tools/make_locked_daily.py` จาก `raw_data/<TICKER>_10Y_Cleaned.csv`
(Investing.com + Yahoo ที่ตรวจ OHLC ทับของเดิมเป๊ะ ผ่าน `tools/fetch_yahoo_daily.py`) ตัดที่วันที่ <= 2026-09-25
(= วันสุดท้ายของไฟล์ 1h ที่ล็อกใน `raw_data_intraday/`) · ตรวจซ้ำได้ด้วย `python tools/make_locked_daily.py --check`

## ทำไมไม่ดึง Yahoo 1d สดมาเป็นประวัติ
Yahoo 1d ที่ดึงเมื่อ 2026-10-17 มีแถวเสียในปี 2024: Volume = 0 และราคาปิดค้าง (KBANK 65 วัน, ADVANC 47 วัน) ทั้งที่เป็นวันซื้อขายจริง
(raw_data มี volume, แท่ง 1h มีข้อมูล) · cache เก่าของ task_a (ดึงก่อนหน้านั้น) ตรงกับไฟล์นี้ 100% ในปี 2024
→ snapshot Yahoo ถูกใช้เฉพาะช่วง live (หลัง 2026-09-25) ที่ `raw_data_daily_live/` พร้อม SHA-256 + meta ทุกครั้งที่ดึง
