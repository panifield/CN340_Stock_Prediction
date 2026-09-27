# Investing.com daily update — ไฟล์ต้นฉบับ (2026-09-27)

ไฟล์ในโฟลเดอร์นี้คือไฟล์ที่ดาวน์โหลดจาก Investing.com **แบบ byte-for-byte ไม่แก้ไข**
ใช้เป็นหลักฐานที่มาของ 20 แถวที่ append ต่อท้าย `raw_data/*_10Y_Cleaned.csv`
**ห้ามแก้ไฟล์ในโฟลเดอร์นี้** และห้ามให้ pipeline อ่านจากที่นี่ (อ่านจาก `raw_data/` เท่านั้น)

| ไฟล์เดิม | ticker | SHA-256 | แถวข้อมูล | ช่วงวันที่ |
|---|---|---|---|---|
| `Kasikornbank Stock Price History 0831-0927.csv` | KBANK.BK | `dafe639f5198e003ec1e2f594be35c63f4f2e1e2c33301cb1cc0c43826e54de9` | 20 | 2026-08-31 → 2026-09-25 |
| `Advanced Info Stock Price History 0831-0927.csv` | ADVANC.BK | `f1d53bf1e7ddc10f92b09f17b98b5b0795166953933d6d68df4eaf319a31be13` | 20 | 2026-08-31 → 2026-09-25 |

- วันที่ได้รับ: 2026-09-27 (ไฟล์ถูกบันทึกในเครื่องผู้ใช้เวลา 22:17 +07:00)
- รูปแบบไฟล์ต้นฉบับ: UTF-8 มี BOM · LF · ทุกช่องมี quote · เรียงวันใหม่ → เก่า ·
  header `Date,Price,Open,High,Low,Vol.,Change %` · Volume เป็น `x.xxM` (ล้านหุ้น)
- การ map ticker ผ่าน 2 ทาง: (a) ชื่อไฟล์ (b) ราคาต่อเนื่อง
  |Open วันแรก / Close 2026-08-28 − 1| = KBANK 0.40% · ADVANC 0.00% (สลับ ticker จะได้ 29.5% / 42.5%)

## การแปลงเป็นรูปแบบ canonical ของ `raw_data/`

| ต้นฉบับ | canonical |
|---|---|
| `"09/25/2026"` | `09/25/2026` (MM/DD/YYYY เดิม) |
| `"252.00"` (Price/Open/High/Low) | `252.0` (float repr แบบ Python) |
| `"8.65M"` (Vol.) | `8650.0` ในคอลัมน์ `Vol. ('000)` (M × 1000 · K × 1 · B × 1,000,000 คำนวณด้วย Decimal) |
| `"0.40%"` | `0.40%` |
| เรียงใหม่ → เก่า | เรียงเก่า → ใหม่ ต่อท้ายไฟล์เดิม · CRLF · ASCII · ไม่มี quote |

แถวเก่าใน `raw_data/` ไม่ถูกแก้แม้แต่ byte เดียว (ไฟล์ใหม่ขึ้นต้นด้วย bytes ของไฟล์เดิมทั้งหมด)

| ไฟล์ | SHA-256 ก่อน append | SHA-256 หลัง append |
|---|---|---|
| `raw_data/KBANK_10Y_Cleaned.csv` | `c2068d4ce1c71db799a361b9103fe563bffc8adde418629fac01e5106fbe61f6` | `546c867d2ba9f5626655622d4ca955ab9e31fc3dc849c727b7ec5d0c8f89904e` |
| `raw_data/ADVANC_10Y_Cleaned.csv` | `cbd5db1d6dbc3dd0bce59266825ee11e8cb49e12d2bcb425a7d33e4300933c68` | `cce8069a1326a973654b7b5b7600243dc77d9e8d996d90a0b1636821737613f8` |

## สถานะของ 20 แถวนี้

post-historical-test, pre-freeze historical rows — ถูกเห็นบางส่วนแล้วผ่าน Yahoo intraday probe
(`tools/intraday_probe.py` ครอบคลุมถึง 2026-09-25) → **ไม่ใช่ prospective, ไม่ใช่ส่วนของ historical test,
ไม่ใช่ pristine** · `main.py` ตัดแถวเหล่านี้ออกจาก historical test ด้วย `config.HISTORICAL_TEST_END`
