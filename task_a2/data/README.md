# data/

โฟลเดอร์นี้เก็บเฉพาะข้อมูล **intraday** ที่ Task A2 เป็นเจ้าของเอง
(คนละอย่างกับ `task_a/data_cache/` ซึ่งเก็บข้อมูล**รายวัน** และเป็นของ
Task A — Task A2 ไม่มี cache รายวันเป็นของตัวเอง อ่านผ่าน
`task_a.data_loader.load_stock()` ตรงๆ ทุกครั้ง ดู root-cause fix ใน
`task_a/data_loader.py` ที่ทำให้แน่ใจว่าจะไม่มี `task_a2/data_cache/`
เกิดขึ้นซ้ำอีก)

**`KBANK_BK_1h_730d.csv`, `ADVANC_BK_1h_730d.csv`**
ข้อมูลหลักของ Task A2 — แท่ง 1h ย้อนหลัง 730 วันจาก yfinance แช่แข็งไว้
ให้ walk-forward CV / test ที่รายงานไปแล้ว reproduce ได้ตรงเป๊ะ (อ่านผ่าน
`features_a2.load_1h()`) **ห้ามลบ/แก้ไฟล์นี้** สำหรับข้อมูลสดใช้
`predict_live.py` แทน (ดึงแยกต่างหาก ไม่แตะไฟล์นี้)

**`KBANK_BK_5m_60d.csv`, `ADVANC_BK_5m_60d.csv`**
ข้อมูลสำรวจจาก `check_intraday.py` (เทียบ resolution 1h vs 5m ตอน
ตัดสินใจเลือกแท่งเวลา) ไม่ได้ใช้ใน pipeline หลัก เก็บไว้อ้างอิงเฉยๆ

**`*_1m_atc_check.csv`, `*_price1600_vs_close.csv`**
ผลการทดลองจาก `check_settrade_atc_match.py` (ยืนยันว่า Settrade 1m
match ราคาปิด ATC จริง) และการเทียบราคา ณ 16:00 กับราคาปิดรายวัน
ใช้ตอนตรวจสอบ data quality ก่อนสร้าง target ไม่ใช่ input ของโมเดล

**`settrade_log/`**
ข้อมูลที่ Windows Task Scheduler เก็บอัตโนมัติทุกวันจาก Settrade Open API
(ผ่าน `collect_settrade_daily.py` + `run_collector.bat`) สำหรับสะสมไว้ใช้
ต่อในอนาคต (ไม่ใช่ input ปัจจุบันของ `main_a2.py`/`predict_live.py`)
