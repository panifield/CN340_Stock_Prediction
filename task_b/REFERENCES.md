# REFERENCES.md — แหล่งอ้างอิงของงาน B (feedback ข้อ 4)

> **ไฟล์นี้ทำอะไร / ตอบ feedback ข้อไหน**
> feedback ข้อ 4 (14 ก.ย. 2026) ขอให้ใส่ reference ให้ครบ ก่อนหน้านี้ `config.py` มีแค่
> คอมเมนต์ว่า *"ทุกค่าด้านล่างต้องใส่ลิงก์อ้างอิงกำกับก่อนส่งรายงาน"* ซึ่งยังไม่มีลิงก์จริง
> ไฟล์นี้คือลิงก์จริงทั้งหมด แบ่ง 4 หมวด: ข้อมูล · ค่าธรรมเนียมและ tick size · วิธีการและโมเดล ·
> ตารางโยงโค้ดกับ reference
>
> **วันที่ตรวจทุกลิงก์ในไฟล์นี้: 2026-10-24** (ยกเว้นที่ระบุไว้เป็นอย่างอื่น)
> ทุกรายการที่ยืนยันไม่ได้ติดป้าย `[ยังไม่ได้ตรวจ]` ไว้ตรง ๆ ไม่มีรายการใดถูกแต่งขึ้น

---

## 1. ข้อมูล

### 1.1 ราคาหุ้นรายวัน — แหล่งหลักของ pipeline

| สิ่งที่ใช้ | แหล่ง | ไฟล์ใน repo | ที่เก็บ SHA-256 |
|---|---|---|---|
| ราคา OHLCV รายวัน 10 ปี (KBANK, ADVANC) | **Investing.com** (ดาวน์โหลดด้วยมือ) + **Yahoo Finance** สำหรับแถวที่ append หลัง 2026-09-27 | `raw_data/KBANK_10Y_Cleaned.csv`, `raw_data/ADVANC_10Y_Cleaned.csv` | `raw_data_sources/*/SOURCES.md` (ทุกโฟลเดอร์มีตาราง SHA-256 ของไฟล์ต้นฉบับ + SHA-256 ของ `raw_data/` ก่อน/หลัง append) |

- Investing.com: <https://www.investing.com/> — หน้า Historical Data ของแต่ละหุ้น
  (โดเมนนี้คืน HTTP 403 ให้ request อัตโนมัติ จึงตรวจด้วยสคริปต์ไม่ได้ — เปิดในเบราว์เซอร์ได้ปกติ)
  (`Kasikornbank (KBANK)`, `Advanced Info Service (ADVANC)`) · ไฟล์ต้นฉบับชุดแรกเก็บไว้ที่
  [`raw_data_sources/investing_20260927/`](raw_data_sources/investing_20260927/SOURCES.md)
  พร้อม SHA-256 ของทั้งไฟล์ต้นฉบับและไฟล์ `raw_data/` ก่อน/หลัง append
- Yahoo Finance: <https://finance.yahoo.com/> — ดึงด้วย `tools/fetch_yahoo_daily.py`
  ทุกครั้งที่ append เก็บไฟล์ต้นฉบับ + SHA-256 ลง `raw_data_sources/yahoo_<YYYYMMDD>/SOURCES.md`
  (มี 13 โฟลเดอร์ ตั้งแต่ `yahoo_20260930` ถึง `yahoo_20261023`)
- `data_loader.dataset_fingerprint()` พิมพ์ SHA-256 ของไฟล์ที่โหลดทุกครั้ง และค่านี้ถูกบันทึกลง
  คอลัมน์ `dataset_sha256` ของ `results/prediction_log.csv` ทุกแถว

### 1.2 ข้อมูลประกอบ (ไม่ได้เข้า pipeline รายวันของงาน B)

| สิ่งที่ใช้ | แหล่ง | เอกสารใน repo |
|---|---|---|
| ราคาปิดทางการรายวันที่ล็อก (target ของโมเดล 16:00 รอบ 2) | Investing.com + Yahoo ที่ตรวจ OHLC ทับกันเป๊ะ สร้างด้วย `tools/make_locked_daily.py` | [`raw_data_daily/README.md`](raw_data_daily/README.md) — ตาราง SHA-256 ของทั้งสองไฟล์ |
| ราคารายชั่วโมง 722 วันทำการ (2023-10-02 → 2026-09-25) | **Yahoo Finance** interval `1h` (คนละแหล่งกับ `raw_data/`) | [`raw_data_intraday/README.md`](raw_data_intraday/README.md) — ตาราง SHA-256 + บันทึกว่าราคาปิดต่างจาก `raw_data/` เฉลี่ย 0.33 บาท (KBANK) / 0.69 บาท (ADVANC) |

> `data_loader.py` ปฏิเสธไฟล์ที่ชื่อมี `_1h` หรือ path มี `intraday` เพื่อกันข้อมูลคนละแหล่ง
> หลุดเข้า pipeline รายวันโดยเงียบ

### 1.3 ปฏิทินวันหยุดตลาดหลักทรัพย์

| สิ่งที่ใช้ | แหล่ง | ไฟล์ใน repo |
|---|---|---|
| วันหยุด SET ที่ตรงกับวันจันทร์–ศุกร์ | **ประกาศวันหยุดตามประเพณีของ SET** — <https://www.set.or.th/th/about/event-calendar/holiday> | [`set_holidays.txt`](set_holidays.txt) |

- ไฟล์นี้เป็น **แหล่งเดียว** ของปฏิทินวันหยุดในงาน B (`daily_next_day.py`, `daily_1600.py` อ่านจากที่นี่
  · `predict_live.py` ไม่เดาวันหยุดเอง)
- กติกาในไฟล์: เพิ่มวันหยุดจากประกาศทางการของ SET เท่านั้น ห้ามเดา · มีบรรทัด
  `confirmed_through:` บอกว่ารายการครบถึงวันไหน และสคริปต์จะไม่ทำนายวันที่เกินค่านั้น
- ช่วง 2026-01-01 → 2026-09-25 ได้มาโดยอนุมานจากวันจันทร์–ศุกร์ที่ไม่มีใน `raw_data/`
  (= ตลาดปิดจริง) ตามที่หัวไฟล์เขียนไว้ **ไม่ใช่คัดลอกจากประกาศทีละวัน** —
  `[ยังไม่ได้ตรวจทีละวันกับประกาศ]`

---

## 2. ค่าธรรมเนียมและ tick size

### 2.1 แหล่งที่ตรวจแล้ว (ตรวจ 2026-10-24)

| # | แหล่ง | ประเภท | URL | สิ่งที่ยืนยันได้ |
|---|---|---|---|---|
| S1 | Stock Exchange of Thailand — *Price Bands and Tick Sizes* | ทางการ (SET) | <https://www.set.or.th/en/market/information/trading-procedure/price-bands-and-tick-sizes> | ตาราง tick size 8 ช่วง มีผลตั้งแต่ 30 มี.ค. 2552 · ยกเว้น unit trust / ETF ที่ tick = 0.01 ทุกระดับราคา |
| S2 | SET — *Market Consultation: Revision of Measures to Enhance Confidence* (พ.ค. 2569) | ทางการ (SET) | <https://media.set.or.th/set/Documents/2026/May/No21-ConsultativePaper-Proposed-Revision-Measures-Enhance-Confidence.pdf> | SET เสนอปรับ tick ช่วง 2–50 บาท · ยืนยันว่าตารางเดิมใช้มาตั้งแต่ มี.ค. 2552 |
| S3 | SET — *Hearing Result* (มิ.ย. 2569) | ทางการ (SET) | <https://media.set.or.th/set/Documents/2026/Jun/No21-HearingResult-Proposed-Revision-Measures-Enhance-Confidence.pdf> | ผลรับฟังความเห็น 13–29 พ.ค. 2569 · SET จะปรับร่างก่อนเสนอ ก.ล.ต. -> **ตารางใหม่ยังไม่มีผลบังคับ ณ วันที่ตรวจ** |
| B1 | **DAOL Securities (Thailand)** — ตารางอัตราค่าธรรมเนียม Equity + TFEX ลงวันที่ 25/05/2026 | โบรกเกอร์จริง | <https://www.daolsecurities.co.th/assets/79216a95-e286-4120-a4b8-ba6ea02ac2e4/Equity+TFEX%20Commission_25052026.pdf> | ONLINE บัญชี Cash Balance / Credit Balance: **0.15%** (≤ 5 ล้านบาท/วัน), 0.13% (5–10), 0.11% (10–20) · Cash Account 0.20% / IC 0.25% · ระบุ Trading Fee 0.005%, Clearing Fee 0.001%, Regulatory Fee 0.001% |
| B2 | **InnovestX Securities** — ค่าธรรมเนียมหุ้นไทย | โบรกเกอร์จริง | <https://www.innovestx.co.th/products/thai-stock-fee> | Internet Cash Account: **0.20%** (≤ 5 ล้านบาท), 0.18% / 0.15% / 0.12% ตามขั้นมูลค่า · SET Trading Fee **0.005%** · TSD Clearing Fee **0.001%** · Regulatory Fee **0.001%** · **VAT 7% ของค่าธรรมเนียมทั้งหมด** · ขั้นต่ำ 50 บาท/วัน (มีเงื่อนไขยกเว้น) |
| B3 | หน้าอัตราค่าธรรมเนียมบนโดเมน Settrade (broker 023, แพลตฟอร์ม INVX / ลิงก์ไป SCB) | โบรกเกอร์จริง (ชื่อบริษัทเป็นการอนุมานจากโลโก้/ลิงก์ → `[ชื่อบริษัทยังไม่ได้ตรวจ]`) | <https://wwwa1.settrade.com/brokerpage/023/StaticPage/home/attachfile/equitycomm.html> | ช่อง Internet: **0.157%** (≤ 5 ลบ.), 0.137% / 0.117% / 0.107% · IC cash account 0.257% · Trading 0.005% / Clearing 0.001% / Regulatory 0.001% + VAT 7% · ขั้นต่ำ 50 บาท/วัน · **Regulatory fee ลดจาก 0.0018% → 0.0010% มีผล 4 ม.ค. 2560 ตามประกาศ ก.ล.ต.** |

**ที่หาไม่ได้ (บันทึกไว้ตรง ๆ)**
- ไม่พบหน้าบน `set.or.th` หรือ `tch.or.th` (TCH / TSD) ที่ประกาศอัตรา trading fee 0.005% /
  clearing fee 0.001% แบบเป็นเอกสารทางการโดยตรง — ค่าทั้งสามจึงยืนยันผ่าน **ตารางค่าธรรมเนียม
  ของโบรกเกอร์ 3 รายที่ตรงกันทุกราย** (B1 / B2 / B3) ไม่ใช่จากประกาศต้นทาง →
  `[ยังไม่ได้ตรวจจากประกาศต้นทางของ SET/TCH]`
- ไม่พบเลขที่ประกาศของ ก.ล.ต. ที่ลด regulatory fee เป็น 0.0010% — ยืนยันได้แค่ว่าโบรกเกอร์ (B3)
  อ้างว่าเป็นประกาศ ก.ล.ต. มีผล 4 ม.ค. 2560 → `[เลขที่ประกาศยังไม่ได้ตรวจ]`

### 2.2 ค่าใน config vs ค่าที่ตรวจได้

**`SET_COSTS`** (ไฟล์ [`config.py`](config.py) — *ไม่แก้ตัวเลขใด ๆ ในรอบนี้*)

| key | ค่าใน `config.py` | ค่าที่ตรวจได้ | ตรงกันไหม | แหล่ง |
|---|---|---|---|---|
| `commission_per_side` | `0.0015` (0.15%) | **0.15%** = DAOL ONLINE Cash Balance ≤ 5 ลบ. · แต่โบรกอื่นในช่องทางออนไลน์คิด **0.157%** (B3) หรือ **0.20%** สำหรับ Cash Account (B2) | ตรงกับ 1 ใน 3 ราย — เป็น **ปลายต่ำ** ของช่วงที่ตรวจได้ (0.15%–0.25%) | B1 / B2 / B3 |
| `trading_fee` | `0.00005` (0.005%) | 0.005% | ตรง (ทั้ง 3 ราย) | B1 / B2 / B3 |
| `clearing_fee` | `0.00001` (0.001%) | 0.001% | ตรง (ทั้ง 3 ราย) | B1 / B2 / B3 |
| `regulatory_fee` | `0.00001` (0.001%) | 0.001% (ลดจาก 0.0018% เมื่อ 4 ม.ค. 2560) | ตรง | B1 / B2 / B3 |
| `vat` | `0.07` | VAT 7% คิดบนค่าธรรมเนียมทั้งหมด ไม่ใช่บนมูลค่าหุ้น | ตรง | B2 / B3 |

ผลรวมต่อขา **ก่อน** VAT = `0.0015 + 0.00005 + 0.00001 + 0.00001 = 0.00157` ซึ่งตรงกับเลข
**0.157%** ที่โบรกเกอร์มักโฆษณาแบบรวมค่าธรรมเนียมแล้วพอดี · ต่อขา **หลัง** VAT =
`0.00157 × 1.07 = 0.0016799` · ไป-กลับ = `2 × 0.0016799 = 0.0033598` = **33.598 bps**
(ตรงกับ `breakeven_bps` ในทุกไฟล์ `results/*_taskB_val_costs.csv`)

**`SET_TICK_BANDS`**

| ช่วงราคา (บาท) | tick ใน `config.py` | tick ตามหน้า SET (S1) | ตรงกันไหม |
|---|---|---|---|
| < 2 | 0.01 | 0.01 | ตรง |
| 2 – < 5 | 0.02 | 0.02 | ตรง |
| 5 – < 10 | 0.05 | 0.05 | ตรง |
| 10 – < 25 | 0.10 | 0.10 | ตรง |
| 25 – < 100 | 0.25 | 0.25 | ตรง |
| 100 – < 200 | 0.50 | 0.50 | ตรง |
| 200 – < 400 | 1.00 | 1.00 | ตรง |
| ≥ 400 | 2.00 | 2.00 | ตรง |

**ตรงทั้ง 8 ช่วง** · สองข้อควรรู้: (ก) ตารางของ SET ไม่ใช้กับ unit trust / ETF (tick 0.01 ทุกระดับ)
ซึ่งไม่กระทบงาน B ที่มีแต่หุ้นสามัญ (ข) SET เสนอปรับ tick ช่วง 2–50 บาทเมื่อ พ.ค. 2569 แต่
**ยังไม่มีผลบังคับ** ณ วันที่ตรวจ (S2 / S3) — ถ้ามีผลในอนาคต ช่วง 25–100 บาทอาจเปลี่ยน
และต้องรีวิว `SET_TICK_BANDS` อีกครั้ง (ยังไม่ต้องแก้ตอนนี้)

### 2.3 ความต่างระหว่าง task: task_b 0.0015 vs task_a / task_a2 0.00157

| | task_b | task_a / task_a2 |
|---|---|---|
| ค่าคงที่ | `SET_COSTS["commission_per_side"] = 0.0015` ([`config.py`](config.py)) | `COMMISSION_RATE = 0.00157` (`task_a/supplementary_analysis.py:59`) · `COMMISSION = 0.00157` (`task_a2/main_a2.py:352`) |
| ความหมาย | ค่าคอมมิชชันล้วน แล้ว **บวก** trading / clearing / regulatory fee + VAT ต่อ | อัตราต่อขาแบบ "จบในตัว" (= 0.157% ที่โบรกโฆษณา) แล้วบวก VAT + market fee 0.00001 |
| สูตรไป-กลับ | `2 × (0.0015 + 0.00005 + 0.00001 + 0.00001) × 1.07` | `2 × (0.00157 × 1.07 + 0.00001)` |
| ผลลัพธ์ | **0.0033598** (33.598 bps) | **0.0033998** (33.998 bps) |

ต่างกัน **0.4 bps (1.2% ของต้นทุน)** — ทิศทางข้อสรุปเหมือนกัน แต่เป็นการนับ VAT กับ market fee
คนละแบบ · **ไม่ได้แก้ task อื่นในรอบนี้** (อยู่นอก `task_b/`) บันทึกไว้ที่นี่เพื่อให้คนอ่านรายงาน
สองงานพร้อมกันไม่สับสน

### 2.4 ข้อสรุป "สัญญาณเล็กกว่าต้นทุน" ไวต่อค่าคอมมิชชันแค่ไหน

`trading_costs.cost_sensitivity()` (ฟังก์ชันใหม่ ไม่แก้ของเดิม) + `main.py --dev` เซฟ
[`results/cost_sensitivity.csv`](results/cost_sensitivity.csv) ให้แล้ว · grid ประกาศไว้ใน
`trading_costs.COMMISSION_GRID` = `[0.0010, 0.0015, 0.00157, 0.0020, 0.0025]`
(คลุมช่วงที่ตรวจได้จริงจากตารางโบรกเกอร์ทั้ง 3 ราย) ·
ค่า `breakeven_bps` ที่ได้: 22.898 / 33.598 / 35.096 / 44.298 / 54.998 bps

> ตารางนี้เป็น **diagnostic** เหมือนตารางที่สามใน `main.py` — ไม่ถูกใช้เลือกโมเดล
> อ่านผลที่ได้จริงใน `MODEL_RATIONALE.md` และ `results/feedback/WHY_NAIVE_WINS.md`

---

## 3. วิธีการและโมเดล

ทุก DOI ในหมวดนี้ตรวจผ่าน Crossref API (`https://api.crossref.org/works/<doi>`) เมื่อ 2026-10-24
และชื่อเรื่อง / ผู้แต่ง / ปี / วารสาร / เล่ม / หน้า ตรงกับที่เขียนไว้

### 3.1 Random walk / efficient market hypothesis

| # | reference | ตรวจแล้ว |
|---|---|---|
| R1 | Fama, E. F. (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work.* **The Journal of Finance, 25**(2), 383–417. DOI: [10.2307/2325486](https://doi.org/10.2307/2325486) | ✅ Crossref |
| R2 | Malkiel, B. G. *A Random Walk Down Wall Street.* W. W. Norton & Company. (ฉบับพิมพ์ครั้งแรก 1973 · มีการปรับปรุงหลายครั้ง) | `[เลขหน้า/ฉบับพิมพ์ยังไม่ได้ตรวจ]` — เป็นหนังสือ ไม่มี DOI · อ้างเป็นแนวคิดเท่านั้น |

ทำไมเกี่ยวกับงาน B: baseline หลักของเราคือ **Naive (random walk)** = ทำนาย return = 0
(`baselines.baseline_naive_return`) ซึ่งเป็นสิ่งที่ EMH แบบ weak form บอกว่าควรเอาชนะได้ยาก
ด้วยข้อมูลราคาในอดีตเพียงอย่างเดียว

### 3.2 โมเดล

| # | reference | ตรวจแล้ว |
|---|---|---|
| R3 | Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors.* **Nature, 323**(6088), 533–536. DOI: [10.1038/323533a0](https://doi.org/10.1038/323533a0) | ✅ Crossref |
| R4 | Breiman, L. (1996). *Bagging predictors.* **Machine Learning, 24**(2), 123–140. DOI: [10.1007/BF00058655](https://doi.org/10.1007/BF00058655) | ✅ Crossref |
| R5 | Breiman, L. (2001). *Random Forests.* **Machine Learning, 45**(1), 5–32. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324) | ✅ Crossref |
| R6 | Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* In **Proceedings of the 22nd ACM SIGKDD** (pp. 785–794). DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785) | ✅ Crossref |
| R7 | Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* **Journal of Machine Learning Research, 12**, 2825–2830. <https://www.jmlr.org/papers/v12/pedregosa11a.html> | ✅ URL คืน 200 และ `<title>` ตรงชื่อเรื่อง |

### 3.3 การประเมินผลและการทดสอบนัยสำคัญ

| # | reference | ตรวจแล้ว |
|---|---|---|
| R8 | Diebold, F. X., & Mariano, R. S. (1995). *Comparing Predictive Accuracy.* **Journal of Business & Economic Statistics, 13**(3), 253–263. DOI: [10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599) | ✅ Crossref |
| R9 | Harvey, D., Leybourne, S., & Newbold, P. (1997). *Testing the equality of prediction mean squared errors.* **International Journal of Forecasting, 13**(2), 281–291. DOI: [10.1016/S0169-2070(96)00719-4](https://doi.org/10.1016/S0169-2070%2896%2900719-4) | ✅ Crossref |
| R10 | Bi, J., & Bennett, K. P. (2003). *Regression Error Characteristic Curves.* In **Proceedings of the 20th International Conference on Machine Learning (ICML-2003)**, 43–50. <https://cdn.aaai.org/ICML/2003/ICML03-009.pdf> | ✅ ดาวน์โหลด PDF แล้วอ่านหน้าแรก: ชื่อเรื่องและผู้แต่ง (Rensselaer Polytechnic Institute) ตรง |
| R11 | Bergmeir, C., & Benítez, J. M. (2012). *On the use of cross-validation for time series predictor evaluation.* **Information Sciences, 191**, 192–213. DOI: [10.1016/j.ins.2011.12.028](https://doi.org/10.1016/j.ins.2011.12.028) | ✅ Crossref |
| R12 | Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice* (3rd ed.), OTexts. บทที่เกี่ยวข้อง: *Time series cross-validation* <https://otexts.com/fpp3/tscv.html> | ✅ URL คืน 200 |

### 3.4 Hyperparameter / feature search

| # | reference | ตรวจแล้ว |
|---|---|---|
| R13 | Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). *Algorithms for Hyper-Parameter Optimization.* **Advances in Neural Information Processing Systems 24**. <https://papers.nips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html> | ✅ URL คืน 200 และ `<title>` ตรงชื่อเรื่อง (TPE = Tree-structured Parzen Estimator มาจากที่นี่) |
| R14 | Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* In **Proceedings of the 25th ACM SIGKDD** (pp. 2623–2631). DOI: [10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701) | ✅ Crossref |

### 3.5 Technical indicator

| # | reference | ตรวจแล้ว |
|---|---|---|
| R15 | Wilder, J. W. (1978). *New Concepts in Technical Trading Systems.* Trend Research, Greensboro, NC. ISBN 0-89459-027-8 (ต้นฉบับของ RSI) | `[ISBN / เลขหน้ายังไม่ได้ตรวจ]` — หนังสือเก่า ไม่มี DOI และไม่มีหน้าเผยแพร่ทางการ |
| R16 | Appel, G. — ผู้คิด MACD (ทศวรรษ 1970) · ฉบับที่อ้างได้: Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors.* FT Prentice Hall. | `[ยังไม่ได้ตรวจ]` — ไม่ยืนยัน ISBN / ปีของสิ่งพิมพ์ต้นฉบับที่ MACD ปรากฏครั้งแรก |
| R17 | Bollinger, J. (2001). *Bollinger on Bollinger Bands.* McGraw-Hill. | `[ยังไม่ได้ตรวจ]` |

> หมายเหตุ: R15–R17 เป็นหนังสือของผู้คิดค้น indicator ซึ่งตรวจ metadata ออนไลน์ไม่ได้ในรอบนี้
> **สูตรที่โค้ดใช้จริง** เขียนไว้ครบใน `features.py` และตรงกับสูตรมาตรฐานที่ใช้กันทั่วไป
> (RSI แบบ Wilder smoothing ด้วย `ewm(alpha=1/period, adjust=False)` · MACD 12/26/9 ·
> Bollinger 20 วัน ± 2 SD) — ถ้าต้องอ้างแบบตรวจได้ ให้อ้างจากตำราวิเคราะห์ทางเทคนิคที่มี DOI แทน

---

## 4. ตารางโยง — สิ่งที่ใช้ในโค้ด → ไฟล์/ฟังก์ชัน → reference

| สิ่งที่ใช้ | ไฟล์ / ฟังก์ชัน | reference |
|---|---|---|
| ราคา OHLCV รายวัน | `data_loader.load_stock` ← `raw_data/*_10Y_Cleaned.csv` | หมวด 1.1 (Investing.com / Yahoo Finance + SHA-256 ใน `raw_data_sources/*/SOURCES.md`) |
| ปฏิทินวันหยุด SET | `daily_next_day.py`, `daily_1600.py` ← `set_holidays.txt` | หมวด 1.3 |
| ทำนาย **return** แทนราคาดิบ | `targets.build_targets` (docstring) · `features.py` (หัวไฟล์) | R1 (random walk / EMH) · ดู `MODEL_RATIONALE.md` |
| Baseline: Naive (RW) | `baselines.baseline_naive_return` | R1, R2 |
| RSI (14) | `features.rsi` | **R15 (Wilder 1978)** |
| MACD (12/26/9) | `features.macd` | **R16 (Appel)** |
| Bollinger %B (20, 2 SD) | `features.bollinger_position` | **R17 (Bollinger)** |
| ANN = MLPRegressor (backprop) | `models.get_regressors` → `sklearn.neural_network.MLPRegressor` | **R3 (Rumelhart et al. 1986)**, R7 |
| Random Forest | `models.get_regressors` → `sklearn.ensemble.RandomForestRegressor` | **R5 (Breiman 2001)** + R4 (bagging) , R7 |
| XGBoost | `models.get_regressors` → `xgboost.XGBRegressor` | **R6 (Chen & Guestrin 2016)** |
| StandardScaler / Pipeline / TransformedTargetRegressor | `models._scaled`, `models._unscaled`, `main.run_task_b` | R7 |
| Walk-forward / time-series CV (ห้าม shuffle) | `splits.walk_forward_splits`, `splits.chronological_split` | **R11 (Bergmeir & Benítez 2012)**, R12 (FPP3 §tscv) |
| C-Curve → ตีความเป็น REC curve | `curves.py`, `results/curves/` | **R10 (Bi & Bennett 2003)** |
| Diebold–Mariano test + HLN small-sample correction | `stats_tests.diebold_mariano` | **R8 (Diebold & Mariano 1995)**, **R9 (Harvey et al. 1997)** |
| TPE sampler ที่ Optuna ใช้ | `optuna_features.py` → `optuna.samplers.TPESampler` | **R13 (Bergstra et al. 2011)** |
| กรอบการค้น hyperparameter / feature | `optuna_features.py` → Optuna | **R14 (Akiba et al. 2019)** |
| tick size 8 ช่วง | `config.SET_TICK_BANDS`, `trading_costs.tick_size` | **S1** (+ S2/S3 สำหรับร่างปรับปรุงที่ยังไม่มีผล) |
| ต้นทุนไป-กลับ / breakeven | `config.SET_COSTS`, `trading_costs.cost_round_trip` | **B1, B2, B3** (หมวด 2.2) |
| Sensitivity ของต้นทุน | `trading_costs.cost_sensitivity`, `results/cost_sensitivity.csv` | หมวด 2.4 |

---

## 5. สรุปสิ่งที่ยังตรวจไม่ได้ (ไม่มีรายการใดถูกแต่งขึ้น)

1. ประกาศต้นทางของ SET / TCH สำหรับอัตรา trading fee 0.005% และ clearing fee 0.001%
   — ยืนยันผ่านตารางโบรกเกอร์ 3 รายที่ตรงกัน ไม่ใช่ประกาศต้นทาง
2. เลขที่ประกาศ ก.ล.ต. ที่ลด regulatory fee 0.0018% → 0.0010% (4 ม.ค. 2560)
3. ชื่อบริษัทของโบรกเกอร์ B3 (อนุมานจากโลโก้/ลิงก์บนหน้า)
4. ปฏิทินวันหยุด SET ช่วง 2026-01-01 → 2026-09-25 ใน `set_holidays.txt`
   (อนุมานจากวันที่ตลาดปิดจริงใน `raw_data/` ไม่ได้เทียบประกาศทีละวัน)
5. metadata ของหนังสือ indicator R15–R17 (Wilder / Appel / Bollinger)
