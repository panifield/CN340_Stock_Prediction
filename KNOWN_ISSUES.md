# Known Issues — สรุปปัญหาทั้งหมด แยกตาม task

> อัปเดตล่าสุด: 2026-09-28 (รวมผลหลัง pull commit `aaae8bd`)
> คอลัมน์ "ที่มา": `เดิม` = พบไปแล้วรอบก่อน, `ใหม่` = พบเพิ่มจากการไล่โค้ดรอบนี้
> คอลัมน์ "สถานะ": อัปเดตหลังเพื่อน pull โค้ดใหม่เข้ามา (2026-09-28)

## task_a (คู่/คี่ next_day)

| # | ปัญหา | รายละเอียด | ที่มา | สถานะ |
|---|-------|------------|-------|-------|
| 1 | ไม่มี live-predict script | `main.py` ทำได้แค่ประเมินย้อนหลัง (CV/test) ไม่มีสคริปต์ "ทายวันพรุ่งนี้จริง" แบบ `task_b/predict_live.py` | เดิม | ✅ แก้แล้ว — เพิ่ม `task_a/predict_live.py` |
| 2 | `data_cache/` ค้างวันที่ | ค้างที่ 2026-08-28 (config `END_DATE` ตายตัวใน `config.py`) ทั้งที่วันนี้คือ 2026-09-28 — ข้อมูลขาดไปเกือบ 1 เดือน | เดิม | ℹ️ ชี้แจงแล้ว ไม่ใช่บั๊ก — `task_a/data_cache/README.md` (ใหม่) ระบุว่าแช่แข็ง**โดยตั้งใจ**เพื่อให้ผล CV/test เดิม reproduce ได้ ข้อมูลสดให้ใช้ `predict_live.py` แทน (ดึงแยก ไม่แตะไฟล์นี้) |
| 3 | ไฟล์ลอยไม่รู้ที่มา | พบ `KBANK_BK_fresh_2026-08-29_2026-09-27.csv` / `ADVANC_BK_fresh_...csv` ใน `data_cache/` มีข้อมูลถึง 2026-09-25 แต่หาสคริปต์ต้นทางที่สร้างไม่เจอ — ควรถามเจ้าของเดิมหรือลบทิ้งถ้าไม่มีที่มา | เดิม | ✅ แก้แล้ว — ไฟล์ลอยทั้งสองถูกลบออกจาก repo แล้ว |
| 4 | cache หุ้นอื่นตกค้าง | มี cache ของหุ้นอื่นอีก ~25 ตัว (AOT, BBL, PTT, SCC ฯลฯ) ที่ไม่ใช่ KBANK/ADVANC เดาว่ามาจาก `scan_universe.py` — ไม่ใช่บั๊กแต่ทำให้โฟลเดอร์รก | เดิม | ℹ️ ชี้แจงแล้ว — README ยืนยันว่ามาจาก `scan_universe.py` จริง (Stage 1 screening) ไม่ใช่ไฟล์ลอย |

## task_a2 (คู่/คี่ t1600)

| # | ปัญหา | รายละเอียด | ที่มา | สถานะ |
|---|-------|------------|-------|-------|
| 1 | ไม่มี live-predict script | มีแต่ backtest จาก CSV นิ่ง (`data/*_1h_730d.csv`) | เดิม | ✅ แก้แล้ว — เพิ่ม `task_a2/predict_live.py` (ดึงสดทั้ง 1h และ daily ทุกครั้ง ไม่พึ่งไฟล์นิ่ง) |
| 2 | ไฟล์ intraday ค้างวันที่ | ค้างที่ 2026-09-25 (3 วันก่อนวันนี้) ไม่ auto-update | เดิม | ✅ แก้แล้ว — `predict_live.py` ดึงสดตรงจาก yfinance ทุกครั้งแทนการอ่านไฟล์ค้าง (ไฟล์ `data_cache/` เดิมถูกลบทิ้งไปเลย ดูข้อ 4) |
| 3 | มีของดีต่อยอดได้ | `check_intraday.py` พิสูจน์แล้วว่า `interval="1h", period="730d"` จาก yfinance ดึงสดได้จริง — เอามาต่อยอดทำ live fetch ได้เลย ไม่ต้องเริ่มจากศูนย์ | เดิม | ✅ ถูกใช้จริงแล้วใน `predict_live.py` |
| 4 | พึ่งพา `task_a/data_cache` ตรงๆ | path hardcode ในหลายไฟล์ (`build_price_at_1600.py`, `check_intraday.py`, `check_settrade_atc_match.py`) ชี้ไปที่ `../task_a/data_cache/{ticker}_2016-08-26_2026-08-28.csv` เพื่อเอาราคาปิดรายวัน — **ยืนยันจากโค้ดแล้วว่าเป็นบั๊กเดียวกับข้อ 2 ของ task_a** ไม่ใช่สองบั๊กแยกกัน อย่าไปแก้ freshness ซ้ำสองที่ ให้คิดเป็น single source of truth ของ daily cache ทีเดียว | เดิม + ยืนยันเพิ่ม (ใหม่) | ✅ แก้แล้ว — commit "single source of truth for daily data_cache" ลบ `task_a2/data_cache/{KBANK,ADVANC}_BK_2016-08-26_2026-08-28.csv` (ไฟล์ซ้ำ) ออก ตอนนี้อ่านผ่าน `task_a.data_loader.load_stock()` ตรงๆ จุดเดียว |
| 5 | โฟลเดอร์ชื่อคล้ายกันสับสน | `task_a2/data/` กับ `task_a2/data_cache/` ชื่อคล้ายกันมากแต่เก็บคนละอย่าง (`data_cache/` = ราคารายวัน/1h ของ task_a2 เอง, `data/` = settrade log + atc-check + ไฟล์ที่ยืมจาก task_a) ไม่มีคำอธิบายในโค้ดว่าอันไหนคือของจริง | ใหม่ | ✅ แก้แล้ว — เพิ่ม `task_a2/data/README.md` อธิบายทุกไฟล์ชัดเจน และ `data_cache/` ของ task_a2 ถูกเลิกใช้ไปแล้ว (ดูข้อ 4) |

## task_b (ราคา, บาท)

| # | ปัญหา | รายละเอียด | ที่มา | สถานะ |
|---|-------|------------|-------|-------|
| 1 | โหมด next_day พึ่งข้อมูลมือ | อ่านข้อมูลจาก `raw_data/*.csv` (investing.com) เท่านั้น ไม่มี fallback ไป Yahoo ต้องอัปเดตด้วยมือทุกวัน (ดาวน์โหลดจาก investing.com เอง แล้วรัน `tools/append_investing.py`) automate เต็มรูปแบบด้วย GitHub Actions ไม่ได้เพราะ investing.com ไม่มี API — ข้อมูลปัจจุบันค้างที่ 2026-09-25 | เดิม | ✅ แก้แล้ว — เพิ่ม `tools/fetch_yahoo_daily.py` (connector Yahoo→รูปแบบ Investing ตรวจ OHLC ทับกันเป๊ะ 100%) + `daily_next_day.py` รวมทุกขั้นตอนอัตโนมัติ + `automation/github_actions_task_b_daily.yml` และ `register_tasks.ps1` (ยังเป็น template อยู่ใน `task_b/automation/` ต้องย้าย/ตั้งค่าเองถึงจะรันจริงตามตาราง) |
| 2 | โหมด t1600 ถูกล็อกไว้ | ต้องมี `PHASE1D_LIVE_APPROVAL.md` + หลักฐาน live dry-run ก่อน (ยังไม่มีทั้งคู่ — มีแค่ `PHASE1D_PLAN.md` ที่เป็นแผน ไม่ใช่เอกสารอนุมัติ) และทุก config ที่เคยลองมาแพ้ baseline Naive อยู่แล้ว | เดิม | ✅ ทีมยืนยันแล้ว (2026-09-28) — commit `afe2968` ยกเลิกเกทอนุมัติเดิม เปลี่ยนไปตรวจ real-time availability แบบย้อนหลังแทน (`prev_close_match`, `bar15_changed_vs_later`) บันทึกเหตุผลไว้ใน `PHASE1D_PLAN.md` ข้อ 18 — ทีมโอเคกับการเปลี่ยนนี้แล้ว **ข้อควรระวังที่ยังอยู่**: การเปิด official mode ได้ ≠ โมเดลดีพอใช้จริง (ยังแพ้ baseline Naive อยู่) จุดนี้เป็นคนละเรื่องกับเกทที่ยกเลิกไป |
| 3 | ยังไม่มี connector ตามที่ออกแบบไว้ | `INTEGRATION.md` ที่ผู้เขียนเดิมออกแบบไว้แล้วว่าต้องมี "ตัวเชื่อมกลาง" (connector) คอยดึงข้อมูลจากภายนอกแล้วป้อนเป็นไฟล์ให้ — ตัวเชื่อมนี้ยังไม่มีใครสร้าง (root cause ของปัญหาข้อ 1) | เดิม | ✅ แก้แล้ว — `tools/fetch_yahoo_daily.py` คือตัวเชื่อมนี้ (`INTEGRATION.md` อัปเดตตามแล้ว) |
| 4 | requirements.txt pin เวอร์ชันเป๊ะ | `numpy==2.1.1, pandas==2.3.3, scikit-learn==1.9.1, xgboost==3.4.1` ห้ามเปลี่ยนโดยไม่รัน regression check — ต้องระวังตอน merge กับ dependency ของ task อื่น | เดิม | ↔️ คงเดิม (แก้แค่คอมเมนต์เป็น ASCII เพราะ pip บน Windows อ่านไฟล์นี้เป็น cp1252) เพิ่ม `requirements-live.txt` ใหม่สำหรับ dependency เฉพาะฝั่ง live fetch |
| 5 | ห้ามรันไม่มี `--dev` | รัน `python main.py` แบบไม่มี `--dev` จะพยายามเปิด historical test ที่ล็อกไว้ (มี `verify_lock()` เช็ก `PRE_TEST_LOCK.md` เป็นตัวบล็อกจริงในโค้ด ไม่ใช่แค่คำเตือน) เพื่อนต้องรู้กฎนี้ก่อนแตะโค้ด | เดิม | ↔️ คงเดิม กฎยังอยู่เหมือนเดิม |

## task_c (ขึ้น/ลง)

| # | ปัญหา | รายละเอียด | ที่มา | สถานะ |
|---|-------|------------|-------|-------|
| 1 | ไม่มี live-predict script | สำหรับ next_day (Task 1) เช่นกัน | เดิม | ✅ แก้แล้ว — เพิ่ม `task_c/predict_live.py` (ดึงข้อมูลสดแยกจาก `data_cache/` เหมือน task_a/task_a2 · เลือกโมเดลด้วย walk-forward OOF แล้ว refit บนข้อมูลทั้งหมด · ทดสอบ `--dry-run` แล้วรันผ่านจริง เขียนลง `results/dryrun/prediction_log_dryrun.csv`) — **ไม่แตะ `intraday_task2.py`/cutoff เลย** เพราะเพื่อนรับไปแก้เอง |
| 2 | "Task 2" ไม่ตรงนิยาม t1600 | `intraday_task2.py` cutoff เป็น 13:00 ไม่ใช่ 16:00 และ target เป็นราคาจากแท่งกราฟที่ไม่เคยตรวจว่า = ราคาปิดทางการ (ต่างจาก task_a2 ที่ตรวจแล้วว่าตรง ATC 100%) | เดิม | 🔴 ตรวจแล้ว พบปัญหาจริง ร้ายแรงกว่าที่คิด (2026-09-29) — เขียน `task_c/check_settrade_atc_match.py` (mirror จาก task_a2) เทียบแท่ง 16:00 ของ `data_cache/{ticker}_1h_730d.csv` (yfinance รายชั่วโมง) กับราคาปิดรายวันทางการจาก Settrade เอง (interval=1d) ย้อนหลัง 721 วัน: **KBANK ตรงกันแค่ 46.0% (332/721), ADVANC ตรงกันแค่ 40.2% (290/721)** — ต่างจาก task_a2 ที่ตรง ATC 100% คนละเรื่องเลย ส่วนใหญ่ที่ไม่ตรงต่างกันแค่ ~1 tick (0.5-1 บาท จาก mean diff ~0.02-0.03, std ~0.5-0.96) แต่ก็มากพอจะพลิกทิศทาง "ขึ้น/ลง" เทียบกับราคา cutoff ได้ในหลายวัน — สาเหตุน่าจะเป็นเพราะแท่ง 1h ของ yfinance คือราคาซื้อขายต่อเนื่องช่วง 15:00-16:00 ไม่ใช่ราคาจากรอบ ATC (auction) ตอนปิดตลาดจริง ซึ่งเป็นกลไกคนละแบบกัน (ขณะที่ yfinance **daily** close ที่ task_b ยืนยันแล้วว่าตรง 100% กับ investing.com คือคนละ query กับ 1h bar นี้) — **นี่อาจอธิบายส่วนหนึ่งว่าทำไม target ของ Task 2 "ไม่ใช่สิ่งที่ตั้งใจจะทาย" อย่างแท้จริง** (ทายทิศทางเทียบกับราคาซื้อขายต่อเนื่อง ไม่ใช่ราคาปิดทางการ) — **✅ แก้แล้ว (2026-09-29, สั่งให้แก้เลยโดยไม่ต้องรอคุยกับเพื่อนก่อน)**: `intraday_task2.py` เพิ่ม `load_daily_close()` (ดึง yfinance interval=1d, cache แยกที่ `data_cache/{ticker}_daily_close.csv`) แล้วใช้แทนแท่ง 1h ที่ Hour==16 เป็น target (`build_task2_dataset(..., daily_close=...)`) — แก้ทั้ง `intraday_task2.py` (backtest) และ `task_c/predict_live_1600.py` (live) ให้ใช้ target เดียวกัน ทดสอบ `--dev` แล้วรันผ่าน: **ผล accuracy ลดลงมาใกล้ base rate มากขึ้น** (KBANK XGBoost 67.6%, ADVANC XGBoost 66.7% — ใกล้กับสัดส่วน "ลง/นิ่ง" ตามธรรมชาติ ~68% ทั้งคู่, recall ของฝั่ง "ขึ้น" ต่ำมาก 8-21% แปลว่าโมเดลแทบไม่มีสัญญาณจริง เกือบทายแค่ majority class) — ตัวเลขนี้ **น่าเชื่อถือกว่าของเดิมมาก** เพราะไม่มีปัญหา target ผิดแล้ว แม้จะดูไม่หวือหวาเท่าเดิม ยังไม่ได้รัน test set จริง (แค่ `--dev`) รอให้พร้อมจะสรุปผลจริงๆ ค่อยเปิด |
| 3 | `data_cache/` ค้างวันที่ | เหมือน task_a (daily ถึง 2026-08-28, 1h ถึง 2026-09-25) | เดิม | ↔️ คงเดิม — `predict_live.py` ใหม่ดึงสดแยกต่างหากแล้ว (เหมือน task_a/a2) แต่ยังไม่ได้เขียน README ชี้แจงไว้ใน `task_c/data_cache/` เหมือนที่ task_a มี |
| 4 | โมเดลชนะ baseline แค่เฉียดๆ | จากเอกสาร `data_for_present.md` เอง: KBANK ชนะ Majority 3.57pp, ADVANC ชนะ 4.38pp และ Sensitivity ต่ำกว่า Specificity มาก — โมเดลจับ "วันขึ้น" ได้ยาก มักทายลง/นิ่งไว้ก่อน เพื่อนควรรู้ว่าตัวเลขที่ดูดีคือ validation ไม่ใช่ test | เดิม | ↔️ คงเดิม (ยืนยันซ้ำจาก dry-run: P(ขึ้น) ที่ได้อยู่แถว 0.39-0.48 ทั้งคู่ คือโมเดลเอนไปทาง "ลง/นิ่ง" เป็นค่าเริ่มต้นตามที่เอกสารเตือนไว้) |
| 5 | flat day ถูกจัดรวมกับ "ลง" | เป็นข้อจำกัดของ target ที่ผู้เขียนเดิมก็ยอมรับเองในเอกสาร | เดิม | ↔️ คงเดิม — `predict_live.py` ใหม่สืบทอดพฤติกรรมเดิมนี้ตรงๆ (label `"ลง/นิ่ง"` รวมกัน) |
| 6 | ยังไม่ตรวจว่าข้อมูล "วันนี้" จบตลาดแล้วหรือยัง | ทดสอบจริงพบว่า **`predict_live.py` ของ task_a และ task_c (next_day)** ดึง `yf.download(..., end=วันนี้+1)` แล้วถือว่าแถวของ "วันนี้" เป็น data_cutoff ที่สมบูรณ์ทันที โดยไม่เช็คว่าตลาดปิดแล้วหรือยัง (รันทดสอบตอน ~12:26 น. ของ 2026-09-28 ก็ยังได้ data_cutoff = 2026-09-28 ทั้งที่ตลาดไทยปิด ~16:30) ถ้ารันก่อนตลาดปิดจริง ราคาปิดของวันนั้นที่ yfinance คืนมาอาจยังไม่นิ่ง ทำให้ feature/label ของวันนั้นเปลี่ยนได้ถ้ารันซ้ำทีหลัง — **แก้คำอธิบายจากรอบก่อน:** `task_a2/predict_live.py` (t1600) **ไม่เข้าข่ายนี้** เช็คโค้ดจริงแล้วพบว่ามี guard อยู่ (บรรทัด `has_1500_bar`) ปฏิเสธการรันแบบ official ถ้ายังไม่มีแท่ง 15:00 ของวันนั้น (ดูรายละเอียดในตาราง task_a2 ด้านบน) — เป็นแค่ `task_a`/`task_c` (next_day) ที่ไม่มีการเช็คเวลาเลย | ใหม่ | ⚠️ ยังไม่แก้ — เฉพาะ `task_a/predict_live.py` และ `task_c/predict_live.py` (next_day) เท่านั้น |

## ปัญหาระดับ repo

| # | ปัญหา | รายละเอียด | ที่มา | สถานะ |
|---|-------|------------|-------|-------|
| 1 | `.DS_Store` / `__pycache__` เคยหลุดเข้า git | stage การลบไว้ให้แล้ว (`task_c/__pycache__/*.pyc`, `.DS_Store`) รอ commit | เดิม | ↔️ คงเดิม ยังไม่ได้ commit (อยู่ใน staged changes) |
| 2 | Secret/credential | ตรวจแล้วสะอาด ไม่มีหลุดใน code หรือ git history | เดิม | ↔️ คงเดิม |
| 3 | ไม่มี CI/CD | ไม่มี `.github/workflows/` ใดๆ อยู่เลยตอนนี้ | เดิม | 🟡 ใกล้เสร็จ — มี `task_b/automation/github_actions_task_b_daily.yml` เตรียมไว้แล้ว แต่ยังอยู่ใน `task_b/automation/` ไม่ใช่ `.github/workflows/` จริง ต้อง copy เข้าตำแหน่งจริงถึงจะทำงานอัตโนมัติ (หรือใช้ `register_tasks.ps1` ตั้ง Windows Task Scheduler แทนก็ได้ ตามที่ออกแบบไว้เป็นสองทางเลือก) |
| 4 | `data_cache/` ถูก commit เข้า git จริง | เช็คด้วย `git ls-files` เจอ 40 ไฟล์ (task_a ~5.2MB, task_a2 ~216KB, task_c ~844KB) ไม่มีอยู่ใน `.gitignore` เลย — **นี่คือกลไกจริงที่ทำให้ cache ค้างวันที่ในทุก task** เป็น snapshot ที่ freeze ตอน commit ไม่มีอะไรมา refresh อัตโนมัติ เพื่อนต้องเลือก: (ก) ใส่ `data_cache/` ใน `.gitignore` แล้วให้โค้ด regenerate เอง หรือ (ข) เก็บไว้ใน git ต่อแต่มี job/สคริปต์ refresh+commit สม่ำเสมอ ไม่งั้นจะค้างซ้ำอีก | ใหม่ | ✅ แก้แล้วสำหรับ task_a/task_a2 — README ใหม่ชี้แจงว่าแช่แข็งโดยตั้งใจเพื่อ reproducibility ของผลเดิม ส่วนข้อมูลสดแยกไปใช้ `predict_live.py` ต่างหาก (ดูตาราง task_a/task_a2) — **task_c ยังไม่ถูกแตะเลยในรอบนี้ ยังค้างเหมือนเดิม** |
| 5 | ไม่มี test เลยใน 3 tasks | `task_a`, `task_a2`, `task_c` ไม่มีไฟล์ test สักไฟล์ มีแต่ `task_b/tests/` (5 ไฟล์: test_append_investing, test_live_1600, test_live_eval, test_phase1d, test_pipeline) — ถ้าเพื่อนแก้ data_loader/splits/targets/features ใน 3 task ที่เหลือ จะไม่มีอะไรเช็คว่าพังหรือเปล่านอกจากรันมือดูเอง ความเสี่ยง regression เงียบสูงกว่าที่คิด | ใหม่ | ✅ แก้ครบทั้ง 3 tasks แล้ว (2026-09-29) — `task_a` (8 tests), `task_a2` (10 tests, ไฟล์เพื่อน — ดึง `check_has_1500_bar`/`check_time_window` ออกมาเป็นฟังก์ชันแยกก่อนถึง test ได้ ยืนยันแล้วว่า behavior เดิมไม่เปลี่ยน), `task_c` (12 tests รวม regression test ของบั๊ก target ที่เพิ่งแก้) รวม 30 tests ผ่านหมด รันแยกต่อ task ได้ทั้ง `python tests/test_predict_live.py` ตรงๆ และ `pytest task_X/tests/` (รวมหลาย task ในคำสั่งเดียวมีปัญหา pytest module-name ชนกันเพราะทุกไฟล์ชื่อเดียวกัน — รันแยกทีละ task ตามที่ตั้งใจแทน เหมือน `task_b/tests/` เดิม) — **ยังไม่มี**: test ของ `main.py`/`intraday_task2.py`/`main_a2.py` ฝั่ง backtest เอง (มีแค่ฝั่ง live scripts) |
| 6 | root `requirements.txt` ไม่ pin เวอร์ชัน | `numpy>=1.24, pandas>=2.0, scikit-learn>=1.3, xgboost>=2.0, lightgbm>=4.0, yfinance>=0.2.40, matplotlib>=3.7` (ใช้กับ task_a/a2/c) ในขณะที่ `task_b/requirements.txt` pin เป๊ะเพราะเจอปัญหา "default ของ library เปลี่ยนตามเวอร์ชัน → ผลไม่ตรง" มาแล้ว — บทเรียนเดียวกันนี้ยังไม่ถูกเอาไปใช้กับอีก 3 task เลย ถ้าเพื่อน `pip install` ใหม่บนเครื่องตัวเองอาจได้ผลต่างจากที่รายงานไว้แบบไม่รู้ตัว | ใหม่ | ↔️ คงเดิม ยังไม่ pin |

---

## อัปเดต 2026-09-28 — หลัง pull commit `aaae8bd`

เพื่อนแก้เยอะกว่าที่บอกไว้ ไม่ใช่แค่ task_b — commits ที่เข้ามา:
`8fb2fd2` Task A/A2 single source of truth for data_cache, `4f6f1fa` task_b Yahoo connector + automation,
`afe2968` task_b ยกเลิก PHASE1D_LIVE_APPROVAL gate, `0f6435c` แก้ encoding คอมเมนต์

**แก้จริงแล้ว:** task_a #1,#3 · task_a2 #1,#2,#4,#5 · task_b #1,#2,#3 · task_c #1 · repo #4 (เฉพาะ task_a/a2)

**ทีมยืนยันแล้ว:** task_b #2 — การยกเลิกเกทอนุมัติ `PHASE1D_LIVE_APPROVAL.md` ทีมโอเคแล้ว (2026-09-28)
ยังคงย้ำไว้ว่าเรื่องนี้แยกจากปัญหาที่โมเดล t1600 ยังแพ้ baseline Naive อยู่ — เปิด official mode รันได้ ≠ ผลดีพอใช้จริง

**พบปัญหาใหม่ระหว่างทดสอบ:** task_c #6 (ใหม่) — `predict_live.py` ของทั้ง task_a, task_a2, task_c ไม่เช็คว่าตลาด
ปิดแล้วหรือยังก่อนถือว่าข้อมูล "วันนี้" นิ่งสมบูรณ์ (ต่างจาก `task_b/predict_1600.py` ที่มี guard 16:00–16:30)
ยังไม่ได้แก้

**ยังไม่ถูกแตะเลย:** repo #1 (.DS_Store ยัง stage รอ commit) · repo #5-6 บางส่วน (task_a/a2/c ยังไม่มี test,
requirements root ยังไม่ pin) · task_b #4-5 (ยังเหมือนเดิม) · target ของ Task 2 ยังไม่เคยตรวจว่า close 16:00
ตรงกับราคาปิดทางการ (ATC) จริงไหม

---

## อัปเดต 2026-09-28 (ค่ำ) — task_c/predict_live_1600.py

เขียน **`task_c/predict_live_1600.py`** เสร็จแล้ว (mirror จาก `task_a2/predict_live.py`) — ทดสอบ `--dry-run`
ผ่านจริง: เทรนบนประวัติ 719 วัน เลือก ANN (MLP) ด้วย walk-forward OOF (KBANK acc 0.706, ADVANC acc 0.648 — ใกล้
เคียงกับตัวเลข validation ที่เพื่อน commit ไว้ 0.697/0.667 พอสมควร เป็นสัญญาณที่ดี) ระหว่างเขียนเจอบั๊ก tz-aware
vs tz-naive datetime เปรียบเทียบกันไม่ได้ (แก้แล้วในไฟล์นี้)

**ตอนนี้ทุก task มี live-predict script ครบทั้ง 2 กลุ่มแล้ว:**

| กลุ่ม | task_a | task_a2 | task_b | task_c |
|---|---|---|---|---|
| next_day | ✅ `predict_live.py` | (ไม่มีโหมดนี้) | ✅ `daily_next_day.py` | ✅ `predict_live.py` |
| t1600 | (ไม่มีโหมดนี้) | ✅ `predict_live.py` | ✅ `predict_1600.py` | ✅ `predict_live_1600.py` (ใหม่) |

**สิ่งที่ยังต่างกันระหว่าง t1600 ของแต่ละ task** (รู้ไว้ก่อนทำ orchestrator): `task_b/predict_1600.py` เข้มงวด
สุด (เช็คช่วงเวลา 16:00–16:30 ทั้งขอบบนขอบล่าง + ต้องมี live snapshot ไฟล์ที่ hash ไว้ + เช็ค feature ตรงกับ
historical) ส่วน `task_a2` และ `task_c` ใหม่เช็คแค่ขอบล่าง (มีแท่ง cutoff หรือยัง) ไม่มีขอบบนของเวลา และดึงจาก
yfinance สดตอนรันเลย ไม่มีการเซฟ snapshot เป็นหลักฐาน — ถ้าจะทำ orchestrator ควรตัดสินใจว่าจะปล่อยให้ต่างกันแบบนี้
หรือยกระดับ task_a2/task_c ให้เข้มเท่า task_b

---

**สรุปสั้น (อัปเดต):** ครบทั้ง 4 tasks x 2 กลุ่มแล้ว (next_day ที่มีโหมดนี้ 3 tasks, t1600 ที่มีโหมดนี้ 3 tasks)
ทุกตัวทดสอบ dry-run ผ่านจริง — สิ่งที่เหลือก่อนเชื่อมเป็นระบบเดียว: (1) next_day scripts (task_a/task_c) ยังไม่มี
time guard กันรันก่อนตลาดปิด (2) ระดับความเข้มงวดของ t1600 guard ต่างกันมากระหว่าง task_b กับ task_a2/task_c (3)
ยังไม่มี orchestrator ที่เรียกทุก script ตามเวลา (4) task_b pin เวอร์ชัน library ต่างจาก task อื่น อาจต้องแยก
environment ตอนรันจริง

ก่อนแก้อะไรต่อ แนะนำลำดับนี้: (1) คุยกับเพื่อนเรื่อง task_b #2 ให้ชัดว่ายอมรับการยกเลิกเกทได้หรือไม่ — เสร็จแล้ว
(2) สร้าง orchestrator 2 กลุ่ม (next_day + t1600) — เสร็จแล้ว (ดูด้านล่าง) (3) เพิ่ม time guard ให้ next_day
scripts ที่ยังไม่มี (4) เพิ่ม test ให้ `predict_live*.py` ทุกตัวที่เพิ่งเขียน และ pin เวอร์ชันใน root
requirements.txt ตามที่ task_b ทำไว้เป็นตัวอย่าง

---

## อัปเดต 2026-09-28 (ดึก) — สร้าง orchestrator ทั้ง 2 กลุ่มแล้ว

**`automation/run_1600.py`** (task_a2 → task_b → task_c) — ทดสอบ `--dry-run` ผ่านครบทั้ง 3 tasks จริง
เรียงลำดับถูกต้อง, ดึง live snapshot ของ task_b สำเร็จ (`tools/fetch_yahoo_intraday.py`), เขียน log ของทุก
task ได้ครบ

**`automation/run_next_day.py`** (task_a → task_c → task_b) — โค้ด/logic ถูกต้อง แต่ **รันจริงไม่ได้ตอนนี้**:
`task_b/set_holidays.txt` มี `confirmed_through: 2026-09-25` ซึ่งเก่ากว่าวันนี้ (2026-09-28) พอคำนวณวันทำการ
ถัดไป (2026-09-29) ก็เกิน `confirmed_through` — guard (ที่ยืมมาจาก `daily_next_day.py` เป๊ะ) เลยปฏิเสธไม่ให้รัน
ต่อ **ไม่ใช่บั๊กของ orchestrator** เป็นเพราะไฟล์วันหยุดไม่ได้อัปเดต — **เท่ากับว่า `task_b/daily_next_day.py`
เองก็รันไม่ได้ตอนนี้เช่นกัน** ด้วยเหตุผลเดียวกัน ต้องให้คนเช็คประกาศวันหยุด SET ทางการแล้วเลื่อน
`confirmed_through` ก่อน — ไม่ได้แก้ไฟล์นี้เองเพราะต้องอ้างอิงประกาศทางการ ห้ามเดา

**หมายเหตุการ commit**: task_a, task_a2, task_c ไม่ commit log ให้ตัวเอง (แค่พิมพ์เตือน) และ
`task_b/predict_1600.py` ก็ไม่ commit ให้ตัวเองเหมือนกัน (ต่างจาก `daily_next_day.py` ที่ commit เอง) —
orchestrator ทั้งสองไฟล์จึงรับหน้าที่ `git add/commit/push` แทนให้ทุก task ที่ไม่ทำเอง ผ่าน flag
`--commit`/`--push` (ยังไม่เคยรันจริงด้วย `--commit` เลยสักครั้ง ทดสอบแค่ `--dry-run`)

---

## อัปเดต 2026-09-29 (ต่อ) — เพิ่ม guard เวลา t1600 ให้ครบ + เพิ่ม code_commit tracking

**เพิ่ม guard 16:00–16:30 ให้ `task_a2/predict_live.py` และ `task_c/predict_live_1600.py`** (เดิมเช็คแค่ขอบล่าง
ไม่มีขอบบน) ให้เข้มเท่า `task_b/predict_1600.py` แล้ว — **หมายเหตุ: แก้ `task_a2/predict_live.py` ด้วย ทั้งที่เป็น
ไฟล์ของเพื่อน (สั่งให้แก้ต่อโดยไม่ต้องถามซ้ำ) ควรแจ้งเพื่อนว่าไฟล์นี้ถูกเพิ่มความเข้มงวดแล้ว** — ทดสอบยืนยันว่า guard
ทำงานถูกต้องและ `--dry-run` ยังบายพาสได้ปกติ — repo #5 (ความเข้มงวด t1600 ไม่เท่ากัน) ตอนนี้แก้ไปแล้วบางส่วน (เรื่อง
เวลา) เหลือแค่เรื่อง snapshot evidence ที่ task_a2/task_c ยังไม่เก็บเป็นไฟล์เหมือน task_b

**เพิ่ม `code_commit`/`worktree_dirty` เข้า LOG_COLUMNS ของ `task_a/predict_live.py`, `task_a2/predict_live.py`,
`task_c/predict_live.py`, `task_c/predict_live_1600.py`** (task_b มีอยู่แล้ว) — แก้ repo #11 (ไม่มีทาง track ว่า
prediction แถวไหนมาจากโค้ดเวอร์ชันไหน) ที่ค้างมาจากรอบคุยเรื่องนโยบายปรับจูนโมเดล — migrate dryrun log CSV เดิม 4
ไฟล์ให้ตรง schema ใหม่แล้ว (แถวเก่าใส่ค่าว่างเพราะย้อนไปหาไม่ได้จริง) ทดสอบผ่านครบทั้ง 4 สคริปต์

---

## อัปเดต 2026-09-29 — แก้ set_holidays.txt แล้ว (confirmed_through: 2026-12-31 พร้อมวันหยุด ต.ค.-ธ.ค. จาก
set.or.th) commit แยกเป็น 2 ก้อนสะอาด แล้วลองรัน `run_next_day.py --dry-run` จริง เจอ 2 เรื่องใหม่ที่สำคัญ:

**1) `data_cutoff` ของ task_a/task_c เพี้ยนไปเป็น 2026-09-25 ทั้งที่ควรจะเป็น 2026-09-28** — ตรวจสาเหตุจริงแล้ว
พบว่า **yfinance คืนแถวของ 2026-09-28 มา แต่ `Close` เป็น `NaN`** (Open/High/Low/Volume มีค่าปกติ) ทั้งที่ตลาดปิด
ไปหลายชั่วโมงแล้ว (yfinance ยังไม่ backfill ราคาปิดให้ทัน) — `data_loader.clean()` มี `dropna(subset=["Close"])`
เลยตัดแถวนี้ทิ้งไปเงียบๆ ถอยไปใช้ 2026-09-25 แทนโดยไม่มีอะไรเตือน **เพราะ `fetch_live()` เรียก
`clean(df, verbose=False)` ปิดเสียงคำเตือนไว้** — เป็นบั๊กจริงที่กระทบ `task_a`, `task_a2`, `task_c`
`predict_live*.py` ทุกตัว (ใช้ `fetch_live()`/pattern เดียวกันหมด) เสี่ยงทำนายจากข้อมูลเก่ากว่าที่คิดโดยไม่รู้ตัว
แนะนำ: อย่างน้อยเปิด `verbose=True` เฉพาะตอนตัดแถวเพราะ NaN Close ในข้อมูลไม่กี่แถวล่าสุด หรือพิมพ์คำเตือนแยก
เพื่อให้เห็นว่า data_cutoff ไม่ใช่ "เมื่อวาน/วันนี้" ตามที่ควรจะเป็น — **ยังไม่ได้แก้**

**2) `task_b` ล้มที่ขั้น `check_env.py`** — เวอร์ชัน library ที่ลงจริงในเครื่อง (numpy 2.5.1, pandas 3.0.5,
scikit-learn 1.9.0) ไม่ตรงกับที่ `task_b/requirements.txt` pin ไว้ (2.1.1 / 2.3.3 / 1.9.1) — **นี่คือ guard
ทำงานถูกต้องตามที่ออกแบบไว้** ไม่ใช่บั๊ก ยืนยันสิ่งที่เคยเตือนไว้ก่อนหน้านี้ว่า task_b ต้องมี venv แยกจาก task
อื่นจริงๆ ถ้าจะรัน orchestrator ให้ครบทุก task ต้องจัดการเรื่อง environment ก่อน (`pip install -r
task_b/requirements.txt` ในสภาพแวดล้อมที่ใช้รันขั้น task_b) — **ยังไม่ได้แก้**

สรุป: `run_next_day.py` เดินไปได้ถึง task_a และ task_c สำเร็จ (แต่ได้ค่าจาก data_cutoff เก่า) แล้วหยุดที่ task_b
เพราะ environment ไม่ตรง — ยังไม่เคยรันจบครบ end-to-end จริงสักครั้ง

---

## อัปเดต 2026-09-29 (เช้ามืด) — Yahoo backfill มาแล้ว + เจอและแก้บั๊ก non-determinism ของ task_b + รันจบครบครั้งแรก

**Yahoo backfill ราคาปิด 28 ก.ย. มาแล้ว** (ก่อนหน้านี้ค้าง NaN นานหลายชั่วโมง) — ลอง `run_next_day.py --dry-run`
ใหม่ เดินผ่าน task_a/task_c ได้ปกติ แต่ไปสะดุดที่ **regression-check guard ของ task_b เอง**
(`daily_next_day.py`: เทียบ `results/*_val.csv` ก่อน/หลัง append ต้องเหมือนเป๊ะ) รายงานว่าไฟล์เปลี่ยนทั้งที่ข้อมูล
ที่เพิ่มใหม่ไม่ได้อยู่ใน val window เลย

**สืบแล้วพบว่าไม่ใช่บั๊กจากข้อมูล** — `RF_PARAMS`/`XGB_PARAMS` ใน `config.py` ตั้ง `n_jobs=-1` (เทรนขนาน) ซึ่ง
`models.py` มีคอมเมนต์เดิมเตือนไว้แล้วว่าทำให้ผลไม่ byte-identical แต่ประเมินขนาด diff ไว้ต่ำเกินไป (~1e-18) —
ทดสอบจริงรัน `main.py --dev` ซ้ำ 5 ครั้งด้วยข้อมูลเดียวกัน เจอ diff 2 ใน 4 ครั้ง (Random Forest) และตอน
เทียบก่อน/หลัง append เจอ diff ของ XGBoost ที่ ~1e-7 (ใหญ่กว่าที่คอมเมนต์เดิมคาดไว้มาก) — **แก้แล้ว**: เปลี่ยน
`n_jobs=-1` → `n_jobs=1` ทั้งสองที่ ทดสอบรัน `main.py --dev` ซ้ำ 4 ครั้งหลังแก้ **byte-identical ทุกครั้ง** ผลสรุป
ว่าโมเดลไหนชนะต่อหุ้นไม่เปลี่ยน (แค่ทศนิยมท้ายขยับครั้งเดียวจากการเปลี่ยนโหมด threading ไม่ใช่ความไม่นิ่งต่อเนื่อง)
รัน `task_b/tests/` ครบ 83 tests ผ่านหมด (แยกพิสูจน์แล้วว่า 6 ตัวที่ fail ตอนแรกมาจาก raw_data ที่ยังไม่ commit
จากการทดสอบก่อนหน้า ไม่เกี่ยวกับ n_jobs)

**ผลลัพธ์: `run_next_day.py --dry-run` รันจบครบทั้ง 3 tasks เป็นครั้งแรก** (task_a → task_c → task_b ผ่านหมดทุกขั้น
รวม regression-check) ทำนายราคาปิด 29 ก.ย. ออกมาสำเร็จ

**⚠️ ยังไม่ตัดสินใจ**: ตอนรันจริง `task_b/raw_data/{KBANK,ADVANC}_10Y_Cleaned.csv` ถูก append ข้อมูล 28 ก.ย. จริง
ในเครื่อง (เป็น side effect ที่เกิดขึ้นเสมอไม่ว่าจะใส่ `--dry-run` หรือไม่ เพราะขั้น fetch/append ของ
`daily_next_day.py` ไม่ได้ผูกกับ flag นี้ มีแค่ขั้นทำนาย+commit ท้ายสุดที่เป็น dry จริง) — ข้อมูลถูกต้อง
(cross-check OHLC ผ่าน, SHA256 อัปเดตถูก) แต่**ยังไม่ commit** เพราะการ commit ข้อมูลตลาดใหม่ควรเป็นการรัน
official จริง (`--commit`) ไม่ใช่ผลพลอยได้จากการ debug — รอให้คุณ/เพื่อนตัดสินใจว่าจะ commit เลยหรือรอรัน
official แยกทีหลัง

---

## อัปเดต 2026-09-29 (ต่อ) — แก้ run_1600.py ให้เรียก daily_1600.py ของ task_b แทนการเรียกสคริปต์ย่อยตรงๆ

ตอนไปดู `task_b/automation/register_tasks.ps1` เจอว่า task_b มี **`daily_1600.py`** เป็น wrapper ทางการของ t1600
เอง (fetch -> predict -> `tools/check_snapshot_1600.py` availability check -> commit) ที่ `run_1600.py`
เดิมไม่ได้เรียกเลย — เรียก `tools/fetch_yahoo_intraday.py` + `predict_1600.py` ตรงๆ แทน ทำให้**ข้าม
availability check ไปเงียบๆ** และมี commit logic ที่ซ้ำซ้อนกับของ `daily_1600.py` เอง **แก้แล้ว**: เปลี่ยนไป
เรียก `daily_1600.py --phase predict` (ส่ง `--official --commit --push` ต่อเมื่อสั่ง `--commit`) แบบเดียวกับที่
`run_next_day.py` เรียก `daily_next_day.py` อยู่แล้ว — ทดสอบเรียก `daily_1600.py --phase predict` ตรงๆ ยืนยันว่า
syntax/flow ถูกต้อง (fail แค่เพราะ guard เวลา 16:00 ของ task_b เอง ไม่ใช่บั๊กจากการเชื่อม)

**หมายเหตุ**: `daily_1600.py` ยังมี **`--phase outcome`** (บันทึกผลจริงหลัง 17:00 เทียบกับที่ทำนายไว้ตอน 16:00)
ที่ตั้งใจไม่รวมเข้า `run_1600.py` เพราะเป็นคนละช่วงเวลา ต้องมี schedule แยกต่างหาก (ดู
`register_tasks.ps1` ของ task_b เป็นตัวอย่าง) — **task_a2 และ task_c ไม่มี phase "outcome" แบบนี้เลย** ยังไม่มี
ใครบันทึกผลจริงของ t1600 เทียบกับที่ทำนายไว้สำหรับสองตัวนี้ (เพิ่มเป็นรายการใหม่ที่ยังไม่แก้)

---

## อัปเดต 2026-09-29 (ต่อ) — สร้าง scheduler template ให้ orchestrator ทั้ง 2 ตัวแล้ว

**`automation/github_actions_next_day.yml`** — สำหรับกลุ่ม next_day (ไม่เร่งเวลามาก ทน cron ของ GitHub ช้าได้)
mirror โครงสร้างจาก `task_b/automation/github_actions_task_b_daily.yml` เดิม แต่ครอบคลุมทั้ง 3 tasks ผ่าน
`automation/run_next_day.py`

**`automation/register_1600.ps1`** — สำหรับกลุ่ม t1600 (ใช้ Windows Task Scheduler แทน GitHub Actions เพราะ
เส้นตาย 16:00-16:30 แคบเกินไป เหตุผลเดียวกับที่ `task_b/automation/register_tasks.ps1` ใช้อยู่แล้ว) ครอบคลุม
ทั้ง 3 tasks ผ่าน `automation/run_1600.py`

ทั้งคู่เป็น **template เท่านั้น ยังไม่ได้ติดตั้งจริง** (ต้อง copy `.yml` ไปที่ `.github/workflows/` เอง หรือรัน
`.ps1` ด้วยมือเพื่อลงทะเบียน Task Scheduler) — ตรวจ syntax แล้วผ่านทั้งคู่ (YAML parse ได้, PowerShell parse
ไม่มี error) และ tests ที่ workflow อ้างถึงผ่านหมด 5/5 (ระหว่างเช็คเจอว่า `task_b/tests/test_fetch_yahoo_daily.py`
และ `test_append_investing.py` fail ชั่วคราวเพราะ raw_data ที่ยังไม่ commit จากการทดสอบก่อนหน้า -- ยืนยันแล้วว่า
ไม่ใช่บั๊กใหม่ เป็นเรื่องเดียวกับที่ค้างตัดสินใจอยู่เรื่อง "จะ commit ข้อมูล 28 ก.ย. ไหม")
