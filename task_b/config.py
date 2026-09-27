"""
config.py — งาน B (ราคาปิด / return)
====================================
ไฟล์ตั้งค่าของงาน B เท่านั้น แก้ที่นี่ไม่กระทบ task_a / task_c
"""

from pathlib import Path

# ---------------------------------------------------------------
# 0) Path — anchor จากโฟลเดอร์ของไฟล์นี้ (E2)
# ---------------------------------------------------------------
# ทำให้ `python task_b/main.py` กับ `cd task_b && python main.py`
# อ่าน/เขียนไฟล์ที่เดียวกันเสมอ
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
OUTPUT_DIR = BASE_DIR / "results"
LOCK_PATH = BASE_DIR / "PRE_TEST_LOCK.md"


# ---------------------------------------------------------------
# 1) ข้อมูลหุ้น (A1)
# ---------------------------------------------------------------
TICKERS = ["KBANK.BK", "ADVANC.BK"]

# ข้อมูลมาจาก investing.com (raw_data/) เท่านั้น ไม่มีโหมด fallback
# ถ้าไฟล์หายต้อง error ทันที ไม่ใช่ได้ข้อมูลจากแหล่งอื่นมาแทนโดยไม่รู้ตัว
DATA_SOURCE = "investing"
RAW_DATA_SUFFIX = "_10Y_Cleaned.csv"


# ---------------------------------------------------------------
# 2) การแบ่งข้อมูล — freeze ด้วยวันที่ (0.1) ห้าม shuffle
# ---------------------------------------------------------------
# วันที่ชุดนี้ทำให้ train/val/test มีขนาดเท่าเวอร์ชันก่อนหน้า (1,688 / 362 / 362)
# จึงเทียบผลได้ว่าอะไรเปลี่ยนเพราะการแก้บั๊ก ไม่ใช่เพราะ split เลื่อน
# (ถ้าจะเปลี่ยนวัน ต้องเขียนเหตุผลไว้ตรงนี้)
SPLIT_BY_DATE = {
    "train_end": "2023-09-01",   # train: 2016-09-26 -> 2023-09-01   1,688 วัน
    "val_end":   "2025-02-25",   # val:   2023-09-04 -> 2025-02-25     362 วัน
}                                # test:  2025-02-26 -> 2026-08-28     362 วัน
TRAIN_RATIO = None               # ไม่ใช้แล้ว -- boundary มาจากวันที่เท่านั้น
VAL_RATIO = None


# ---------------------------------------------------------------
# 3) Feature engineering — เฉพาะงาน B
# ---------------------------------------------------------------
LAG_DAYS = [1, 2, 3, 5, 10]        # ย้อนหลังกี่วัน
MA_WINDOWS = [5, 10, 20]           # เส้นค่าเฉลี่ย
VOL_WINDOWS = [5, 20]              # ความผันผวน
RSI_PERIOD = 14

# ใส่ feature วันในสัปดาห์ไหม
USE_DAY_OF_WEEK = True

# งาน B ไม่ใช้ feature parity ย้อนหลัง (เฉพาะงาน A เท่านั้น)


# ---------------------------------------------------------------
# 4) โมเดล (C2)
# ---------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------
# *** ทั้ง 3 โมเดลด้านล่าง "ไม่ได้จูนบน validation" เหมือนกันหมด ***
# เป็นค่า conservative ที่เลือกจากหลักการ: ข้อมูล train 1,688 แถว สัญญาณอ่อนมาก
# จึงเลือกโมเดลความจุต่ำ + regularization จริงจัง เพื่อไม่ให้จำ noise
#
# การจูนจริงอยู่ในเฟส 2 ซึ่งจะทำกับทั้ง 3 โมเดลภายใต้ walk-forward + protocol
# เดียวกัน -- ไม่ใช่ทำเฉพาะบางตัวแล้วเทียบกัน
# ---------------------------------------------------------------

# ค่าเดิมของ v1 คือ (64,32) alpha=1e-3 ซึ่ง overfit หนัก
# (วัดใน rebuild นี้ ก่อนเปลี่ยนค่า: val R2_return = -2.39 KBANK / -2.37 ADVANC)
# จึงลดความจุและเพิ่ม L2:
#   (16,8) บน 29 features = ราว 625 พารามิเตอร์ ต่อข้อมูล 1,688 แถว
#   = ราว 2.7 แถวต่อพารามิเตอร์ ซึ่งยังตึง -> ต้องมี L2 ที่มีน้ำหนักจริง
#
#   alpha = 1.0 เป็นค่า L2 ที่กำหนดแบบ conservative ให้แรงกว่า default ของ sklearn
#   (1e-4) อย่างชัดเจน เพื่อจำกัดความซับซ้อนของ ANN ในเฟส 1
#   *** ไม่ได้เลือกจากผล validation -- ค่าที่เหมาะสมจริงจะค้นหาในเฟส 2 ***
ANN_PARAMS = {
    "hidden_layer_sizes": (16, 8),
    "activation": "relu",
    "alpha": 1.0,
    "solver": "adam",
    "learning_rate_init": 1e-3,
    # เดิม 500 -- ANN ชน ceiling ทั้ง 20/20 fit (10 seeds x 2 หุ้น) โดย loss
    # ยังลดต่ออีก 0.1-1.8% ทุก 50 iterations แปลว่า optimizer ยังเดินอยู่จริง
    # ทดสอบด้วย ceiling 3000 แล้วทุก seed หยุดเองที่ 502-869 iterations
    # => 500 เป็น ceiling ที่ต่ำเกินไป ไม่ใช่ว่าโมเดล converge แล้ว
    #
    # การแก้นี้ *ไม่ใช่* การจูนบน validation -- เหตุผลมาจาก training convergence
    # ล้วนๆ (ConvergenceWarning + loss curve) ไม่ได้ดูค่า MAE ของ val เลย
    "max_iter": 3000,
    # ปิด early_stopping: MLPRegressor จะ shuffle แบ่ง val ของตัวเองออกจาก train
    # ซึ่งขัดกับหลัก "ห้าม shuffle" ของ time series
    "early_stopping": False,
    "n_iter_no_change": 20,
    "random_state": RANDOM_STATE,   # ถูกแทนด้วยแต่ละค่าใน ANN_SEEDS ตอนเทรนจริง
}

# seed averaging ไม่ใช่การจูน -- เป็นส่วนหนึ่งของ "นิยามโมเดล"
# ผลของ MLP ขึ้นกับการสุ่ม weight เริ่มต้น การรายงาน seed เดียว = เลือกผลที่ถูกใจได้
ANN_SEEDS = list(range(10))

# ต้นไม้ลึกจำ noise ได้ง่าย -> จำกัดความลึกและบังคับให้ leaf ใหญ่พอ
# (ไม่ใช่ค่า default ของ sklearn ซึ่งเป็น max_depth=None, min_samples_leaf=1
#  ซึ่งปล่อยให้ต้นไม้โตจนจำ training set ได้หมด)
RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# depth ตื้น + lr ต่ำ + subsample = ค่ามาตรฐานสาย conservative สำหรับ tabular ที่ noise สูง
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ---------------------------------------------------------------
# 5) การตรวจสอบ Data Leakage
# ---------------------------------------------------------------
# ถ้า feature ตัวไหนมี |correlation| กับ target return เกินค่านี้
# แปลว่าน่าจะ leak -> โปรแกรมจะเตือน
LEAK_CORR_THRESHOLD = 0.50


# ---------------------------------------------------------------
# 6) Lock ก่อนเปิด test (0.5)
# ---------------------------------------------------------------
# PRE_TEST_LOCK.md ต้องมีทุกหัวข้อนี้และมีค่าหลัง ":" จริง
# *** ห้ามสร้าง PRE_TEST_LOCK.md ตลอดเฟส 1 ***
REQUIRED_LOCK_FIELDS = [
    "DATASET:", "TRAIN_END:", "VAL_END:", "FEATURE_SET:",
    "MODELS:", "HYPERPARAMS:", "PRIMARY_METRIC:",
    "BASELINES:", "EXPECTED:", "LOCK_DATE:",
]



# ---------------------------------------------------------------
# 7) ป้ายกำกับชุดค่าโมเดลที่ใช้อยู่ -- บันทึกลง prediction_log ทุกแถว (§1.5)
# ต้องเปลี่ยนค่านี้ทุกครั้งที่แก้ ANN_PARAMS / RF_PARAMS / XGB_PARAMS
# หรือ RECENCY_HALF_LIFE เพื่อให้แยกได้ว่าแถวไหนมาจากโมเดลชุดไหน
# ---------------------------------------------------------------
CONFIG_TAG = "phase1b-untuned"

if not CONFIG_TAG.strip():
    raise ValueError("config.CONFIG_TAG ต้องไม่ว่าง")


# ---------------------------------------------------------------
# 8) Recency weighting (feedback ข้อ 7) -- §3
# ---------------------------------------------------------------
# หน่วย = จำนวนแถว (วันทำการ) ที่น้ำหนักลดลงครึ่งหนึ่ง
# None = ไม่ถ่วงน้ำหนัก (พฤติกรรมเดิมทุกอย่าง)
#
# เป็น dict "ต่อโมเดล" เพราะ §4 stage 1 เลือก half-life แยกแต่ละโมเดล
# (ใช้ค่าเดียวกันทั้ง KBANK และ ADVANC -- เหมือน ANN_PARAMS/RF_PARAMS/XGB_PARAMS)
# key ต้องตรงกับชื่อใน models.get_regressors() ทุกตัวอักษร
# ค่าทั้งหมดถูกเลือกด้วย §4 stage 1 -- ตอนนี้ยังเป็น None ทั้งหมด
RECENCY_HALF_LIFE = {
    "ANN (MLP)":     None,
    "Random Forest": None,
    "XGBoost":       None,
}
