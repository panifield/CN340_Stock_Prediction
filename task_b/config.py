"""
config.py — งาน B (ราคาปิด / return)
====================================
ไฟล์ตั้งค่าของงาน B เท่านั้น แก้ที่นี่ไม่กระทบ task_a / task_c
"""

# ---------------------------------------------------------------
# 1) ข้อมูลหุ้น
# ---------------------------------------------------------------
TICKERS = ["KBANK.BK", "ADVANC.BK"]

START_DATE = "2016-08-26"
END_DATE = "2026-08-28"

# ถ้าโหลด yfinance ไม่ได้ (เน็ตมีปัญหา / รันออฟไลน์)
# ตั้งเป็น True เพื่อใช้ข้อมูลจำลองทดสอบว่าโค้ดรันผ่านไหม
# *** ห้ามใช้ข้อมูลจำลองในรายงานเด็ดขาด ***
USE_SYNTHETIC_DATA = False

# โฟลเดอร์เก็บไฟล์ csv ที่โหลดมาแล้ว (จะได้ไม่ต้องโหลดซ้ำ)
CACHE_DIR = "data_cache"


# ---------------------------------------------------------------
# 2) การแบ่งข้อมูล (ต้องแบ่งตามเวลา ห้าม shuffle)
# ---------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# ที่เหลือเป็น test = 0.15

# ถ้าอยากกำหนดวันเองแทนการใช้สัดส่วน ให้ใส่วันที่ตรงนี้
# (ถ้าเป็น None จะใช้สัดส่วนด้านบน)
SPLIT_BY_DATE = None
# ตัวอย่าง:
# SPLIT_BY_DATE = {"train_end": "2022-12-31", "val_end": "2023-12-31"}


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
# 4) โมเดล
# ---------------------------------------------------------------
RANDOM_STATE = 42

ANN_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "alpha": 1e-3,              # L2 regularization
    "learning_rate_init": 1e-3,
    "max_iter": 500,
    # ปิด early_stopping: เรามี validation set ที่แบ่งตามเวลาเองอยู่แล้ว
    # (splits.py) ถ้าเปิดไว้ MLPRegressor จะสุ่ม shuffle
    # แบ่ง validation ของตัวเองออกจาก train อีกชุด ซึ่งขัดกับหลัก
    # "ห้าม shuffle" ของข้อมูล time series ที่ทั้งโปรเจกต์นี้ยึดถือ
    "early_stopping": False,
    "n_iter_no_change": 20,     # เช็ค convergence จาก training loss เอง
    "random_state": RANDOM_STATE,
}

RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 8,
    "min_samples_leaf": 20,     # กันไม่ให้ overfit noise
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
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
# 6) Output
# ---------------------------------------------------------------
OUTPUT_DIR = "results"
SAVE_PLOTS = True
