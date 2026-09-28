"""
config.py — งาน C (ขึ้น/ลง)
===========================
ไฟล์ตั้งค่าของงาน C เท่านั้น แก้ที่นี่ไม่กระทบ task_a / task_b
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
# 3) Feature engineering — เฉพาะงาน C
# ---------------------------------------------------------------
LAG_DAYS = [1, 2, 3, 5, 10]        # ย้อนหลังกี่วัน
# เก็บเฉพาะเส้นสั้นและกลาง เพื่อลด feature ที่สัมพันธ์กันสูงเกินไป
MA_WINDOWS = [5, 20]
VOL_WINDOWS = [5, 20]              # ความผันผวน
RSI_PERIOD = 14
ATR_PERIOD = 14

# Winsorize เฉพาะ signal ที่เป็น return/การเปลี่ยนแปลง โดยใช้ quantile
# แบบ expanding (คำนวณจากอดีตถึงวันนั้นเท่านั้น) เพื่อไม่มองข้อมูลอนาคต
WINSOR_LOWER_QUANTILE = 0.01
WINSOR_UPPER_QUANTILE = 0.99
WINSOR_MIN_PERIODS = 60

# ใส่ feature วันในสัปดาห์ไหม
USE_DAY_OF_WEEK = True

# งาน C ไม่ใช้ feature parity ย้อนหลัง (เฉพาะงาน A เท่านั้น)

# ---------------------------------------------------------------
# 3.1) Feature selection experiment
# ---------------------------------------------------------------
# "none" คือชุดสำรอง/ค่าเดิม: ใช้ครบ 33 features โดยไม่ตัดอะไรออก
# วิธีอื่นถูก fit จาก train set เท่านั้น แล้วจึงนำชื่อ feature ไปใช้กับ val/test
FEATURE_SELECTION_METHOD = "none"     # "none", "l1", หรือ "top_k"
TOP_K_FEATURES = 20                   # ใช้เมื่อ method = "top_k"
L1_SELECTOR_C = 0.05                  # C ต่ำ -> L1 บีบ coefficient เป็น 0 มากขึ้น
L1_SELECTOR_MAX_ITER = 5000


# ---------------------------------------------------------------
# 4) โมเดล
# ---------------------------------------------------------------
RANDOM_STATE = 42

ANN_PARAMS = {
    # ลดขนาดเครือข่ายเพื่อลดการจำ noise ของชุด train
    "hidden_layer_sizes": (16, 8),
    "activation": "relu",
    "alpha": 1e-2,              # เพิ่ม L2 regularization เพื่อลด overfitting
    "learning_rate_init": 5e-4,
    "max_iter": 150,
    # ปิด early_stopping: เรามี validation set ที่แบ่งตามเวลาเองอยู่แล้ว
    # (splits.py) ถ้าเปิดไว้ MLPClassifier จะสุ่ม shuffle
    # แบ่ง validation ของตัวเองออกจาก train อีกชุด ซึ่งขัดกับหลัก
    # "ห้าม shuffle" ของข้อมูล time series ที่ทั้งโปรเจกต์นี้ยึดถือ
    "early_stopping": False,
    "n_iter_no_change": 20,     # เช็ค convergence จาก training loss เอง
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

LOGREG_PARAMS = {
    "penalty": "l2",
    "solver": "lbfgs",
    "C": 1.0,
    "max_iter": 2000,
    "random_state": RANDOM_STATE,
}


# ---------------------------------------------------------------
# 5) การตรวจสอบ Data Leakage
# ---------------------------------------------------------------
# ถ้า feature ตัวไหนมี |correlation| กับ target ขึ้น/ลง เกินค่านี้
# แปลว่าน่าจะ leak -> โปรแกรมจะเตือน
LEAK_CORR_THRESHOLD = 0.50

# ถ้า accuracy งานขึ้น/ลง สูงเกินค่านี้ ให้สงสัยไว้ก่อนว่า leak
SUSPICIOUS_ACCURACY = 0.65


# ---------------------------------------------------------------
# 6) Output
# ---------------------------------------------------------------
OUTPUT_DIR = "results"
SAVE_PLOTS = True


# ---------------------------------------------------------------
# 7) Task 2 — Intraday: ทำนายทิศทางปิดตลาดจากข้อมูลระหว่างวัน
# ---------------------------------------------------------------
# ใช้แท่ง 1 ชั่วโมงล่าสุดที่มี timestamp ไม่เกินเวลานี้ แล้วทำนาย Close 16:00
# หากวันใดไม่มีแท่ง 15:00 (เช่นโครงสร้าง session ต่างกัน) จะ fallback เป็น
# แท่งก่อนหน้า เพื่อไม่ใช้ข้อมูลอนาคต
INTRADAY_CUTOFF_HOUR = 15
INTRADAY_CLOSE_HOUR = 16
