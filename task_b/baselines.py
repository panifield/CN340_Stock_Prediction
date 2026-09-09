"""
baselines.py — งาน B (ราคาปิด / return)
=======================================
Baseline ของงาน B  <-- ส่วนที่สำคัญที่สุดของโปรเจกต์นี้

ถ้ารายงานบอกว่า "XGBoost ได้ MAE 1.08 ดีที่สุด"
แต่ไม่บอกว่า naive baseline ได้ 0.95
อาจารย์ถามคำถามเดียวก็จบ

Baseline ที่ต้องมี:
  - naive        : ทำนายว่าพรุ่งนี้ = วันนี้ (return = 0)  <-- คู่แข่งตัวจริง
  - mean return  : ทำนายด้วยค่าเฉลี่ย return ของ train
  - always up    : ทำนายว่าขึ้นทุกวัน  <-- baseline ด้าน "ทิศทาง"

*** ทำไมต้องมี always up ทั้งที่มี naive อยู่แล้ว ? ***
naive ทำนาย return = 0 ซึ่ง "ไม่ได้ให้สัญญาณทิศทาง" เลย ค่า DirAcc ของมัน
จึงเป็น NaN (ดู evaluate.directional_accuracy) พอไม่มีตัวเทียบด้านทิศทาง
เราจะไม่รู้ว่า DirAcc ของโมเดลที่ได้มานั้นดีกว่าการทายมั่ว ๆ ว่า
"ขึ้นทุกวัน" หรือไม่ — ตลาดขาขึ้นระยะยาวทำให้การทายแบบนั้นก็ได้ราว 50%+
"""

import numpy as np


def baseline_naive_return(y_test):
    """
    Random walk: ทำนายว่าพรุ่งนี้ราคาเท่าวันนี้ -> return = 0
    ตัวนี้คือคู่แข่งตัวจริงของงานทำนายราคา
    ถ้าโมเดล ML แพ้ตัวนี้ แปลว่าโมเดลไม่มีค่า
    """
    return np.zeros(len(y_test))


def baseline_mean_return(y_train, y_test):
    """ทำนายด้วยค่าเฉลี่ย return ของ train set"""
    return np.full(len(y_test), y_train.mean())


def baseline_always_up(y_train, y_test):
    """
    ทำนายว่า "ขึ้นทุกวัน" ด้วยค่าบวกคงที่ = |ค่าเฉลี่ย return ของ train|

    ใช้ขนาดเท่ากับ mean return เพื่อให้ MAE เทียบกันได้อย่างเป็นธรรม
    ต่างกันแค่เครื่องหมายที่บังคับให้เป็นบวกเสมอ
    """
    return np.full(len(y_test), abs(float(y_train.mean())))


def always_up_note(y_train):
    """
    เตือนเรื่องที่ต้องระวังตอนอ่านตาราง

    ถ้า `y_train.mean() > 0` (ซึ่งเป็นกรณีปกติของหุ้นระยะยาว) แล้ว
    Always Up จะทำนายค่าบวกเหมือน Mean Return ทุกแถว ทำให้ `np.sign()`
    ของสองตัวนี้เท่ากันหมด -> **DirAcc จะเท่ากันเป๊ะ**

    เพราะฉะนั้นห้ามรายงานว่าเป็น baseline ด้านทิศทาง 2 ตัวที่อิสระต่อกัน
    ให้ระบุในรายงานว่าทั้งคู่คือ "ทายขึ้นทุกวัน" เหมือนกัน ต่างกันแค่ขนาด
    ที่ทำนายออกมา (จึงต่างกันแค่ MAE / RMSE ไม่ต่างกันที่ DirAcc)
    """
    mu = float(y_train.mean())
    if mu > 0:
        return (
            f"  หมายเหตุ: mean return ของ train = {mu:+.6f} (บวก)\n"
            f"  -> Always Up กับ Mean Return ทายทิศทางเหมือนกันทุกวัน "
            f"DirAcc จึงเท่ากันเป๊ะ\n"
            f"     ต่างกันแค่ขนาดที่ทำนาย (MAE/RMSE) ห้ามนับเป็น baseline "
            f"ทิศทาง 2 ตัวที่อิสระกัน"
        )
    return (
        f"  หมายเหตุ: mean return ของ train = {mu:+.6f} (ไม่เป็นบวก)\n"
        f"  -> Always Up กับ Mean Return ทายทิศทางตรงข้ามกัน "
        f"เป็น baseline คนละตัวจริง"
    )


def get_regression_baselines(y_train, y_test):
    """คืน dict {ชื่อ: prediction array}"""
    return {
        "Baseline: Naive (RW)": baseline_naive_return(y_test),
        "Baseline: Mean Return": baseline_mean_return(y_train, y_test),
        "Baseline: Always Up": baseline_always_up(y_train, y_test),
    }
