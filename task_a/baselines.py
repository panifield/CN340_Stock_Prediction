"""
baselines.py — งาน A (คู่/คี่)
=============================
Baseline ทั้งหมดของงาน A  <-- ส่วนที่สำคัญที่สุดของโปรเจกต์นี้

ถ้ารายงานบอกว่า "XGBoost ได้ accuracy 0.90 ดีที่สุด"
แต่ไม่บอกว่า persistence baseline ได้ 0.88 อยู่แล้ว
อาจารย์ถามคำถามเดียวก็จบ

Baseline ที่ต้องมี:
  - majority     : ทายคลาสที่เจอบ่อยที่สุดตลอด
  - persistence  : ทายว่า parity วันนี้ = parity เมื่อวาน  <-- ตัวโหดที่สุด
  - random       : เดาสุ่ม 50/50
"""

import numpy as np


def baseline_majority(y_train, y_test):
    """ทายคลาสที่เจอบ่อยที่สุดใน train set ตลอด"""
    majority_class = y_train.mode().iloc[0]
    return np.full(len(y_test), majority_class)


def baseline_persistence(prev_values):
    """
    ทายว่า "วันนี้เหมือนเมื่อวาน"
    prev_values = ค่าของเมื่อวาน (ต้อง align กับ y_test แล้ว)

    สำหรับงาน parity ตัวนี้แข็งมาก เพราะถ้าราคาขยับน้อยกว่า 1 บาท
    เลขจำนวนเต็มจะไม่เปลี่ยน -> parity เท่าเดิม
    """
    return np.asarray(prev_values, dtype=float)


def baseline_random(y_test, seed=42):
    """เดาสุ่ม 50/50"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, len(y_test)).astype(float)


def get_classification_baselines(y_train, y_test, prev_values=None):
    """คืน dict {ชื่อ: prediction array}"""
    out = {
        "Baseline: Majority": baseline_majority(y_train, y_test),
        "Baseline: Random": baseline_random(y_test),
    }
    if prev_values is not None:
        out["Baseline: Persistence"] = baseline_persistence(prev_values)
    return out
