"""
baselines.py — งาน C (ขึ้น/ลง)
==============================
Baseline ของงาน C  <-- ส่วนที่สำคัญที่สุดของโปรเจกต์นี้

ถ้ารายงานบอกว่า "XGBoost ได้ accuracy 0.55 ดีที่สุด"
แต่ไม่บอกว่า always-up baseline ได้ 0.52 อยู่แล้ว
อาจารย์ถามคำถามเดียวก็จบ

Baseline ที่ต้องมี:
  - always_up    : ทาย "ขึ้น" ตลอด (ตลาดขาขึ้นระยะยาว base rate ~52%)
  - majority     : ทายคลาสที่เจอบ่อยที่สุดใน train
  - persistence  : ทายว่าวันนี้ไปทางเดียวกับเมื่อวาน
  - random       : เดาสุ่ม 50/50
"""

import numpy as np


def baseline_majority(y_train, y_test):
    """ทายคลาสที่เจอบ่อยที่สุดใน train set ตลอด"""
    majority_class = y_train.mode().iloc[0]
    return np.full(len(y_test), majority_class)


def baseline_persistence(prev_values):
    """ทายว่า "วันนี้เหมือนเมื่อวาน" (prev_values ต้อง align กับ y_test แล้ว)"""
    return np.asarray(prev_values, dtype=float)


def baseline_random(y_test, seed=42):
    """เดาสุ่ม 50/50"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, len(y_test)).astype(float)


def baseline_always_positive(y_test):
    """ทาย 1 ตลอด (ทายว่าขึ้นตลอด)"""
    return np.ones(len(y_test))


def get_classification_baselines(y_train, y_test, prev_values=None,
                                 include_always_up=True):
    """คืน dict {ชื่อ: prediction array}"""
    out = {
        "Baseline: Majority": baseline_majority(y_train, y_test),
        "Baseline: Random": baseline_random(y_test),
    }
    if prev_values is not None:
        out["Baseline: Persistence"] = baseline_persistence(prev_values)
    if include_always_up:
        out["Baseline: Always Up"] = baseline_always_positive(y_test)
    return out
