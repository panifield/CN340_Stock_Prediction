"""
baselines.py — งาน A (คู่/คี่)
=============================
Baseline ทั้งหมดของงาน A  <-- ส่วนที่สำคัญที่สุดของโปรเจกต์นี้

ถ้ารายงานบอกว่า "โมเดล X ได้ accuracy 0.90 ดีที่สุด"
แต่ไม่บอกว่า persistence baseline ได้ 0.88 อยู่แล้ว
อาจารย์ถามคำถามเดียวก็จบ

Baseline ที่ต้องมี:
  - majority     : ทายคลาสที่เจอบ่อยที่สุดตลอด
  - persistence  : ทายว่า parity วันนี้ = parity เมื่อวาน  <-- ตัวโหดที่สุด
  - markov1      : Markov chain อันดับ 1 บน parity (ทั่วไปกว่า persistence
                   เพราะเรียนรู้ P(parity_t | parity_{t-1}) จาก train จริง
                   แทนที่จะสมมติว่า "เท่าเดิมเสมอ")
"""

import numpy as np
import pandas as pd


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


def baseline_markov1(y_train, prev_values):
    """
    Markov chain อันดับ 1 บน parity: เรียนรู้ P(parity_t=1 | parity_{t-1})
    จาก train set แล้วทายคลาสที่มีโอกาสมากกว่าตาม state ก่อนหน้า

    ถ้า parity เหนียวมาก (ตามที่งานนี้คาดไว้) ผลจะออกมาใกล้เคียง
    persistence โดยอัตโนมัติ แต่คำนวณจากข้อมูลจริง ไม่ใช่สมมติฐานตายตัว
    """
    y_train = pd.Series(y_train).reset_index(drop=True)
    prev_state = y_train.shift(1)
    pairs = pd.DataFrame({"prev": prev_state, "curr": y_train}).dropna()

    def p1_given(state):
        subset = pairs.loc[pairs["prev"] == state, "curr"]
        return subset.mean() if len(subset) > 0 else 0.5

    p1_given_0 = p1_given(0)
    p1_given_1 = p1_given(1)

    prev_values = np.asarray(prev_values, dtype=float)
    pred = np.where(
        prev_values == 1,
        float(p1_given_1 >= 0.5),
        float(p1_given_0 >= 0.5),
    )
    return pred


def get_classification_baselines(y_train, y_test, prev_values=None):
    """คืน dict {ชื่อ: prediction array}"""
    out = {
        "Baseline: Majority": baseline_majority(y_train, y_test),
    }
    if prev_values is not None:
        out["Baseline: Persistence"] = baseline_persistence(prev_values)
        out["Baseline: Markov(1)"] = baseline_markov1(y_train, prev_values)
    return out
