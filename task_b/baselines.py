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
"""

import numpy as np


def baseline_naive_return(y_test):
    """
    Random walk: ทำนายว่าพรุ่งนี้ราคาเท่าวันนี้ -> return = 0
    ตัวนี้คือคู่แข่งหลักของงานทำนายราคา

    ถ้า MAE แพ้ naive ยังไม่มีหลักฐานจาก metric นี้ว่า ML ให้ประโยชน์
    เหนือ baseline -- ต้องพิจารณาความมีนัยสำคัญและมูลค่าเชิงเศรษฐศาสตร์เพิ่มเติม

    ทำแล้ว (feedback ข้อ 2): DM test + HLN correction อยู่ที่
    `stats_tests.diebold_mariano()` · ผลบน fixed val อยู่ที่
    `results/<TICKER>_taskB_val_stats.csv` · ผลบน walk-forward 6 evaluation อยู่ที่
    `results/feedback/walkforward_stats.csv` · คำอธิบายกลไกที่ Naive ได้เปรียบ
    (วันราคานิ่ง) อยู่ที่ `results/feedback/WHY_NAIVE_WINS.md`
    ส่วนมูลค่าเชิงเศรษฐศาสตร์อยู่ที่ `trading_costs.py` (ตารางที่สาม)
    """
    return np.zeros(len(y_test))


def baseline_mean_return(y_train, y_test):
    """ทำนายด้วยค่าเฉลี่ย return ของ train set"""
    return np.full(len(y_test), y_train.mean())


def get_regression_baselines(y_train, y_test):
    """คืน dict {ชื่อ: prediction array}"""
    return {
        "Baseline: Naive (RW)": baseline_naive_return(y_test),
        "Baseline: Mean Return": baseline_mean_return(y_train, y_test),
    }
