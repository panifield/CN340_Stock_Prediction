"""
splits.py
=========
แบ่งข้อมูล train / validation / test

*** ห้ามใช้ train_test_split(shuffle=True) เด็ดขาด ***

ข้อมูลหุ้นเป็น time series ถ้า shuffle จะเกิดสถานการณ์
"เอาข้อมูลปี 2025 ไปเทรน แล้วกลับมาทำนายปี 2018"
ซึ่งเป็นไปไม่ได้ในโลกจริง และทำให้ผลดูดีเกินจริงมาก

ต้องแบ่งตามเวลาเท่านั้น:
    [-------- Train --------][-- Val --][-- Test --]
    2015                  2022      2023        2025
"""

import numpy as np
import pandas as pd

from config import TRAIN_RATIO, VAL_RATIO, SPLIT_BY_DATE


def chronological_split(X, y, verbose=True, name=""):
    """
    แบ่งตามเวลา คืน dict ของ (X, y) แต่ละชุด
    """
    assert X.index.equals(y.index), "index ของ X และ y ไม่ตรงกัน"
    assert X.index.is_monotonic_increasing, "index ต้องเรียงตามเวลา"

    n = len(X)

    if SPLIT_BY_DATE is not None:
        train_end = pd.Timestamp(SPLIT_BY_DATE["train_end"])
        val_end = pd.Timestamp(SPLIT_BY_DATE["val_end"])
        i_train = int((X.index <= train_end).sum())
        i_val = int((X.index <= val_end).sum())
    else:
        i_train = int(n * TRAIN_RATIO)
        i_val = int(n * (TRAIN_RATIO + VAL_RATIO))

    parts = {
        "train": (X.iloc[:i_train], y.iloc[:i_train]),
        "val":   (X.iloc[i_train:i_val], y.iloc[i_train:i_val]),
        "test":  (X.iloc[i_val:], y.iloc[i_val:]),
    }

    for k, (Xp, yp) in parts.items():
        assert len(Xp) > 0, f"ชุด {k} ว่างเปล่า ลองปรับสัดส่วนใน config"

    if verbose:
        print(f"[split] {name}")
        for k, (Xp, _) in parts.items():
            print(f"        {k:5s}: {len(Xp):5d} แถว  "
                  f"({Xp.index[0].date()} -> {Xp.index[-1].date()})")

    # ตรวจว่าไม่มีวันซ้อนทับกัน
    assert parts["train"][0].index[-1] < parts["val"][0].index[0]
    assert parts["val"][0].index[-1] < parts["test"][0].index[0]

    return parts


def walk_forward_splits(X, n_splits=5, min_train=250):
    """
    (ทางเลือก) Walk-forward validation
    ใช้ตอนอยากได้ผลที่น่าเชื่อถือกว่า split เดียว

    fold 1: train[0:200]   test[200:250]
    fold 2: train[0:250]   test[250:300]
    ...
    """
    n = len(X)
    fold_size = (n - min_train) // n_splits
    for i in range(n_splits):
        train_end = min_train + i * fold_size
        test_end = min(train_end + fold_size, n)
        if test_end <= train_end:
            break
        yield (np.arange(0, train_end), np.arange(train_end, test_end))
