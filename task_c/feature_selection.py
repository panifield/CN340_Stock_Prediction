"""
feature_selection.py — คัด feature ของงาน C โดยไม่ทำ data leakage
===================================================================

Feature ทั้ง 33 ตัวยังคงสร้างใน features.py เหมือนเดิมเสมอ ไฟล์นี้เลือก
เฉพาะ "ชื่อ column" จาก train set แล้วใช้ชื่อชุดเดิมกับ validation/test
จึงย้อนกลับไปใช้ทุก feature ได้ทันทีด้วย FEATURE_SELECTION_METHOD = "none".
"""

from functools import partial

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    RANDOM_STATE, TOP_K_FEATURES, L1_SELECTOR_C, L1_SELECTOR_MAX_ITER,
)


VALID_METHODS = {"none", "l1", "top_k"}


def select_feature_names(X_train, y_train, method):
    """Fit selector บน train เท่านั้น แล้วคืนชื่อ feature ที่ถูกเลือก.

    - l1: Logistic Regression แบบ L1; coefficient ที่ถูกบีบเป็นศูนย์จะถูกตัด
    - top_k: เลือก K feature ที่มี mutual information กับ target สูงสุด

    ขั้น impute/scale ภายใน selector fit จาก train เพื่อให้ไม่มีข้อมูล val/test
    หลุดมามีผลต่อการเลือก feature.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method ต้องเป็นหนึ่งใน {sorted(VALID_METHODS)}")
    if method == "none":
        return list(X_train.columns), None

    preprocess = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    X_train_ready = preprocess.fit_transform(X_train)

    if method == "l1":
        estimator = LogisticRegression(
            penalty="l1", solver="saga", C=L1_SELECTOR_C,
            class_weight="balanced", max_iter=L1_SELECTOR_MAX_ITER,
            random_state=RANDOM_STATE,
        )
        # threshold เล็กมาก: feature ถูกเก็บเมื่อ L1 ไม่ได้บีบ coefficient เป็น 0
        selector = SelectFromModel(estimator, threshold=1e-12)
        selector.fit(X_train_ready, y_train)
        support = selector.get_support()
        scores = np.abs(selector.estimator_.coef_).ravel()
    else:  # top_k
        k = min(TOP_K_FEATURES, X_train.shape[1])
        score_fn = partial(mutual_info_classif, random_state=RANDOM_STATE)
        selector = SelectKBest(score_func=score_fn, k=k)
        selector.fit(X_train_ready, y_train)
        support = selector.get_support()
        scores = selector.scores_

    selected = list(X_train.columns[support])
    if not selected:
        raise RuntimeError("feature selector ไม่เลือก feature เลย; ปรับค่า config")

    ranking = (pd.DataFrame({"feature": X_train.columns, "score": scores})
               .sort_values("score", ascending=False, na_position="last"))
    return selected, ranking
