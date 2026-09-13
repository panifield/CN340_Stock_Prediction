"""
loss_function.py — สูตร loss และ regularization ของงาน C
=========================================================

ไฟล์นี้อธิบายและคำนวณสูตรที่โมเดลใช้ระหว่าง ``fit`` โดยไม่ได้นำมา
แทน loss ภายใน scikit-learn/XGBoost (library เหล่านั้นคำนวณ gradient
และ optimize เอง)

นิยามร่วม
-----------
    y_i ∈ {0, 1}          label จริง (1=ขึ้น, 0=ลง/เท่าเดิม)
    p_i = P(y_i=1|x_i)    ความน่าจะเป็นที่โมเดลทำนาย
    ε                    ค่าน้อยมาก ป้องกัน log(0)

Binary cross-entropy (log loss)
-------------------------------
    BCE = -(1/n) Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]

ANN (sklearn ``MLPClassifier``)
--------------------------------
ใช้ binary cross-entropy ร่วมกับ L2 weight decay ตาม ``ANN_PARAMS``
ที่ alpha = 0.001:

    J_ANN = BCE + alpha/(2n) Σ_l ||W_l||²_F

โดย regularization คิดกับ matrix น้ำหนัก W ของทุก layer ไม่รวม bias
และ sklearn หารเทอม L2 ด้วยจำนวนตัวอย่าง n.

Logistic Regression
-------------------
ใช้ weighted binary cross-entropy เพราะตั้ง ``class_weight='balanced'``
และ L2 penalty (``penalty='l2'``). ในรูปแบบแนวคิดที่ loss ถูกเฉลี่ย:

    J_LR = weighted_BCE + 1/(2C) ||w||²_2

ค่า C=1.0 เป็นค่ากลับของความแรง regularization: C เล็กลง = ลงโทษ
น้ำหนักมากขึ้น. รายละเอียดการ scale คงที่ภายใน solver ``lbfgs`` ของ
scikit-learn อาจต่างตามการ normalize sample weight แต่ความสัมพันธ์
ระหว่าง C กับ regularization เป็นตามสูตรนี้.

XGBoost
-------
ตั้ง ``eval_metric='logloss'`` และ objective binary logistic (ค่า default
ของ ``XGBClassifier``). objective ของต้นไม้หนึ่งชุดประกอบด้วย:

    J_XGB = Σ_i BCE_i + reg_lambda/2 Σ_j w_j² + gamma * T

โดย w_j คือน้ำหนัก leaf, T คือจำนวน leaf. โปรเจกต์ตั้ง
``reg_lambda=1.0`` และใช้ gamma ค่า default (=0). L2 ของ XGBoost จึง
regularize *leaf weights* ไม่ใช่ weight ของ feature แบบ Logistic Regression.
"""

import numpy as np


EPSILON = 1e-15


def binary_cross_entropy(y_true, y_prob, sample_weight=None):
    """คำนวณ BCE เฉลี่ยตามสูตรด้านบน.

    Parameters
    ----------
    y_true : array-like
        Label 0 หรือ 1.
    y_prob : array-like
        ความน่าจะเป็นของคลาส 1.
    sample_weight : array-like, optional
        ใช้เมื่ออยากคำนวณ weighted BCE เช่นกรณี class_weight='balanced'.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), EPSILON, 1 - EPSILON)
    if y.shape != p.shape:
        raise ValueError("y_true และ y_prob ต้องมี shape เดียวกัน")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true ต้องมีเฉพาะ label 0 หรือ 1")

    per_sample = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if sample_weight is None:
        return float(per_sample.mean())

    weight = np.asarray(sample_weight, dtype=float)
    if weight.shape != y.shape:
        raise ValueError("sample_weight ต้องมี shape เดียวกับ y_true")
    if np.any(weight < 0) or weight.sum() == 0:
        raise ValueError("sample_weight ต้องไม่ติดลบ และผลรวมต้องมากกว่า 0")
    return float(np.average(per_sample, weights=weight))


def l2_penalty(weights):
    """คืนค่า Σw² สำหรับน้ำหนักหนึ่งชุดหรือหลาย layer (ไม่รวม bias)."""
    if isinstance(weights, (list, tuple)):
        return float(sum(np.sum(np.square(np.asarray(w, dtype=float)))
                         for w in weights))
    return float(np.sum(np.square(np.asarray(weights, dtype=float))))


def ann_loss(y_true, y_prob, layer_weights, alpha=1e-3):
    """J_ANN = BCE + alpha/(2n) * Σ_l ||W_l||²_F."""
    y = np.asarray(y_true)
    if len(y) == 0:
        raise ValueError("ต้องมีอย่างน้อย 1 ตัวอย่าง")
    return binary_cross_entropy(y, y_prob) + alpha * l2_penalty(layer_weights) / (2 * len(y))


def logistic_regression_loss(y_true, y_prob, coefficients, c=1.0,
                             sample_weight=None):
    """รูปแบบแนวคิด J_LR = weighted_BCE + ||w||²/(2C)."""
    if c <= 0:
        raise ValueError("C ต้องมากกว่า 0")
    return (binary_cross_entropy(y_true, y_prob, sample_weight)
            + l2_penalty(coefficients) / (2 * c))


def xgboost_objective(y_true, y_prob, leaf_weights, reg_lambda=1.0,
                      gamma=0.0, n_leaves=0, sample_weight=None):
    """J_XGB = Σ BCE_i + reg_lambda/2 Σw_leaf² + gamma*T.

    ต่างจาก ``binary_cross_entropy`` ฟังก์ชันนี้ใช้ผลรวม BCE เพื่อให้ตรง
    กับการเขียน objective ของ XGBoost. หากให้ sample_weight จะใช้
    weighted sum แทน.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), EPSILON, 1 - EPSILON)
    if y.shape != p.shape:
        raise ValueError("y_true และ y_prob ต้องมี shape เดียวกัน")
    per_sample = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if sample_weight is None:
        data_loss = float(per_sample.sum())
    else:
        weight = np.asarray(sample_weight, dtype=float)
        if weight.shape != y.shape:
            raise ValueError("sample_weight ต้องมี shape เดียวกับ y_true")
        data_loss = float(np.dot(per_sample, weight))
    return data_loss + reg_lambda * l2_penalty(leaf_weights) / 2 + gamma * n_leaves
