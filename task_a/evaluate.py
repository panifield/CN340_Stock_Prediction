"""
evaluate.py — งาน A (คู่/คี่)
============================
คำนวณ metric และสร้างตารางผลลัพธ์ (classification)

*** จุดสำคัญ ***
ทุกตารางต้องมี baseline อยู่ในตารางเดียวกับโมเดล
จะได้เห็นชัดๆ ว่าโมเดลชนะ baseline หรือไม่
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)


def classification_metrics(y_true, y_pred, y_proba=None):
    """คืน dict ของ metric ทั้งหมด"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    m = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            m["ROC-AUC"] = roc_auc_score(y_true, y_proba)
        except Exception:
            m["ROC-AUC"] = np.nan
    else:
        m["ROC-AUC"] = np.nan

    # สัดส่วนที่โมเดลทาย 1 -> ถ้าเป็น 0 หรือ 1 แปลว่าโมเดลทายข้างเดียว
    m["Pred_1_Rate"] = float(np.mean(y_pred))
    return m


def print_confusion(y_true, y_pred, labels=("0", "1"), title=""):
    """พิมพ์ confusion matrix แบบอ่านง่าย"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n  Confusion Matrix {title}")
    print(f"                ทำนาย {labels[0]:>6s}  ทำนาย {labels[1]:>6s}")
    print(f"    จริง {labels[0]:>6s}  {cm[0,0]:>10d}  {cm[0,1]:>12d}")
    print(f"    จริง {labels[1]:>6s}  {cm[1,0]:>10d}  {cm[1,1]:>12d}")
    return cm


def results_table(results_dict, sort_by=None, ascending=True):
    """
    results_dict = {ชื่อโมเดล: dict ของ metric}
    คืน DataFrame เรียงตาม metric ที่เลือก
    """
    df = pd.DataFrame(results_dict).T
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    return df.round(4)


def print_table(df, title=""):
    """พิมพ์ตารางพร้อมหัวข้อ"""
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")
    print(df.to_string())


