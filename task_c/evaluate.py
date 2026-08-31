"""
evaluate.py — งาน C (ขึ้น/ลง)
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


def compare_to_baseline(df, metric, baseline_prefix="Baseline",
                        higher_is_better=True, model_name=None):
    """
    ตรวจว่าโมเดล ML ชนะ baseline ที่ดีที่สุดหรือไม่
    คืนข้อความสรุปสำหรับเขียนลงรายงาน

    model_name: ชื่อโมเดลที่จะเทียบ (ควรเป็นตัวที่เลือกมาจาก val แล้ว)
    ถ้าไม่ใส่ จะ fallback ไปหาโมเดลที่ดีที่สุด "ในตาราง df นี้" เอง
    (ระวัง: ถ้า df เป็นตาราง test การ fallback แบบนี้เท่ากับเอา test
    มาเลือกโมเดลทางอ้อม ไม่ควรใช้ fallback กับตาราง test)
    """
    is_base = df.index.str.startswith(baseline_prefix)
    baselines = df[is_base]
    models = df[~is_base]

    if len(baselines) == 0 or len(models) == 0:
        return "ไม่มีข้อมูลพอสำหรับเปรียบเทียบ"

    if higher_is_better:
        best_base = baselines[metric].max()
        best_base_name = baselines[metric].idxmax()
        if model_name is not None:
            best_model_name, best_model = model_name, models.loc[model_name, metric]
        else:
            best_model = models[metric].max()
            best_model_name = models[metric].idxmax()
        won = best_model > best_base
    else:
        best_base = baselines[metric].min()
        best_base_name = baselines[metric].idxmin()
        if model_name is not None:
            best_model_name, best_model = model_name, models.loc[model_name, metric]
        else:
            best_model = models[metric].min()
            best_model_name = models[metric].idxmin()
        won = best_model < best_base

    diff = abs(best_model - best_base)
    verdict = "ชนะ" if won else "แพ้"

    return (
        f"  Baseline ที่ดีที่สุด : {best_base_name} = {best_base:.4f}\n"
        f"  โมเดลที่ดีที่สุด     : {best_model_name} = {best_model:.4f}\n"
        f"  ผลสรุป ({metric}) : โมเดล ML {verdict} baseline "
        f"(ต่างกัน {diff:.4f})"
    )
