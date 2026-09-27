"""
evaluate.py — งาน B (ราคาปิด / return)
======================================
คำนวณ metric และสร้างตารางผลลัพธ์ (regression)

*** จุดสำคัญ ***
ทุกตารางต้องมี baseline อยู่ในตารางเดียวกับโมเดล
จะได้เห็นชัดๆ ว่าโมเดลชนะ baseline หรือไม่

Metric มี 5 ค่า (B1):
  primary   : MAE_return, RMSE_return, R2_return  (หน่วยที่โมเดลทำงานจริง)
  secondary : MAE_baht, RMSE_baht                 (อธิบายขนาด error เป็นบาท)
"""

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ห้ามเพิ่ม R2 ที่คำนวณบนราคา (R2_price)
# ค่านั้นได้ ~0.98 ทุกโมเดลรวมทั้ง naive -- วัดข้อมูล ไม่ได้วัดโมเดล
#
# DirAcc ใช้กับ regression ได้ แต่เป็นแค่ secondary directional diagnostic
# และต้องกำหนดนโยบายวันราคานิ่งก่อน -> ย้ายไปเฟส 2


def regression_metrics(y_true, y_pred, prev_close=None):
    """
    y_true / y_pred เป็น "return"
    ถ้าใส่ prev_close มาด้วย จะคำนวณ error ในหน่วยบาทให้ด้วย
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    m = {
        # หน่วย return -- โมเดลทำงานในหน่วยนี้ ใช้ตัดสินใจ (primary)
        "MAE_return":  mean_absolute_error(y_true, y_pred),
        "RMSE_return": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2_return":   r2_score(y_true, y_pred),
    }

    if prev_close is not None:
        prev = np.asarray(prev_close, dtype=float)
        price_true = prev * (1 + y_true)
        price_pred = prev * (1 + y_pred)
        # หน่วยบาท -- secondary interpretability metric
        m["MAE_baht"] = mean_absolute_error(price_true, price_pred)
        m["RMSE_baht"] = float(np.sqrt(mean_squared_error(price_true, price_pred)))

    return m


def results_table(results_dict, sort_by=None, ascending=True):
    """
    results_dict = {ชื่อโมเดล: dict ของ metric}
    คืน DataFrame เรียงตาม metric ที่เลือก **แบบ full precision**

    ไม่ปัดทศนิยมที่นี่ (A2): ค่าที่ปัดแล้วไม่ควรไหลเข้า compare_to_baseline()
    หรือลง CSV เพราะ CSV เป็น artifact สำหรับ audit
    ตอนนี้โมเดลต่างกันที่ราว 4e-5 จึงยังไม่กระทบ verdict
    แต่ถ้าวันหนึ่งต่างกัน 3e-7 การปัดอาจพลิกผลแพ้/ชนะได้
    การปัดทำตอนแสดงผลเท่านั้น -> print_table()
    """
    df = pd.DataFrame(results_dict).T
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    return df


def print_table(df, title="", decimals=6):
    """
    พิมพ์ตารางพร้อมหัวข้อ -- ปัดทศนิยมที่นี่ที่เดียว

    decimals=6: MAE_return อยู่ราว 0.008-0.01 และโมเดลต่างกันที่หลักที่ 5
    ถ้าปัด 4 ตำแหน่ง (แบบ v1) จะแยกโมเดลไม่ออก
    """
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")
    print(df.round(decimals).to_string())


def compare_to_baseline(df, metric, model_name,
                        baseline_name="Baseline: Naive (RW)",
                        higher_is_better=False):
    """
    เทียบโมเดลที่เลือกจาก val กับ Naive -- คู่แข่งหลักตัวเดียว (B2)
    คืนข้อความสรุปสำหรับเขียนลงรายงาน

    model_name ต้องเป็นตัวที่เลือกมาจาก val แล้วเสมอ
    (ไม่มี fallback ไปหาโมเดลที่ดีที่สุดในตาราง เพราะถ้า df เป็นตาราง test
    จะเท่ากับเอา test มาเลือกโมเดลทางอ้อม)
    """
    if model_name not in df.index or baseline_name not in df.index:
        return "ไม่มีข้อมูลพอสำหรับเปรียบเทียบ"

    model_val = df.loc[model_name, metric]
    base_val = df.loc[baseline_name, metric]
    won = model_val > base_val if higher_is_better else model_val < base_val
    verdict = "ชนะ" if won else "แพ้"

    return (
        f"  Baseline หลัก        : {baseline_name} = {base_val:.6f}\n"
        f"  โมเดลที่เลือกจาก val : {model_name} = {model_val:.6f}\n"
        f"  ผลสรุป ({metric}) : โมเดล ML {verdict} naive "
        f"(ต่างกัน {abs(model_val - base_val):.6f})"
    )
