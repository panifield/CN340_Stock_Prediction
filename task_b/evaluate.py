"""
evaluate.py — งาน B (ราคาปิด / return)
======================================
คำนวณ metric และสร้างตารางผลลัพธ์ (regression)

*** จุดสำคัญ ***
ทุกตารางต้องมี baseline อยู่ในตารางเดียวกับโมเดล
จะได้เห็นชัดๆ ว่าโมเดลชนะ baseline หรือไม่
"""

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, prev_close=None):
    """
    y_true / y_pred เป็น "return"
    ถ้าใส่ prev_close มาด้วย จะคำนวณ error ในหน่วยบาทให้ด้วย
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    m = {
        "MAE_return": mean_absolute_error(y_true, y_pred),
        "RMSE_return": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2_return": r2_score(y_true, y_pred),
        # ทายทิศทางถูกกี่ % (สำคัญกว่า R² ในทางปฏิบัติ)
        "DirAcc": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }

    if prev_close is not None:
        prev = np.asarray(prev_close, dtype=float)
        price_true = prev * (1 + y_true)
        price_pred = prev * (1 + y_pred)
        m["MAE_baht"] = mean_absolute_error(price_true, price_pred)
        m["RMSE_baht"] = float(
            np.sqrt(mean_squared_error(price_true, price_pred))
        )
        # R² ในหน่วยราคา -> ตัวนี้แหละที่จะดูสูงหลอกๆ
        m["R2_price"] = r2_score(price_true, price_pred)

    return m


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
