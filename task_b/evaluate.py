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


def directional_accuracy(y_true, y_pred):
    """
    ทายทิศทางถูกกี่ % — นับเฉพาะวันที่ราคาขยับจริง (y_true != 0)

    *** ทำไมต้องตัดวันราคานิ่งออก ? (นี่คือบั๊กที่แก้ในเวอร์ชันนี้) ***

    ของเดิมคำนวณ `np.mean(np.sign(y_true) == np.sign(y_pred))` ตรง ๆ
    ซึ่งให้ตัวเลขที่ "ตีความผิด" ได้ 2 ชั้น

    1. Naive baseline ทำนาย return = 0 เสมอ ทำให้ `np.sign(0) = 0`
       ไปตรงกับวันที่ราคาไม่ขยับเลยพอดี ค่า DirAcc = 0.1397 ของ KBANK
       จึงไม่ได้แปลว่า "ทายทิศทางถูก 14%" แต่แปลว่า
       "ชุดข้อมูลนี้มีวันราคานิ่งเป๊ะอยู่ 14%" ซึ่งคนละเรื่องกันเลย
    2. โมเดลที่ทำนายค่าต่อเนื่องแทบไม่มีทางทำนายออกมาได้ 0 พอดี จึงทาย
       วันราคานิ่งผิดเสมอ ทำให้เพดานสูงสุดของ DirAcc อยู่ที่ราว 86%
       (KBANK) และ 82% (ADVANC) ไม่ใช่ 100% การเอาตัวเลขนี้ไปเทียบกับ
       50% แล้วสรุปว่า "ดีกว่าการเดา" จึงเป็นการเทียบที่ผิดฐาน

    วันราคานิ่งเยอะเพราะ tick size ของ SET (ช่วง 100-200 บาทขยับทีละ
    0.50 / 200-400 บาทขยับทีละ 1.00) ถ้าหุ้นแกว่งน้อยกว่าครึ่งหนึ่งของ
    tick ราคาปิดจะถูกปัดกลับมาที่เดิม เป็นกลไกตลาดจริง ไม่ใช่ข้อมูลเสีย
    (ตรวจแล้วว่าไม่มีแถวไหน Volume = 0 เลย จึงไม่ใช่วันหยุดที่ถูกเติม)

    คืน np.nan ถ้าตัวทำนายไม่ให้สัญญาณทิศทางเลย (ทำนาย 0 ทุกแถว)
    """
    moved = y_true != 0
    if moved.sum() == 0:
        return np.nan
    if np.all(np.sign(y_pred[moved]) == 0):
        return np.nan          # Naive (RW) ทำนาย return = 0 เสมอ
    return float(np.mean(np.sign(y_true[moved]) == np.sign(y_pred[moved])))


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
        # นับเฉพาะวันที่ราคาขยับจริง -- ดู docstring ของ directional_accuracy
        "DirAcc": directional_accuracy(y_true, y_pred),
        # สัดส่วนวันราคานิ่งเป๊ะ ใส่ไว้ให้อ่าน DirAcc ได้ถูกบริบท
        # (เป็นค่าของชุดข้อมูล ไม่ใช่ของโมเดล จึงเท่ากันทุกแถวในตาราง)
        "FlatRate": float(np.mean(y_true == 0)),
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
