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


def prediction_shape(y_true, y_pred):
    """
    แยก "ทักษะ" ออกจาก "การปรับเทียบ" -- คืน (StdRatio, Rho, Bias)

    *** ทำไมต้องมี 3 ตัวนี้ ? ***

    R² กับ MAE บอกแค่ว่าโมเดล "แย่" แต่บอกไม่ได้ว่าแย่เพราะอะไร
    ทั้งที่ R² แตกออกเป็น 3 ส่วนได้ตรง ๆ:

        R² = 2*Rho*StdRatio - StdRatio^2 - Bias^2

    โดย
      StdRatio = SD(pred) / SD(true)
                 ความ "กว้าง" ของการทำนายเทียบกับความจริง
                 ค่าที่เหมาะสมทางทฤษฎีคือ StdRatio = Rho
                 ถ้า >> Rho แปลว่าทำนายแกว่งเกินจริง (ปรับเทียบผิด)
      Rho      = correlation ระหว่าง pred กับ true  = ทักษะที่แท้จริง
                 เพดานของ R² ที่ทำได้หลังปรับเทียบคือ Rho^2
      Bias     = (mean(pred) - mean(true)) / SD(true)
                 ทำนายเอียงไปทางเดียวอย่างเป็นระบบแค่ไหน
                 เป็นสัญญาณตรงของ distribution shift ระหว่าง train กับ
                 ชุดที่ประเมิน (เช่น ราคาใน val หลุดช่วงที่ train เคยเห็น)

    *** ข้อควรระวัง: ต้องคำนวณ Rho ตรง ๆ ห้ามย้อนจากสูตร ***
    ถ้าย้อนหา Rho จาก R² กับ StdRatio โดยลืมพจน์ Bias^2 จะได้ค่าที่ผิด
    และผิดมากพอจะพลิกเครื่องหมายได้ (เคยเจอกรณีที่คำนวณย้อนได้ Rho ติดลบ
    ทั้งที่ค่าจริงเป็นบวก) ฟังก์ชันนี้จึงใช้ np.corrcoef ตรง ๆ

    ตัวทำนายที่ให้ค่าคงที่ (Naive / Mean Return / Always Up) ไม่มีการ
    กระจายเลย -> StdRatio = 0 และ Rho = NaN เพราะ correlation ไม่นิยาม

    หมายเหตุ: เช็คว่า "คงที่" ด้วย np.ptp (ค่าสูงสุด - ค่าต่ำสุด) ไม่ใช่
    เช็คว่า SD == 0 เพราะ np.std ของอาเรย์ค่าคงที่ที่ไม่ใช่ศูนย์จะได้ค่า
    เล็กมาก (~1e-19) แต่ไม่เป็นศูนย์เป๊ะจาก floating point ทำให้เผลอไป
    คำนวณ correlation ของ noise แล้วได้ Rho ปลอม ๆ ออกมาเป็น 0.0000
    """
    sd_true = float(np.std(y_true))

    if sd_true == 0:
        return np.nan, np.nan, np.nan

    bias = float(np.mean(y_pred) - np.mean(y_true)) / sd_true

    if np.ptp(y_pred) == 0:          # ตัวทำนายค่าคงที่
        return 0.0, np.nan, bias

    std_ratio = float(np.std(y_pred)) / sd_true
    rho = float(np.corrcoef(y_pred, y_true)[0, 1])

    return std_ratio, rho, bias


def rho_confidence_interval(rho, n, z=1.96):
    """
    ช่วงความเชื่อมั่น 95% ของ Rho -- คืน (lo, hi)

    *** ทำไมต้องมี ? ***
    Rho ที่คำนวณได้มาจาก "ตัวอย่างชุดเดียว" (val 365 วัน) ไม่ใช่ค่าจริงของ
    ประชากร ถ้ารายงานแค่ตัวเลขเปล่า ๆ คนอ่านจะตีความ Rho = 0.10 ว่า
    "โมเดลมีทักษะ" ทั้งที่ค่าจริงอาจเป็น 0 ก็ได้ การใส่ CI คือการบอกว่า
    ตัวเลขนี้ "ไม่แน่นอนแค่ไหน"

    ใช้การประมาณแบบปกติ  SE(Rho) ~= 1 / sqrt(n)
    ที่ n = 365 -> SE = 0.0523 -> ครึ่งความกว้าง CI = 1.96 * 0.0523 = 0.1026
    แปลว่า **ต้องมี |Rho| > 0.103 ขึ้นไป ถึงจะพูดได้ว่าต่างจากศูนย์**
    ซึ่งสูงกว่า Rho ของเกือบทุกโมเดลในงานนี้

    ทางเลือกที่ทั่วไปกว่าคือ Fisher z-transform (arctanh) ที่ถูกต้องกับ Rho
    ทุกค่า แต่ในช่วง |Rho| < 0.2 ซึ่งเป็นกรณีของงานนี้ทั้งหมด ให้ผลต่างกัน
    ไม่ถึง 0.005 จึงใช้สูตรง่ายที่อธิบายในรายงานได้ตรงไปตรงมากว่า

    *** ข้อควรระวังที่ต้องเขียนในรายงาน ***
    สูตร 1/sqrt(n) ตั้งอยู่บนสมมติฐานว่าแต่ละวันเป็นอิสระต่อกัน แต่ผลตอบแทน
    รายวันจริงมี volatility clustering (วันผันผวนเกาะกลุ่มกัน) ทำให้จำนวน
    ตัวอย่างที่อิสระจริงน้อยกว่า n -> **CI จริงกว้างกว่าที่คำนวณตรงนี้**
    เกณฑ์นี้จึง "ใจดีกับโมเดลแล้ว" ถ้า CI ยังคร่อม 0 อยู่ ก็ยิ่งสรุปได้แน่น
    ขึ้นว่าไม่มีทักษะ (การทำให้แม่นกว่านี้ต้องใช้ HAC standard error)
    """
    if n is None or n < 4 or not np.isfinite(rho):
        return np.nan, np.nan

    half = z / np.sqrt(n)
    # correlation ถูกจำกัดใน [-1, 1] อยู่แล้วโดยนิยาม
    return max(-1.0, float(rho) - half), min(1.0, float(rho) + half)


def rho_significance_note(results_dict):
    """
    สรุปว่าโมเดลไหน "มีทักษะจริง" บ้าง โดยดูว่า 95% CI ของ Rho คร่อม 0 ไหม

    ตัวทำนายค่าคงที่ (Naive / Mean Return / Always Up) มี Rho = NaN
    จึงถูกข้ามไปโดยอัตโนมัติ -- เทียบได้เฉพาะโมเดลที่ให้ค่าแปรผันจริง
    """
    tested = []
    for name, m in results_dict.items():
        lo = m.get("Rho_lo", np.nan)
        hi = m.get("Rho_hi", np.nan)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue           # baseline ค่าคงที่ -> correlation ไม่นิยาม
        tested.append((name, float(m["Rho"]), float(lo), float(hi)))

    if not tested:
        return "  (ไม่มีตัวทำนายที่คำนวณ Rho ได้)"

    lines = ["  ทักษะจริง (Rho) ต่างจากศูนย์ไหม เมื่อดูช่วงความเชื่อมั่น 95%:"]

    n_sig = 0
    for name, rho, lo, hi in sorted(tested, key=lambda r: -r[1]):
        crosses_zero = lo <= 0 <= hi
        if crosses_zero:
            mark = "คร่อม 0 -> สรุปไม่ได้ว่ามีทักษะ"
        else:
            mark = "ไม่คร่อม 0 -> มีนัยสำคัญ"
            n_sig += 1
        lines.append(f"    {name:16s} Rho = {rho:+.4f}  "
                     f"CI = [{lo:+.4f}, {hi:+.4f}]  {mark}")

    lines.append(f"  -> {n_sig} จาก {len(tested)} โมเดล ที่ทักษะต่างจากศูนย์"
                 f"อย่างมีนัยสำคัญ")

    if n_sig > 0:
        lines.append("     ระวัง 2 เรื่องก่อนสรุปว่า 'เจอสัญญาณจริง':")
        lines.append("     (1) การทดสอบหลายครั้ง: ทั้งโปรเจกต์มี 3 โมเดล x 2 หุ้น "
                     "= 6 การทดสอบ")
        lines.append("         ที่ระดับ 95% คาดว่าจะเจอตัวรอดแบบบังเอิญ "
                     "6 x 0.05 = 0.3 ตัว")
        lines.append("         และโอกาสเจออย่างน้อย 1 ตัวโดยบังเอิญ "
                     "= 1 - 0.95^6 = 26.5%")
        lines.append("     (2) selection bias: ANN ถูกเลือกค่าพารามิเตอร์จาก val "
                     "(กวาด 98 ชุด)")
        lines.append("         ค่า Rho ของ ANN บน val จึงเป็นค่าที่ 'ผ่านการคัดมาแล้ว' "
                     "ย่อมเข้าข้างตัวเอง")
        lines.append("         ตัวเลขที่ไม่เอียงต้องดูจาก test ซึ่งยังไม่เปิด")

    lines.append("     และ SE = 1/sqrt(n) สมมติว่าแต่ละวันอิสระกัน ทั้งที่ผลตอบแทนจริง")
    lines.append("     มี volatility clustering -> CI จริงกว้างกว่านี้ "
                 "(เกณฑ์นี้ใจดีกับโมเดลแล้ว)")

    return "\n".join(lines)


def regression_metrics(y_true, y_pred, prev_close=None):
    """
    y_true / y_pred เป็น "return"
    ถ้าใส่ prev_close มาด้วย จะคำนวณ error ในหน่วยบาทให้ด้วย
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    std_ratio, rho, bias = prediction_shape(y_true, y_pred)
    rho_lo, rho_hi = rho_confidence_interval(rho, len(y_true))

    m = {
        "MAE_return": mean_absolute_error(y_true, y_pred),
        "RMSE_return": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2_return": r2_score(y_true, y_pred),
        # 3 ตัวนี้แตก R2_return ออกเป็นส่วน ๆ -- ดู prediction_shape()
        "StdRatio": std_ratio,
        "Rho": rho,
        # ช่วงความเชื่อมั่น 95% ของ Rho -- ถ้าคร่อม 0 แปลว่าทักษะที่วัดได้
        # ยังแยกไม่ออกจากศูนย์ ดู rho_confidence_interval()
        "Rho_lo": rho_lo,
        "Rho_hi": rho_hi,
        "Bias": bias,
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
                        higher_is_better=True, model_name=None,
                        noise_threshold=0.02):
    """
    ตรวจว่าโมเดล ML ชนะ baseline ที่ดีที่สุดหรือไม่
    คืนข้อความสรุปสำหรับเขียนลงรายงาน

    model_name: ชื่อโมเดลที่จะเทียบ (ควรเป็นตัวที่เลือกมาจาก val แล้ว)
    ถ้าไม่ใส่ จะ fallback ไปหาโมเดลที่ดีที่สุด "ในตาราง df นี้" เอง
    (ระวัง: ถ้า df เป็นตาราง test การ fallback แบบนี้เท่ากับเอา test
    มาเลือกโมเดลทางอ้อม ไม่ควรใช้ fallback กับตาราง test)

    noise_threshold: ถ้าโมเดลกับ baseline ต่างกัน "น้อยกว่า" สัดส่วนนี้
    ของค่า baseline (ค่าเริ่มต้น 2%) ให้ถือว่าต่างกันในระดับ noise —
    ยังสรุปไม่ได้ว่าใครดีกว่ากัน ไม่รายงานว่า "ชนะ"
    เหตุผล: ตัวอย่าง val/test มีแค่ ~365 วัน ความต่าง MAE ระดับ 0.2%
    จมอยู่ในความแปรปรวนของการสุ่มตัวอย่าง การเคลมชนะบนส่วนต่างขนาดนั้น
    คือการอ่าน noise เป็นสัญญาณ (ดู thesis หลักของงาน: อย่าเคลมว่าชนะ
    random walk ทั้งที่ส่วนต่างไม่มีนัยสำคัญ)
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
    # ส่วนต่างเชิงสัมพัทธ์เทียบกับค่า baseline (กัน baseline = 0)
    rel_diff = diff / abs(best_base) if best_base != 0 else np.inf
    is_noise = rel_diff < noise_threshold

    if is_noise:
        # ต่างกันน้อยเกินกว่าจะสรุปได้ ไม่ว่าจะเป็นฝั่งชนะหรือแพ้
        verdict_line = (
            f"  ผลสรุป ({metric}) : ต่างกันแค่ {rel_diff*100:.2f}% "
            f"({diff:.4f}) < เกณฑ์ noise {noise_threshold*100:.0f}%\n"
            f"    -> อยู่ในระดับ noise ยังสรุปไม่ได้ว่าโมเดล ML "
            f"ดีกว่า baseline จริง (เสมอกันในเชิงสถิติ)"
        )
    else:
        verdict = "ชนะ" if won else "แพ้"
        verdict_line = (
            f"  ผลสรุป ({metric}) : โมเดล ML {verdict} baseline "
            f"(ต่างกัน {diff:.4f} = {rel_diff*100:.2f}%)"
        )

    return (
        f"  Baseline ที่ดีที่สุด : {best_base_name} = {best_base:.4f}\n"
        f"  โมเดลที่ดีที่สุด     : {best_model_name} = {best_model:.4f}\n"
        + verdict_line
    )
