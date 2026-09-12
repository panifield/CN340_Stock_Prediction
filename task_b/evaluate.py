"""
evaluate.py — งาน B (ราคาปิด / return)
======================================
คำนวณ metric และสร้างตารางผลลัพธ์ (regression)

*** จุดสำคัญ ***
ทุกตารางต้องมี baseline อยู่ในตารางเดียวกับโมเดล
จะได้เห็นชัดๆ ว่าโมเดลชนะ baseline หรือไม่
"""

import math

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import TICKERS, SWEEP_HISTORY


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


def rho_significance_note(results_dict, stage="val"):
    """
    สรุปว่าโมเดลไหน "มีทักษะจริง" บ้าง โดยดูว่า 95% CI ของ Rho คร่อม 0 ไหม

    ตัวทำนายค่าคงที่ (Naive / Mean Return / Always Up) มี Rho = NaN
    จึงถูกข้ามไปโดยอัตโนมัติ -- เทียบได้เฉพาะโมเดลที่ให้ค่าแปรผันจริง

    stage = "val" หรือ "test" -- ข้อความเรื่อง selection bias ต่างกันตามชุด
    ที่ประเมิน (บน val ค่า ANN ผ่านการคัดมาแล้ว / บน test ไม่ได้ผ่านการคัด)
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

    lines.append(f"  -> {n_sig} จาก {len(tested)} ตัวทำนาย ที่ทักษะต่างจากศูนย์"
                 f"อย่างมีนัยสำคัญ")

    if n_sig > 0:
        # นับเฉพาะโมเดล ML ในการคิดจำนวนการทดสอบ -- baseline ที่ไม่ได้ทำนาย
        # ค่าคงที่ (rolling mean) ก็คำนวณ Rho ได้และถูกแสดงในรายการข้างบน
        # แต่ไม่ได้ถูกนับเป็น "การทดสอบสมมติฐานของโมเดล"
        n_models = len([t for t in tested if not t[0].startswith("Baseline")])
        n_tests = n_models * len(TICKERS)

        # จำนวน config ที่แต่ละโมเดลถูกกวาดบน val -- ไม่เท่ากัน ต้องรายงานตามจริง
        tuned = []
        for model_name, hist in SWEEP_HISTORY.items():
            if hist:
                total = sum(c for _, c in hist)
                detail = " + ".join(f"{c} บน {f} feat" for f, c in hist)
                tuned.append(f"{model_name}: {total} ชุด ({detail})")
            else:
                tuned.append(f"{model_name}: ไม่ได้จูน (ใช้ค่าตั้งต้น)")

        lines.append("     ระวัง 2 เรื่องก่อนสรุปว่า 'เจอสัญญาณจริง':")
        lines.append(f"     (1) การทดสอบหลายครั้ง: ทั้งโปรเจกต์มี {n_models} โมเดล x "
                     f"{len(TICKERS)} หุ้น = {n_tests} การทดสอบ")
        lines.append("         ที่ระดับ 95% คาดว่าจะเจอตัวรอดแบบบังเอิญ "
                     f"{n_tests} x 0.05 = {n_tests * 0.05:.1f} ตัว")
        lines.append("         และโอกาสเจออย่างน้อย 1 ตัวโดยบังเอิญ "
                     f"= 1 - 0.95^{n_tests} = {(1 - 0.95 ** n_tests) * 100:.1f}%")
        lines.append("         (ถ้ามี Ensemble อยู่ในรายการ การทดสอบไม่อิสระกันจริง "
                     "เพราะมันคือค่าเฉลี่ยของอีก 3 ตัว)")
        if stage == "val":
            lines.append("     (2) selection bias: พารามิเตอร์ถูกเลือกจาก val "
                         "และแต่ละโมเดลได้โอกาสไม่เท่ากัน")
            for t in tuned:
                lines.append(f"         - {t}")
            lines.append("         ค่าที่วัดได้บน val จึง 'ผ่านการคัดมาแล้ว' "
                         "ย่อมเข้าข้างตัวเอง (โมเดลที่กวาดมากกว่ายิ่งเอียงมากกว่า)")
            lines.append("         ตัวเลขที่ไม่เอียงต้องดูจาก test ซึ่งยังไม่เปิด")
        else:
            lines.append("     (2) ค่าบน test ไม่ได้ผ่านการคัด: พารามิเตอร์ถูกเลือกจาก val "
                         "โดยแต่ละโมเดลได้โอกาสไม่เท่ากัน")
            for t in tuned:
                lines.append(f"         - {t}")
            lines.append("         ไม่มีการใช้ test เลือกอะไรเลย -> ค่าบน test "
                         "ไม่เอียงจากการจูน")
            lines.append("         ถ้า Rho บน test ต่ำกว่าบน val ชัดเจน = สัญญาณว่าค่าบน val "
                         "เคยเข้าข้างตัวเอง")

    lines.append("     และ SE = 1/sqrt(n) สมมติว่าแต่ละวันอิสระกัน ทั้งที่ผลตอบแทนจริง")
    lines.append("     มี volatility clustering -> CI จริงกว้างกว่านี้ "
                 "(เกณฑ์นี้ใจดีกับโมเดลแล้ว)")

    return "\n".join(lines)


def diebold_mariano(y_true, pred_model, pred_base, lag=None):
    """
    Diebold-Mariano test: โมเดลคลาดเคลื่อนต่างจาก baseline จริงไหม
    คืน (z, p)  z < 0 = โมเดลคลาดเคลื่อนน้อยกว่า baseline, p แบบสองทาง

    *** ทำไมต้องมี ? (ใช้แทนเกณฑ์ noise 2% ของ compare_to_baseline) ***
    เกณฑ์ 2% เป็นตัวเลขที่ตั้งเอง ไม่ได้ขึ้นกับว่าข้อมูลแกว่งแค่ไหน
    DM test ดูที่ "ผลต่างของ loss รายวัน" d_t = |e_model,t| - |e_base,t|
    แล้วถามว่าค่าเฉลี่ยของ d_t ต่างจาก 0 เกินกว่าความแปรปรวนของมันไหม
    = การทดสอบมาตรฐานสำหรับเทียบความแม่นของการพยากรณ์สองชุด

    loss = ค่าคลาดเคลื่อนสัมบูรณ์ของ return (สอดคล้องกับ MAE_return)

    ความแปรปรวนใช้ Newey-West (HAC) เพราะ d_t เกาะกลุ่มกันตาม volatility
    clustering -- ถ้าใช้สูตรที่สมมติว่าแต่ละวันอิสระกัน p จะเล็กเกินจริง
    lag ตั้งตามกฎ Newey-West: floor(4 * (n/100)^(2/9)) -> n = 362 ได้ lag = 5

    หมายเหตุ: ไม่ได้ใส่ small-sample correction ของ Harvey-Leybourne-Newbold
    เพราะที่ n ~ 360 และทำนายล่วงหน้า 1 วัน ตัวคูณมีค่า ~0.999 แทบไม่ต่าง
    """
    y = np.asarray(y_true, dtype=float)
    d = (np.abs(y - np.asarray(pred_model, dtype=float))
         - np.abs(y - np.asarray(pred_base, dtype=float)))
    n = len(d)
    if n < 10 or np.ptp(d) == 0:
        return np.nan, np.nan          # ตัวทำนายเหมือนกันทุกวัน -> ไม่มีอะไรให้ทดสอบ

    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))

    dc = d - d.mean()
    var = np.sum(dc * dc) / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * np.sum(dc[k:] * dc[:-k]) / n
    if var <= 0:
        return np.nan, np.nan

    z = float(d.mean() / np.sqrt(var / n))
    p = float(math.erfc(abs(z) / math.sqrt(2)))
    return z, p


def dm_note(results_dict, base_name, alpha=0.05):
    """สรุปผล DM test ของทุกตัวทำนายเทียบกับ base_name เป็นข้อความ"""
    lines = [f"  Diebold-Mariano test เทียบกับ {base_name} "
             f"(loss = |error ของ return|, HAC):"]
    n_better = n_worse = 0
    for name, m in results_dict.items():
        z, p = m.get("DM_z", np.nan), m.get("DM_p", np.nan)
        if not (np.isfinite(z) and np.isfinite(p)):
            continue
        if p >= alpha:
            mark = "ต่างกันไม่มีนัยสำคัญ"
        elif z < 0:
            mark = "ดีกว่าอย่างมีนัยสำคัญ"
            n_better += 1
        else:
            mark = "แย่กว่าอย่างมีนัยสำคัญ"
            n_worse += 1
        lines.append(f"    {name:22s} z = {z:+.3f}  p = {p:.3f}  {mark}")
    lines.append(f"  -> ดีกว่า {base_name} อย่างมีนัยสำคัญ {n_better} ตัว / "
                 f"แย่กว่า {n_worse} ตัว (ที่ระดับ {alpha})")
    lines.append("     z < 0 = คลาดเคลื่อนน้อยกว่า baseline "
                 "(ไม่ได้ปรับสำหรับการทดสอบหลายครั้ง)")
    return "\n".join(lines)


def r2_oos(y_true, y_pred, benchmark_pred=None):
    """
    R² out-of-sample แบบ Campbell-Thompson -- เทียบกับ "ตัวเปรียบเทียบ" ที่ประกาศไว้

        R2_OOS = 1 - SSE(model) / SSE(benchmark)

    ค่าเริ่มต้นของ benchmark คือ Naive (ทำนาย return = 0) ซึ่งเป็นคู่แข่งที่
    ประกาศไว้ตั้งแต่ต้นโปรเจกต์

    *** ต่างจาก R2_return อย่างไร ***
    R2_return (r2_score ของ sklearn) เทียบกับ "ค่าเฉลี่ยของชุดที่กำลังประเมิน"
    ซึ่งเป็นค่าที่ ณ เวลาทำนายยังไม่มีทางรู้ได้ (ต้องรู้อนาคตทั้งชุดก่อน)
    R2_OOS เทียบกับตัวทำนายที่ใช้ได้จริง ณ เวลานั้น จึงตีความได้ตรงกว่าว่า
    "โมเดลลด squared error ลงจาก Naive ได้กี่ %"

      R2_OOS > 0  = ดีกว่า Naive
      R2_OOS = 0  = เท่ากับ Naive พอดี (Naive เทียบกับตัวเองได้ 0 เสมอ)
      R2_OOS < 0  = แย่กว่า Naive

    หมายเหตุ: ค่านี้ไม่บอกนัยสำคัญทางสถิติ ต้องอ่านคู่กับ DM test เสมอ
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    b = np.zeros_like(y) if benchmark_pred is None else np.asarray(
        benchmark_pred, dtype=float)

    sse_bench = float(np.sum((y - b) ** 2))
    if sse_bench == 0:
        return np.nan
    return 1.0 - float(np.sum((y - p) ** 2)) / sse_bench


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
        # R2 เทียบกับ Naive (ทำนาย return = 0) -- ดู r2_oos()
        "R2_OOS": r2_oos(y_true, y_pred),
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


# ---------------------------------------------------------------
# Error analysis แยกตาม market regime
# ---------------------------------------------------------------

def regime_error_analysis(y_true, preds_dict, vol_prev, vol_threshold,
                          baseline_name="Baseline: Naive (RW)"):
    """
    แยกชุดที่ประเมินออกเป็นกลุ่มตามสภาวะตลาด แล้วคำนวณ MAE_return แยกแต่ละกลุ่ม

    ใช้ prediction ที่มีอยู่แล้ว ไม่มีการเทรนใหม่ จึงไม่ได้ใช้ข้อมูลเพิ่มเติม
    ในการตัดสินใจอะไรทั้งสิ้น -- เป็นการ "อ่านผลที่มีอยู่ให้ละเอียดขึ้น"

    *** เส้นแบ่ง volatility ต้องมาจาก train เท่านั้น ***
    ถ้าใช้ median ของชุดที่กำลังประเมิน = เอาข้อมูลที่ประเมินมากำหนดเกณฑ์
    ซึ่งทำให้ขนาดของสองกลุ่มถูกบังคับให้เท่ากันเสมอโดยไม่มีเหตุผลเชิงเนื้อหา
    และทำให้เทียบข้ามชุด (val กับ test) ไม่ได้ เพราะเส้นแบ่งคนละเส้น

    กลุ่มที่แบ่ง:
      - volatility สูง / ต่ำ  (เทียบ median ของ volatility_20d_prev บน train)
      - วันขึ้น / วันลง / วันนิ่ง  (ดูจาก y_true)
      - ทั้งหมด (ไว้ตรวจว่าถ่วงน้ำหนักแล้วตรงกับตารางหลัก)

    คืน DataFrame แบบยาว (regime, n_days, model, mae_return, vs_naive_pct)
    โดย vs_naive_pct > 0 = ดีกว่า Naive ในกลุ่มนั้น (MAE ต่ำกว่ากี่ %)
    """
    y = pd.Series(np.asarray(y_true, dtype=float),
                  index=pd.Index(y_true.index) if hasattr(y_true, "index")
                  else None)
    vol = pd.Series(np.asarray(vol_prev, dtype=float), index=y.index)

    regimes = {
        "ทั้งหมด": pd.Series(True, index=y.index),
        f"vol สูง (> {vol_threshold:.5f})": vol > vol_threshold,
        f"vol ต่ำ (<= {vol_threshold:.5f})": vol <= vol_threshold,
        "วันขึ้น (y > 0)": y > 0,
        "วันลง (y < 0)": y < 0,
        "วันนิ่ง (y = 0)": y == 0,
    }

    rows = []
    for regime, mask in regimes.items():
        n = int(mask.sum())
        if n == 0:
            continue

        mae = {}
        for name, p in preds_dict.items():
            p = np.asarray(p, dtype=float)
            mae[name] = float(np.mean(np.abs(y.values[mask.values]
                                             - p[mask.values])))

        base_mae = mae.get(baseline_name, np.nan)
        for name, value in mae.items():
            if np.isfinite(base_mae) and base_mae != 0:
                vs = (base_mae - value) / base_mae * 100.0
            else:
                vs = np.nan
            rows.append({
                "regime": regime, "n_days": n, "model": name,
                "mae_return": value, "vs_naive_pct": vs,
            })

    return pd.DataFrame(rows)


def print_regime_table(df, title="", baseline_name="Baseline: Naive (RW)"):
    """พิมพ์ผล regime analysis เป็นตาราง (แถว = ตัวทำนาย, คอลัมน์ = กลุ่ม)"""
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")

    mae = df.pivot(index="model", columns="regime", values="mae_return")
    vs = df.pivot(index="model", columns="regime", values="vs_naive_pct")
    n_days = df.drop_duplicates("regime").set_index("regime")["n_days"]

    # เรียงคอลัมน์ตามลำดับที่สร้างไว้ (pivot เรียงตามตัวอักษร)
    order = [r for r in df["regime"].unique() if r in mae.columns]
    mae, vs = mae[order], vs[order]

    print("  MAE_return แยกตามกลุ่ม (จำนวนวันในวงเล็บ)")
    header = "  " + " " * 24 + "".join(
        f"{r.split(' (')[0]:>18s}" for r in order)
    print(header)
    print("  " + " " * 24 + "".join(f"{'n=' + str(n_days[r]):>18s}"
                                    for r in order))
    for m in mae.index:
        print(f"  {m:24s}" + "".join(f"{mae.loc[m, r]:18.6f}" for r in order))

    print(f"\n  ดีกว่า {baseline_name} กี่ % ในกลุ่มนั้น (+ = ดีกว่า)")
    print(header)
    for m in vs.index:
        if m == baseline_name:
            continue
        cells = []
        for r in order:
            v = vs.loc[m, r]
            # กลุ่ม "วันนิ่ง" ที่ Naive ได้ MAE = 0 พอดีโดยนิยาม (ทำนาย 0 และ
            # ราคาไม่ขยับจริง) การหารด้วยศูนย์ไม่มีความหมาย -> แสดงเป็น n/a
            cells.append(f"{v:+17.2f}%" if np.isfinite(v) else f"{'n/a':>18s}")
        print(f"  {m:24s}" + "".join(cells))

    if not np.isfinite(vs.to_numpy()).all():
        print(f"  (n/a = กลุ่มที่ {baseline_name} ได้ MAE = 0 พอดีโดยนิยาม "
              f"เทียบเป็น % ไม่ได้)")
