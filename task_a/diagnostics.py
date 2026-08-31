"""
diagnostics.py — งาน A (คู่/คี่)
===============================
วิเคราะห์ข้อมูลก่อนเทรน  <-- ส่วนที่จะทำให้รายงานดูแข็งแรง

ต้องรันส่วนนี้ก่อนเสมอ และเอาผลไปใส่ในรายงาน
เพราะมันพิสูจน์ว่าเราเข้าใจข้อมูล ไม่ใช่แค่ยัดเข้าโมเดล

มี 3 การวิเคราะห์:
  1. tick_size_analysis   - ราคาลงท้ายด้วยอะไรบ้าง (สำคัญมากกับงานคู่/คี่)
  2. parity_analysis      - base rate + ความ "เหนียว" ของ parity
  3. leak_check           - ตรวจหา feature ที่น่าสงสัยว่า leak
"""

import numpy as np
import pandas as pd

from config import LEAK_CORR_THRESHOLD


def tick_size_analysis(df, verbose=True):
    """
    ดูว่าราคาปิดลงท้ายด้วยสตางค์อะไรบ้าง

    ทำไมสำคัญ: SET มี tick size บังคับตามช่วงราคา
       ราคา 10-25 บาท   ขยับทีละ 0.10
       ราคา 25-100 บาท  ขยับทีละ 0.25
       ราคา 100-200 บาท ขยับทีละ 0.50
       ราคา 200-400 บาท ขยับทีละ 1.00
    (ตัวเลขนี้ควรเช็คกับประกาศตลาดหลักทรัพย์อีกรอบ เพราะมีการปรับเป็นระยะ)

    ถ้าหุ้นอยู่ในช่วง tick = 0.50 ราคาจะลงท้าย .00 หรือ .50 เท่านั้น
    """
    close = df["Close"].dropna()
    satang = np.rint(close * 100).astype(int) % 100
    counts = pd.Series(satang).value_counts().sort_index()
    pct = (counts / len(satang) * 100).round(2)

    if verbose:
        print("\n--- การวิเคราะห์ Tick Size ---")
        print(f"ช่วงราคา: {close.min():.2f} - {close.max():.2f} บาท "
              f"(เฉลี่ย {close.mean():.2f})")
        print(f"จำนวนค่าสตางค์ที่พบ: {len(counts)} แบบ")
        top = pct.sort_values(ascending=False).head(8)
        print("สตางค์ที่พบบ่อยสุด:")
        for s, p in top.items():
            print(f"   .{s:02d} -> {p:5.2f}%")
        if len(counts) <= 4:
            print("!! เตือน: ราคาติดอยู่บน tick grid แคบมาก")
            print("   งานคู่/คี่อาจได้ accuracy สูงหลอกๆ")

    return pct


def parity_analysis(targets, verbose=True):
    """
    วิเคราะห์งาน A: คู่/คี่

    2 ตัวเลขที่ต้องดู:
      1. base rate      - คู่กี่ % คี่กี่ %  (ควรใกล้ 50/50)
      2. flip rate      - parity เปลี่ยนจากเมื่อวานกี่ %
                          ยิ่งต่ำ = ยิ่ง "เหนียว" = persistence baseline ยิ่งแข็ง
    """
    y = targets["y_parity"].dropna()
    int_price = targets["int_price"].dropna()

    even_pct = float((y == 0).mean() * 100)
    odd_pct = float((y == 1).mean() * 100)

    flips = (y != y.shift(1)).iloc[1:]
    flip_rate = float(flips.mean() * 100)

    # จำนวนเต็มบาทเปลี่ยนกี่ % ของวัน
    int_changed = (int_price.diff() != 0).iloc[1:]
    int_change_rate = float(int_changed.mean() * 100)
    avg_jump = float(int_price.diff().abs().mean())

    stats = {
        "even_pct": even_pct,
        "odd_pct": odd_pct,
        "flip_rate": flip_rate,
        "int_change_rate": int_change_rate,
        "avg_int_jump": avg_jump,
    }

    if verbose:
        print("\n--- การวิเคราะห์งาน A (คู่/คี่) ---")
        print(f"Base rate         : คู่ {even_pct:.2f}% / คี่ {odd_pct:.2f}%")
        print(f"Majority baseline : {max(even_pct, odd_pct):.2f}%")
        print(f"Flip rate         : {flip_rate:.2f}% "
              f"(parity เปลี่ยนจากเมื่อวาน)")
        print(f"=> Persistence baseline : {100 - flip_rate:.2f}%  "
              f"<-- ตัวนี้คือคู่แข่งตัวจริง")
        print(f"เลขจำนวนเต็มบาทเปลี่ยน : {int_change_rate:.2f}% ของวัน")
        print(f"กระโดดเฉลี่ย            : {avg_jump:.2f} บาท/วัน")

        if flip_rate < 35:
            print("!! parity เหนียวมาก (ราคาขยับน้อยกว่า 1 บาท/วัน)")
            print("   โมเดลจะได้ accuracy สูง แต่ไม่ใช่การทำนายตลาด")
            print("   ต้องเทียบกับ persistence baseline ให้ชัด")
        elif flip_rate > 45:
            print("=> parity พลิกเกือบสุ่ม -> คาดว่าโมเดลจะได้ราว 50%")

    return stats


def leak_check(X, y_parity, verbose=True):
    """
    ตรวจหา feature ที่น่าสงสัยว่า leak

    วิธี: ดู correlation ระหว่าง feature กับ target parity
    ถ้า correlate สูงเกินไป -> น่าสงสัยว่ามีข้อมูลอนาคตหลุดเข้ามา
    """
    suspicious = []

    common = X.index.intersection(y_parity.dropna().index)
    Xc = X.loc[common]
    yp = y_parity.loc[common]

    if verbose:
        print("\n--- ตรวจสอบ Data Leakage ---")

    for col in Xc.columns:
        s = Xc[col]
        if s.nunique() <= 1:
            continue
        c = abs(s.corr(yp))
        c = c if pd.notna(c) else 0
        if c > LEAK_CORR_THRESHOLD:
            suspicious.append((col, c))

    if verbose:
        if suspicious:
            print(f"!! พบ {len(suspicious)} feature น่าสงสัย "
                  f"(|corr| > {LEAK_CORR_THRESHOLD}):")
            for col, c in sorted(suspicious, key=lambda x: -x[1]):
                print(f"   {col:35s} |corr| = {c:.3f}")
            print("   ตรวจสอบว่า feature พวกนี้ shift แล้วจริงหรือไม่")
        else:
            print(f"ผ่าน: ไม่มี feature ไหน correlate กับ parity "
                  f"เกิน {LEAK_CORR_THRESHOLD}")

    return suspicious


def run_all_diagnostics(df, targets, X, verbose=True):
    """รันการวิเคราะห์ทั้งหมดรวดเดียว"""
    print("\n" + "#" * 78)
    print("#  การวิเคราะห์ข้อมูลก่อนเทรน (เอาผลส่วนนี้ใส่รายงานด้วย)")
    print("#" * 78)

    out = {}
    out["tick"] = tick_size_analysis(df, verbose)
    out["parity"] = parity_analysis(targets, verbose)
    out["leak"] = leak_check(X, targets["y_parity"], verbose)
    return out
