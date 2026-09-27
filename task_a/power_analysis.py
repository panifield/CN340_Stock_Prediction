"""
power_analysis.py  — A1
=======================
ตอบคำถามที่สำคัญที่สุดของรายงาน:

    "ถ้าจริง ๆ แล้วมี edge เล็ก ๆ อยู่ เราจะตรวจจับมันเจอไหม
     ด้วยข้อมูลเท่าที่มี?"

ทำไมต้องมี
----------
null result มีสองแบบ ซึ่งมีน้ำหนักต่างกันมาก

  แบบอ่อน : "เราหาไม่เจอ"
            -> อาจแปลว่าหาไม่เก่ง หรือข้อมูลน้อยไป

  แบบแข็ง : "ต่อให้มี edge อยู่จริง ข้อมูลเท่าที่มีก็ตรวจจับไม่ได้
             และการจะตรวจจับได้ต้องใช้ข้อมูลมากกว่าที่มีอยู่บนโลก"
            -> ปิดข้อโต้แย้งได้หมด

ไฟล์นี้เปลี่ยนรายงานจากแบบอ่อนเป็นแบบแข็ง

สิ่งที่คำนวณ
-----------
  1. Minimum Detectable Effect (MDE) ที่ n ปัจจุบัน
     = edge ที่เล็กที่สุดที่เราจะตรวจจับเจอด้วย power 80%
  2. n ที่ต้องใช้ เพื่อตรวจจับ edge ขนาด 1pp / 2pp / 3pp / 5pp
  3. แปลง n เป็น "กี่ปี" ของข้อมูลรายวัน
  4. Power ที่เรามีจริง ณ ตอนนี้ สำหรับ edge แต่ละขนาด

ไม่แตะ test set (เป็นการคำนวณเชิงสถิติล้วน ไม่ใช้ label เลย)

วิธีใช้
    python power_analysis.py
    python power_analysis.py --n 1815 --observed 0.5129
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 245


# ---------------------------------------------------------------
def required_n(effect_pp: float, alpha: float = 0.05,
               power: float = 0.80, two_sided: bool = True) -> int:
    """
    จำนวนตัวอย่างที่ต้องใช้ เพื่อตรวจจับความต่างจาก 0.50 ขนาด effect_pp
    (หน่วย percentage point) ด้วย power ที่กำหนด

    สูตรสำหรับ one-sample proportion test:
        n = (z_alpha * sqrt(p0*(1-p0)) + z_beta * sqrt(p1*(1-p1)))^2 / (p1-p0)^2

    ใช้ p0 = 0.50 (การเดาสุ่ม) และ p1 = 0.50 + effect
    """
    p0 = 0.50
    delta = effect_pp / 100.0
    p1 = p0 + delta

    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_beta = stats.norm.ppf(power)

    numerator = (z_alpha * np.sqrt(p0 * (1 - p0))
                 + z_beta * np.sqrt(p1 * (1 - p1))) ** 2
    return int(np.ceil(numerator / delta ** 2))


def achieved_power(n: int, effect_pp: float, alpha: float = 0.05,
                   two_sided: bool = True) -> float:
    """
    power ที่เรามีจริงที่ n ปัจจุบัน สำหรับ edge ขนาด effect_pp

    power = P(ปฏิเสธ H0 | H1 เป็นจริง)
    """
    p0, delta = 0.50, effect_pp / 100.0
    p1 = p0 + delta

    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    se0 = np.sqrt(p0 * (1 - p0) / n)
    se1 = np.sqrt(p1 * (1 - p1) / n)

    critical = p0 + z_alpha * se0
    return float(1 - stats.norm.cdf((critical - p1) / se1))


def mde(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """
    Minimum Detectable Effect: edge ที่เล็กที่สุดที่ n นี้ตรวจจับได้
    หาโดยไล่ค่าจนกว่า power จะถึงเกณฑ์
    """
    for effect in np.arange(0.1, 20.0, 0.01):
        if achieved_power(n, effect, alpha) >= power:
            return float(effect)
    return float("nan")


def half_width(n: int, p: float = 0.50, z: float = 1.96) -> float:
    """ครึ่งความกว้างของ CI 95% ที่ n นี้ (หน่วย percentage point)"""
    return float(z * np.sqrt(p * (1 - p) / n) * 100)


# ---------------------------------------------------------------
def run(n: int, observed: float | None, outdir: str,
        alpha: float = 0.05, power: float = 0.80):
    print("=" * 78)
    print("  A1 — Power Analysis")
    print("=" * 78)
    print(f"  n ที่ใช้จริง (walk-forward OOF) = {n} วัน "
          f"({n / TRADING_DAYS_PER_YEAR:.1f} ปี)")
    print(f"  alpha = {alpha} (two-sided) | power เป้าหมาย = {power}")

    hw = half_width(n)
    print(f"\n  ความละเอียดที่ n นี้")
    print(f"    ครึ่งความกว้าง CI 95% = ±{hw:.2f} pp")
    print(f"    => ผลใด ๆ ในช่วง {50-hw:.2f}% ถึง {50+hw:.2f}% "
          f"แยกจากการโยนเหรียญไม่ออก")

    m = mde(n, alpha, power)
    print(f"\n  Minimum Detectable Effect (MDE)")
    print(f"    edge ที่เล็กที่สุดที่ตรวจจับได้ = {m:.2f} pp "
          f"(accuracy {50 + m:.2f}%)")
    print(f"    => edge ที่เล็กกว่านี้ 'มองไม่เห็น' ด้วยข้อมูลเท่าที่มี")

    # ---- ตารางหลัก ----
    rows = []
    for effect in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        need = required_n(effect, alpha, power)
        rows.append({
            "edge_pp": effect,
            "accuracy": 50 + effect,
            "n_required": need,
            "years_required": round(need / TRADING_DAYS_PER_YEAR, 1),
            "power_at_current_n": round(achieved_power(n, effect, alpha), 3),
            "detectable_now": "YES" if need <= n else "no",
        })
    table = pd.DataFrame(rows)

    print(f"\n{'='*78}")
    print("  ต้องใช้ข้อมูลเท่าไหร่ ถึงจะตรวจจับ edge แต่ละขนาดได้")
    print(f"{'='*78}")
    print(table.to_string(index=False))

    # ---- ตีความ ----
    print(f"\n{'='*78}")
    print("  การตีความ (เอาไปเขียนรายงานได้เลย)")
    print(f"{'='*78}")

    e2 = table[table.edge_pp == 2.0].iloc[0]
    e1 = table[table.edge_pp == 1.0].iloc[0]

    print(f"  - edge ขนาด 1 pp (accuracy 51%) ต้องใช้ {e1.n_required:,} วัน "
          f"= {e1.years_required} ปี")
    print(f"    power ที่เรามีตอนนี้ = {e1.power_at_current_n:.1%} "
          f"(ต่ำมาก แทบมองไม่เห็น)")
    print(f"  - edge ขนาด 2 pp (accuracy 52%) ต้องใช้ {e2.n_required:,} วัน "
          f"= {e2.years_required} ปี")
    print(f"    power ที่เรามีตอนนี้ = {e2.power_at_current_n:.1%}")
    print()
    print(f"  ข้อจำกัดเชิงโครงสร้าง")
    print(f"    หุ้นไทยเปิดซื้อขายมาไม่ถึง 50 ปี และ tick regime เปลี่ยน")
    print(f"    ทุกไม่กี่ปี ทำให้ข้อมูลที่ 'เทียบกันได้' มีอยู่จำกัดมาก")
    print(f"    => edge ขนาดเล็กกว่า {m:.1f} pp ตรวจจับไม่ได้ในทางหลักการ")
    print(f"       ไม่ใช่เพราะวิธีการเราไม่ดี แต่เพราะข้อมูลไม่พอโดยธรรมชาติ")

    if observed is not None:
        obs_pp = (observed - 0.50) * 100
        pw = achieved_power(n, abs(obs_pp), alpha) if obs_pp != 0 else alpha
        print(f"\n  เทียบกับผลจริงที่ได้ ({observed:.4f})")
        print(f"    edge ที่สังเกตได้ = {obs_pp:+.2f} pp")
        print(f"    ต้องใช้ n = {required_n(max(abs(obs_pp), 0.01)):,} วัน "
              f"= {required_n(max(abs(obs_pp), 0.01))/TRADING_DAYS_PER_YEAR:.1f} ปี "
              f"ถึงจะยืนยันว่าเป็นของจริง")
        print(f"    power ที่เรามี = {pw:.1%} "
              f"-> {'พอ' if pw >= 0.8 else 'ไม่พอ ยืนยันไม่ได้'}")

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "A1_power_analysis.csv")
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[csv] {path}")
    return table


def plot(n: int, outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    effects = np.arange(0.25, 8.0, 0.05)
    powers = [achieved_power(n, e) for e in effects]
    needs = [required_n(e) / TRADING_DAYS_PER_YEAR for e in effects]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(effects, powers, lw=2, color="#4c78a8")
    ax1.axhline(0.80, color="red", ls="--", lw=1.5, label="Target power = 0.80")
    ax1.axvline(mde(n), color="green", ls=":", lw=1.5,
                label=f"MDE = {mde(n):.2f} pp")
    ax1.set_xlabel("True edge (percentage points above 50%)")
    ax1.set_ylabel("Statistical power")
    ax1.set_title(f"Power we actually have at n = {n}")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(effects, needs, lw=2, color="#e45756")
    ax2.axhline(n / TRADING_DAYS_PER_YEAR, color="black", ls="--", lw=1.5,
                label=f"Data we have = {n/TRADING_DAYS_PER_YEAR:.1f} yr")
    ax2.set_yscale("log")
    ax2.set_xlabel("True edge (percentage points above 50%)")
    ax2.set_ylabel("Years of daily data required (log scale)")
    ax2.set_title("Data required to detect an edge (power = 0.80)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "A1_power_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1815,
                    help="n ของ walk-forward OOF")
    ap.add_argument("--observed", type=float, default=None,
                    help="accuracy ที่ดีที่สุดที่ได้ เช่น 0.5129")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--outdir", default="results/dev")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    run(args.n, args.observed, args.outdir, args.alpha, args.power)
    if not args.no_plot:
        plot(args.n, args.outdir)
