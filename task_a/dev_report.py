"""
dev_report.py  — แก้ข้อ 8, 9, 10, 13
====================================
สร้างตารางทั้งหมดที่รายงานยังขาด โดยคำนวณบน DEV เท่านั้น

*** ไฟล์นี้ไม่แตะ test set ***
ทุกตัวเลขตัดที่ dev_cutoff() ซึ่งคำนวณจากสัดส่วนใน config
มีการ assert ยืนยันขอบเขตไว้ด้วย

ตารางที่ได้
  1. baselines      Majority / Persistence / flip rate   (ข้อ 8)
  2. tick_regime    แยกผลตาม tick regime + digit clustering (ข้อ 9)
  3. sensitivity    เปรียบเทียบกฎการปัด gt vs gte        (ข้อ 10)
  4. yearly         flip rate รายปี ดูความเสถียร          (ข้อ 13)

ทำไมทุกอย่างต้องอยู่บน dev
--------------------------
ตัวเลขพวกนี้ใช้ประกอบการตัดสินใจ (เลือกกฎการปัด / ตีความว่ามี regime effect
ไหม) ถ้าคำนวณบน test แล้วเอามาตัดสินใจ = peeking
พอสรุปทุกอย่างเสร็จแล้วค่อยรัน pipeline เต็มครั้งเดียวบน test

วิธีใช้
    python dev_report.py
    python dev_report.py --outdir results/dev --plots
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from data_adapter import load_investing_csv
from tick_utils import (
    DEV_RATIO, dev_cutoff, tick_size, to_int_baht, parity, resolve_data_path,
)

WARMUP = 21          # ปรับได้ด้วย --warmup (ตั้ง 0 ให้ตรงกับ main.py ที่ใช้ imputer)


# ---------------------------------------------------------------
def prepare_dev(df: pd.DataFrame, mode: str = "gt",
                warmup: int | None = None) -> pd.DataFrame:
    """
    ตัดเอาเฉพาะส่วน dev แล้วสร้างคอลัมน์ที่ต้องใช้

    ตัด warmup ออกก่อนเสมอ เพื่อให้ขอบ dev/test ตรงกับตอนที่ pipeline
    หลักสร้าง feature แล้ว dropna
    """
    warmup = WARMUP if warmup is None else warmup
    usable = df.iloc[warmup:].copy()
    cut = dev_cutoff(len(usable))
    dev = usable.iloc[:cut].copy()

    assert len(dev) == cut, "ขนาด dev ไม่ตรงกับที่คำนวณ"
    assert dev.index[-1] < usable.index[cut], "dev ล้ำเข้า test"

    close = dev["Close"]
    dev["tick"] = tick_size(close)
    dev["int_price"] = to_int_baht(close, mode=mode)
    dev["y"] = parity(dev["int_price"])
    dev["prev_y"] = dev["y"].shift(1)
    dev["flip"] = (dev["y"] != dev["prev_y"]).astype(float)
    dev.loc[dev.index[0], "flip"] = np.nan
    return dev


def _wilson(k: int, n: int, z: float = 1.96):
    """ช่วงความเชื่อมั่น Wilson — แม่นกว่า normal approx เมื่อ n ไม่ใหญ่"""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (centre - half, centre + half)


# ---------------------------------------------------------------
# ตารางที่ 1 — Baselines  (ข้อ 8)
# ---------------------------------------------------------------
def table_baselines(dev: pd.DataFrame, name: str) -> dict:
    y = dev["y"].dropna()
    flip = dev["flip"].dropna()

    even = float((y == 0).mean())
    majority = max(even, 1 - even)
    flip_rate = float(flip.mean())
    persistence = 1 - flip_rate

    n = len(flip)
    lo_m, hi_m = _wilson(int(round(majority * len(y))), len(y))
    lo_p, hi_p = _wilson(int(round(persistence * n)), n)

    int_changed = float((dev["int_price"].diff() != 0).iloc[1:].mean())

    return {
        "ticker": name,
        "n_dev": len(y),
        "even_pct": even * 100,
        "odd_pct": (1 - even) * 100,
        "majority_baseline": majority * 100,
        "majority_ci": f"[{lo_m*100:.2f}, {hi_m*100:.2f}]",
        "flip_rate": flip_rate * 100,
        "persistence_baseline": persistence * 100,
        "persistence_ci": f"[{lo_p*100:.2f}, {hi_p*100:.2f}]",
        "int_price_changed_pct": int_changed * 100,
    }


# ---------------------------------------------------------------
# ตารางที่ 2 — แยกตาม tick regime  (ข้อ 9)
# ---------------------------------------------------------------
def table_tick_regime(dev: pd.DataFrame, name: str, min_n: int = 50):
    rows = []
    for tk, g in dev.groupby("tick"):
        g = g.dropna(subset=["flip"])
        if len(g) < min_n:
            continue
        even = float((g["y"] == 0).mean())
        flip_rate = float(g["flip"].mean())
        lo, hi = _wilson(int((g["y"] == 0).sum()), len(g))
        rows.append({
            "ticker": name,
            "tick": tk,
            "n": len(g),
            "start": g.index.min().date(),
            "end": g.index.max().date(),
            "even_pct": even * 100,
            "even_ci": f"[{lo*100:.2f}, {hi*100:.2f}]",
            "majority_baseline": max(even, 1 - even) * 100,
            "flip_rate": flip_rate * 100,
            "persistence": (1 - flip_rate) * 100,
        })
    return rows


def table_digit_clustering(dev: pd.DataFrame, name: str):
    """
    การกระจายเลขหลักหน่วยในช่วง tick = 1.00 (ราคาเป็นจำนวนเต็มบาท)

    ถ้าเลขคู่มากกว่า 50% อย่างมีนัยสำคัญ แปลว่า accuracy ที่เกิน 50%
    มาจาก price clustering ของโครงสร้างตลาด ไม่ใช่จากการทำนาย
    """
    sub = dev[dev["tick"] == 1.00]
    if len(sub) < 100:
        return None

    digits = np.rint(sub["Close"]).astype(int) % 10
    dist = (digits.value_counts(normalize=True).sort_index() * 100).round(2)
    even = float((np.rint(sub["Close"]).astype(int) % 2 == 0).mean())
    lo, hi = _wilson(int((np.rint(sub["Close"]).astype(int) % 2 == 0).sum()),
                     len(sub))

    return {
        "ticker": name,
        "n": len(sub),
        "even_pct": even * 100,
        "even_ci": f"[{lo*100:.2f}, {hi*100:.2f}]",
        "significant": "YES" if lo > 0.5 else "no",
        "round_0_or_5_pct": float(digits.isin([0, 5]).mean()) * 100,
        "digit_dist": dist.to_dict(),
    }


# ---------------------------------------------------------------
# ตารางที่ 3 — Sensitivity gt vs gte  (ข้อ 10)
# ---------------------------------------------------------------
def table_rounding_sensitivity(df: pd.DataFrame, name: str,
                               warmup: int | None = None):
    """
    กฎการปัดมีผลใหญ่เพราะราคาลงท้าย .50 มีสัดส่วนสูง
    ต้องยืนยันกับอาจารย์ว่าใช้กฎไหน แล้วเลือกอันเดียว
    ตารางนี้ทำบน dev เพื่อประกอบการตัดสินใจ ไม่ใช่เพื่อเลือกอันที่ได้ผลดีกว่า
    """
    rows = []
    for mode in ["gt", "gte"]:
        dev = prepare_dev(df, mode=mode, warmup=warmup)
        base = table_baselines(dev, name)
        satang = (np.rint(dev["Close"] * 100).astype(int) % 100)
        rows.append({
            "ticker": name,
            "round_mode": mode,
            "rule": "เศษ > 0.50 ปัดขึ้น" if mode == "gt"
                    else "เศษ >= 0.50 ปัดขึ้น",
            "pct_close_at_.50": float((satang == 50).mean()) * 100,
            "even_pct": base["even_pct"],
            "majority_baseline": base["majority_baseline"],
            "persistence_baseline": base["persistence_baseline"],
        })

    delta = abs(rows[0]["majority_baseline"] - rows[1]["majority_baseline"])
    for r in rows:
        r["swing_pp"] = round(delta, 2)
    return rows


# ---------------------------------------------------------------
# ตารางที่ 4 — flip rate รายปี  (ข้อ 13)
# ---------------------------------------------------------------
def table_yearly(dev: pd.DataFrame, name: str):
    rows = []
    valid = dev.dropna(subset=["flip"])
    for year, g in valid.groupby(valid.index.year):
        if len(g) < 30:
            continue
        rows.append({
            "ticker": name,
            "year": int(year),
            "n": len(g),
            "even_pct": float((g["y"] == 0).mean()) * 100,
            "flip_rate": float(g["flip"].mean()) * 100,
            "persistence": (1 - float(g["flip"].mean())) * 100,
        })
    return rows


# ---------------------------------------------------------------
def make_plots(regime_rows, yearly_rows, outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yr = pd.DataFrame(yearly_rows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for ticker, g in yr.groupby("ticker"):
        ax.plot(g["year"], g["flip_rate"], marker="o", label=ticker)
    ax.axhline(50, color="black", ls=":", lw=1, label="Random = 50%")
    ax.set_xlabel("Year")
    ax.set_ylabel("Flip rate (%)")
    ax.set_title("Parity flip rate by year (dev set only)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "flip_rate_by_year.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")

    rg = pd.DataFrame(regime_rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"{r.ticker}\ntick={r.tick}" for r in rg.itertuples()]
    ax.bar(labels, rg["even_pct"], color="#4c78a8")
    ax.axhline(50, color="red", ls="--", lw=1.5, label="Random = 50%")
    ax.set_ylabel("Even close (%)")
    ax.set_title("Even-price rate by tick regime (dev set only)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "even_rate_by_regime.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


def show(title: str, rows, outdir: str, fname: str):
    frame = pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame([rows])
    print(f"\n{'='*78}\n  {title}\n{'='*78}")
    print(frame.to_string(index=False))
    path = os.path.join(outdir, fname)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*",
                    default=["KBANK_10Y_Cleaned.csv", "ADVANC_10Y_Cleaned.csv"])
    ap.add_argument("--outdir", default="results/dev")
    ap.add_argument("--mode", default="gt", choices=["gt", "gte"],
                    help="กฎการปัดหลักที่ใช้ในตารางอื่น")
    ap.add_argument("--date-format", default="%m/%d/%Y")
    ap.add_argument("--warmup", type=int, default=WARMUP,
                    help="แถวที่ตัดหัวทิ้ง; ใช้ 0 ให้ตรงกับ main.py")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("#" * 78)
    print("#  DEV REPORT — ทุกตัวเลขคำนวณบน dev เท่านั้น ไม่แตะ test set")
    print(f"#  สัดส่วน dev = {DEV_RATIO:.2f} | warmup = {args.warmup} แถว")
    print(f"#  กฎการปัดหลัก = '{args.mode}'")
    print("#" * 78)

    baselines, regimes, clusters, sensitivity, yearly = [], [], [], [], []

    for path in args.files:
        name = os.path.basename(path).split("_")[0]
        path = resolve_data_path(path)
        frame = load_investing_csv(path, date_format=args.date_format,
                                   verbose=False)
        dev = prepare_dev(frame, mode=args.mode, warmup=args.warmup)

        print(f"\n[{name}] dev = {len(dev)} แถว "
              f"({dev.index[0].date()} -> {dev.index[-1].date()})")

        baselines.append(table_baselines(dev, name))
        regimes.extend(table_tick_regime(dev, name))
        c = table_digit_clustering(dev, name)
        if c:
            clusters.append(c)
        sensitivity.extend(table_rounding_sensitivity(frame, name,
                                                      args.warmup))
        yearly.extend(table_yearly(dev, name))

    show("ตารางที่ 1 — Baselines (ข้อ 8)  << ใส่หน้าแรกของผลลัพธ์",
         baselines, args.outdir, "table1_baselines.csv")

    show("ตารางที่ 2 — แยกตาม tick regime (ข้อ 9)",
         regimes, args.outdir, "table2_tick_regime.csv")

    if clusters:
        show("ตารางที่ 2b — Digit clustering ที่ tick = 1.00 (ข้อ 9)",
             [{k: v for k, v in c.items() if k != "digit_dist"}
              for c in clusters],
             args.outdir, "table2b_digit_clustering.csv")
        for c in clusters:
            print(f"\n  {c['ticker']} การกระจายเลขหลักหน่วย (%):")
            print(f"    {c['digit_dist']}")

    show("ตารางที่ 3 — Sensitivity กฎการปัด gt vs gte (ข้อ 10)",
         sensitivity, args.outdir, "table3_rounding_sensitivity.csv")

    show("ตารางที่ 4 — flip rate รายปี (ข้อ 13)",
         yearly, args.outdir, "table4_yearly.csv")

    if args.plots:
        make_plots(regimes, yearly, args.outdir)

    print(f"\n{'='*78}")
    print("  ข้อสังเกตสำหรับเขียนรายงาน")
    print(f"{'='*78}")
    for b in baselines:
        print(f"  {b['ticker']}: majority = {b['majority_baseline']:.2f}% "
              f"| persistence = {b['persistence_baseline']:.2f}%")
        print(f"    -> โมเดลต้องชนะตัวที่สูงกว่านี้ถึงจะมีความหมาย")
    for c in clusters:
        if c["significant"] == "YES":
            print(f"  {c['ticker']} ที่ tick=1.00: เลขคู่ {c['even_pct']:.2f}% "
                  f"CI {c['even_ci']} -> สูงกว่า 50% อย่างมีนัยสำคัญ")
            print(f"    -> 'ทายคู่ตลอด' ในช่วงนี้ได้ {c['even_pct']:.2f}% "
                  f"ซึ่งมาจาก market microstructure ไม่ใช่การทำนาย")
    print(f"\n  ไฟล์ csv ทั้งหมดอยู่ใน {args.outdir}/")
    print("  *** ยังไม่มีตัวเลขไหนมาจาก test set ***")