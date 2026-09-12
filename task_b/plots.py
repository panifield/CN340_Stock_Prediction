"""
plots.py — งาน B (ราคาปิด / return)
===================================
วาดกราฟจาก "ไฟล์ผลที่บันทึกไว้แล้ว" ใน results/

    cd task_b
    python plots.py                  # วาดจากผล val ล่าสุด (ค่าเริ่มต้น)
    python plots.py --stage test     # วาดจากผล test (หลังเปิด test แล้วเท่านั้น)

---------------------------------------------------------------------------
*** สคริปต์นี้ไม่เทรนโมเดล และไม่อ่านข้อมูลราคาดิบเลยแม้แต่ไฟล์เดียว ***

อ่านเฉพาะ csv ที่ main.py บันทึกไว้:
  {หุ้น}_taskB_{stage}_{n}feat_{เวลา}.csv        <- ตาราง metric
  {หุ้น}_taskB_preds_{stage}_{n}feat_{เวลา}.csv  <- คำทำนายรายวัน
  {หุ้น}_taskB_regime_{stage}_{n}feat_{เวลา}.csv <- error แยกสภาวะตลาด

ผลที่ตามมาซึ่งสำคัญกับ protocol ของงานนี้:
  1. รันกี่รอบก็ได้ ปรับกราฟกี่ครั้งก็ได้ โดย **ไม่ต้องรันโมเดลบน test ซ้ำ**
     (นี่คือเหตุผลที่ main.py บันทึกคำทำนายรายวันไว้ตั้งแต่รอบแรก)
  2. ไม่มีทางแตะ test โดยไม่ตั้งใจ เพราะไม่มีโค้ดส่วนไหนเปิดไฟล์ raw_data/

*** โมเดลที่นำมาวาดถูกเลือกจาก val เสมอ ***
แม้ตอนวาดกราฟของ test ก็ยังอ่านว่า "โมเดลไหนถูกเลือก" จากตารางผล **val**
ถ้าไปเลือกตัวที่ดีที่สุดบนตาราง test จะกลายเป็นการเลือกโมเดลหลังเห็น test
ซึ่งขัดกับ PRE_TEST_LOCK.md ข้อ 2
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # ไม่ต้องมีหน้าจอ เขียนเป็นไฟล์อย่างเดียว
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from config import TICKERS, OUTPUT_DIR
from models import ENSEMBLE_NAME

NAIVE = "Baseline: Naive (RW)"


def set_thai_font():
    """หาฟอนต์ที่วาดภาษาไทยได้ ไม่งั้นข้อความไทยจะกลายเป็นกล่องสี่เหลี่ยม"""
    prefer = ["Tahoma", "Leelawadee UI", "TH Sarabun New", "Angsana New",
              "Microsoft Sans Serif"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in prefer:
        if name in available:
            plt.rcParams["font.family"] = name
            return name
    print("[plots] เตือน: ไม่พบฟอนต์ไทย ข้อความไทยในกราฟอาจเป็นกล่องสี่เหลี่ยม")
    return None


def latest(pattern):
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, pattern)))
    return files[-1] if files else None


def _tag(ticker):
    return ticker.replace(".", "_").replace("^", "")


def load_stage(ticker, stage):
    """อ่านไฟล์ผลล่าสุดของ stage ที่ขอ -- คืน (metrics, preds, regime, stamp)"""
    t = _tag(ticker)
    m_path = latest(f"{t}_taskB_{stage}_*feat_*.csv")
    p_path = latest(f"{t}_taskB_preds_{stage}_*feat_*.csv")
    r_path = latest(f"{t}_taskB_regime_{stage}_*feat_*.csv")

    if m_path is None or p_path is None:
        raise FileNotFoundError(
            f"ไม่พบไฟล์ผลของ {ticker} stage={stage} ใน {OUTPUT_DIR}/\n"
            f"  ต้องรัน main.py ({'--dev' if stage == 'val' else 'ไม่ใส่ --dev'}) ก่อน"
        )

    metrics = pd.read_csv(m_path, index_col=0)
    preds = pd.read_csv(p_path, index_col=0, parse_dates=True)
    regime = pd.read_csv(r_path) if r_path else None
    stamp = os.path.basename(p_path).rsplit("_", 1)[-1].replace(".csv", "")
    return metrics, preds, regime, stamp


def selected_model(ticker):
    """
    โมเดลที่ถูกเลือกตาม protocol -- อ่านจากตารางผล **val** เสมอ

    ไม่รวม Ensemble เพราะตาม PRE_TEST_LOCK.md ให้เลือกจาก 3 โมเดลฐานเท่านั้น
    ส่วน Ensemble เป็นตัวเทียบที่รายงานคู่กันไปเสมอ
    """
    m_path = latest(f"{_tag(ticker)}_taskB_val_*feat_*.csv")
    if m_path is None:
        return None
    df = pd.read_csv(m_path, index_col=0)
    base = [i for i in df.index
            if not i.startswith("Baseline") and i != ENSEMBLE_NAME]
    return df.loc[base, "MAE_return"].idxmin()


# ---------------------------------------------------------------
# กราฟหลัก 4 ช่อง
# ---------------------------------------------------------------
def plot_main(ticker, stage, metrics, preds, model, stamp):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"งาน B — {ticker} ({stage} set, {len(preds)} วัน)  "
                 f"โมเดลที่เลือกจาก val = {model}", fontsize=13)

    prev = preds["prev_close"]
    y_true = preds["y_true_return"]
    y_pred = preds[model]

    # --- (1) ราคาจริง vs ราคาทำนาย ---
    ax = axes[0][0]
    ax.plot(preds.index, preds["close_true"], lw=1.3, label="ราคาจริง")
    ax.plot(preds.index, prev * (1 + y_pred), lw=1.0, label=f"ทำนายโดย {model}")
    ax.plot(preds.index, prev, lw=0.9, ls="--", label="Naive (= ราคาเมื่อวาน)")
    ax.set_title("ราคาปิด: จริง vs ทำนาย — สามเส้นทับกันแทบสนิท\n"
                 "นี่คือเหตุผลที่ R2_price สูงหลอก ๆ", fontsize=10)
    ax.set_ylabel("บาท")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- (2) error สะสม เทียบ Naive ---
    ax = axes[0][1]
    err_model = np.abs(prev * (y_true - y_pred)).cumsum()
    err_naive = np.abs(prev * y_true).cumsum()
    ax.plot(preds.index, err_model, lw=1.2, label=model)
    ax.plot(preds.index, err_naive, lw=1.2, ls="--", label="Naive (RW)")
    gap = float(err_naive.iloc[-1] - err_model.iloc[-1])
    ax.set_title(f"ค่าคลาดเคลื่อนสัมบูรณ์สะสม (บาท)\n"
                 f"ต่างกันตอนจบ {gap:+.2f} บาท จาก {err_naive.iloc[-1]:.0f} บาท "
                 f"({gap / err_naive.iloc[-1] * 100:+.2f}%)", fontsize=10)
    ax.set_ylabel("บาทสะสม")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- (3) scatter: ทำนาย vs จริง (ดูการปรับเทียบ) ---
    ax = axes[1][0]
    ax.scatter(y_true, y_pred, s=9, alpha=0.45, edgecolors="none")
    lim = float(np.max(np.abs(y_true))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], ls="--", lw=1,
            color="gray", label="y = x (ทำนายได้สมบูรณ์แบบ)")
    slope, intercept = np.polyfit(y_true.values, y_pred.values, 1)
    xs = np.linspace(-lim, lim, 50)
    # ความชันของเส้นถดถอย = Rho * StdRatio ตามนิยาม ไม่ใช่ Rho เฉย ๆ
    # ยิ่งแบนเทียบกับ y = x แปลว่าโมเดลยิ่งหดการทำนายเข้าหาค่าเฉลี่ย
    ax.plot(xs, intercept + slope * xs, lw=1.2, color="crimson",
            label=f"เส้นถดถอย (ความชัน {slope:.3f} = Rho x StdRatio)")
    rho = float(metrics.loc[model, "Rho"])
    sr = float(metrics.loc[model, "StdRatio"])
    ax.set_title(f"return ที่ทำนาย vs ที่เกิดจริง\n"
                 f"Rho = {rho:+.4f} · StdRatio = {sr:.4f} "
                 f"(ค่าที่เหมาะสมคือ StdRatio = Rho)", fontsize=10)
    ax.set_xlabel("return จริง")
    ax.set_ylabel("return ที่ทำนาย")
    ax.axhline(0, lw=0.5, color="black")
    ax.axvline(0, lw=0.5, color="black")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- (4) MAE_return ของทุกตัวทำนาย ---
    ax = axes[1][1]
    order = metrics["MAE_return"].sort_values(ascending=False)
    colors = []
    for name in order.index:
        if name == NAIVE:
            colors.append("crimson")
        elif name == model:
            colors.append("seagreen")
        elif name.startswith("Baseline"):
            colors.append("lightgray")
        else:
            colors.append("steelblue")
    ax.barh(range(len(order)), order.values, color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order.index, fontsize=8)
    ax.axvline(float(metrics.loc[NAIVE, "MAE_return"]), ls="--", lw=1,
               color="crimson")
    dm_p = metrics.loc[model, "DM_p"]
    ax.set_title(f"MAE_return ของทุกตัวทำนาย (เส้นประ = Naive)\n"
                 f"DM test ของ {model} เทียบ Naive: p = {dm_p:.3f}"
                 f"{' (เสมอกันเชิงสถิติ)' if dm_p > 0.05 else ''}", fontsize=10)
    ax.set_xlabel("MAE_return (ยิ่งน้อยยิ่งดี)")
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUTPUT_DIR, f"{_tag(ticker)}_taskB_{stage}_plots_{stamp}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------
# กราฟ error แยกตามสภาวะตลาด
# ---------------------------------------------------------------
def plot_regime(ticker, stage, regime, model, stamp):
    order = [r for r in regime["regime"].unique()]
    sub = regime[regime["model"].isin([model, ENSEMBLE_NAME])]
    pivot = (sub.pivot(index="regime", columns="model", values="vs_naive_pct")
             .reindex(order).dropna(how="all"))
    n_days = regime.drop_duplicates("regime").set_index("regime")["n_days"]

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.axhline(0, lw=1, color="black")
    ax.set_title(f"งาน B — {ticker} ({stage}): ดีกว่า Naive กี่ % ในแต่ละสภาวะตลาด\n"
                 f"ค่าบวก = คลาดเคลื่อนน้อยกว่า Naive", fontsize=11)
    ax.set_ylabel("% ที่ MAE ต่ำกว่า Naive")
    ax.set_xlabel("")
    ax.set_xticklabels([f"{r.split(' (')[0]}\n(n={n_days[r]})" for r in pivot.index],
                       rotation=0, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR,
                        f"{_tag(ticker)}_taskB_{stage}_regimeplot_{stamp}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(
        description="วาดกราฟจากไฟล์ผลใน results/ (ไม่เทรนโมเดลใหม่)")
    p.add_argument("--stage", choices=["val", "test"], default="val",
                   help="ชุดข้อมูลที่จะวาด (ค่าเริ่มต้น val)")
    p.add_argument("--model", default=None,
                   help="บังคับเลือกโมเดลเอง (ปกติอ่านจากตารางผล val)")
    args = p.parse_args()

    print("=" * 78)
    print(f"  งาน B : วาดกราฟจากผล {args.stage}")
    print("  *** อ่านจาก csv เท่านั้น ไม่เทรนโมเดล ไม่อ่านข้อมูลราคาดิบ ***")
    print("=" * 78)
    font = set_thai_font()
    if font:
        print(f"[plots] ใช้ฟอนต์ {font}")

    saved = []
    for ticker in TICKERS:
        metrics, preds, regime, stamp = load_stage(ticker, args.stage)
        model = args.model or selected_model(ticker)
        if model is None or model not in preds.columns:
            raise ValueError(
                f"ไม่รู้ว่าจะวาดโมเดลไหนสำหรับ {ticker} "
                f"(ไม่พบตารางผล val หรือไม่มีคอลัมน์ '{model}') ใช้ --model ระบุเอง"
            )
        print(f"\n  {ticker}: ใช้ผลรอบ {stamp} | โมเดลที่เลือกจาก val = {model}")
        saved.append(plot_main(ticker, args.stage, metrics, preds, model, stamp))
        if regime is not None:
            saved.append(plot_regime(ticker, args.stage, regime, model, stamp))

    print(f"\n[plots] บันทึกกราฟ {len(saved)} ไฟล์:")
    for s in saved:
        print(f"         {s}")
    print("\nเสร็จสิ้น")
    return saved


if __name__ == "__main__":
    main()
