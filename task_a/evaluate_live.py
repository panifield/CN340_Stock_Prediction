"""
evaluate_live.py — เทียบ prediction log จริงของ task_a และ task_a2 กับราคาปิดทางการ
==================================================================================

    python evaluate_live.py

เขียน task_a/results/outcomes.csv และ task_a2/results/outcomes.csv แล้วพิมพ์สรุป
ไม่นับแถว is_dry_run · วันที่ยังไม่มีราคาปิดใน task_b/raw_data/ = pending (ไม่นับ)

ผลจริง: parity ของราคาปิดใน task_b/raw_data/ (ไฟล์ที่ task_b ตรวจ OHLC ทุกวัน)
baseline A : คู่/คี่เหมือนวันก่อน (prev_parity ใน log)
baseline A2: คู่/คี่เหมือนราคา ณ 16:00 (parity_1600 ใน log)

bar15_settled (เฉพาะ A2): price_1600 ใน log (แท่ง 15:00 ที่ Yahoo ให้ตอนทาย ~16:02)
ตรงกับแท่ง 15:00 ฉบับสุดท้ายที่ดึงตอนนี้ไหม -- แท่งนี้ยังเปลี่ยนได้หลัง 16:02 (พบต่าง
1 tick ใน 4 จาก 8 วัน) ซึ่งพอจะพลิกคู่/คี่ วันที่ False จึงควรแยกดูตอนวัดผล
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from data_adapter import load_investing_csv
from rounding import parity, to_int_baht

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
RAW_DATA_DIR = ROOT / "task_b" / "raw_data"
A_LOG = BASE_DIR / "results" / "prediction_log.csv"
A2_LOG = ROOT / "task_a2" / "results" / "prediction_log.csv"
A_OUT = BASE_DIR / "results" / "outcomes.csv"
A2_OUT = ROOT / "task_a2" / "results" / "outcomes.csv"


def official_close(ticker):
    path = RAW_DATA_DIR / f"{ticker.replace('.BK', '')}_10Y_Cleaned.csv"
    return load_investing_csv(str(path), verbose=False)["Close"]


def final_bar15(ticker):
    """Close ของแท่ง 1h เวลา 15:00 (เวลาไทย) ทุกวันที่ Yahoo ยังมี -> Series index=date"""
    df = yf.download(ticker, interval="1h", period="730d",
                     auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = df.index if df.index.tz is not None else df.index.tz_localize("UTC")
    df.index = idx.tz_convert("Asia/Bangkok")
    bars = df[df.index.hour == 15]["Close"]
    return pd.Series(bars.values, index=pd.to_datetime(bars.index.date))


def score(log, baseline_col):
    """เติม actual_close / actual_parity / correct / baseline_correct / status"""
    log = log[log["is_dry_run"].astype(str).str.lower() != "true"].copy()
    log["target_date"] = pd.to_datetime(log["target_date"])
    closes = {t: official_close(t) for t in log["ticker"].unique()}
    log["actual_close"] = [closes[t].get(d, np.nan)
                           for t, d in zip(log["ticker"], log["target_date"])]
    done = log["actual_close"].notna()
    log["status"] = np.where(done, "scored", "pending")
    log["actual_parity"] = np.nan
    log.loc[done, "actual_parity"] = parity(to_int_baht(log.loc[done, "actual_close"])).values
    log["correct"] = np.where(done, log["predicted_parity"] == log["actual_parity"], np.nan)
    log["baseline_correct"] = np.where(done, log[baseline_col] == log["actual_parity"], np.nan)
    return log


def summarize(name, out, baseline_label, split_col=None):
    s = out[out["status"] == "scored"]
    print(f"\n=== {name} === scored {s['target_date'].nunique()} วัน · "
          f"pending {(out['status'] == 'pending').sum()} แถว")
    if s.empty:
        return

    def line(label, df):
        days = df.drop_duplicates(["ticker", "target_date"])
        print(f"  {label:<22} ถูก {int(df['correct'].sum())}/{len(df)} "
              f"({df['correct'].mean():.0%}) · baseline {baseline_label} "
              f"{int(days['baseline_correct'].sum())}/{len(days)} "
              f"({days['baseline_correct'].mean():.0%})")

    for model, g in s.groupby("model"):
        line(model, g)
    line("รวมทุกโมเดล", s)
    if split_col:
        for flag, g in s.groupby(s[split_col].astype(str)):
            line(f"{split_col}={flag}", g)


def main():
    a = score(pd.read_csv(A_LOG), "prev_parity")
    a.to_csv(A_OUT, index=False)
    summarize("Task A (คู่/คี่ วันถัดไป)", a, "เหมือนเมื่อวาน")

    a2 = score(pd.read_csv(A2_LOG), "parity_1600")
    bar15 = {t: final_bar15(t) for t in a2["ticker"].unique()}
    final = [bar15[t].get(d, np.nan) for t, d in zip(a2["ticker"], a2["target_date"])]
    a2["bar15_final"] = final
    a2["bar15_settled"] = [None if pd.isna(f) else bool(np.isclose(p, f))
                           for p, f in zip(a2["price_1600"], final)]
    a2.to_csv(A2_OUT, index=False)
    summarize("Task A2 (คู่/คี่ รอบ 16:00)", a2, "เหมือนราคา 16:00",
              split_col="bar15_settled")

    print(f"\nเขียน {A_OUT.relative_to(ROOT)} และ {A2_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
