"""
evaluate_live.py — เทียบ prediction log จริงของ task_a และ task_a2 กับราคาปิดทางการ
==================================================================================

    python evaluate_live.py

เขียน task_a/results/outcomes.csv และ task_a2/results/outcomes.csv แล้วพิมพ์สรุป
ไม่นับแถว is_dry_run · วันที่ยังไม่มีราคาปิดใน task_b/raw_data/ = pending (ไม่นับ)

ผลจริง: parity ของราคาปิดใน task_b/raw_data/ (ไฟล์ที่ task_b ตรวจ OHLC ทุกวัน)
baseline A : คู่/คี่เหมือนวันก่อน (prev_parity ใน log)
baseline A2: คู่/คี่เหมือนราคา ณ 16:00 (parity_1600 ใน log)

label 2 แบบ (2026-10-24): แถวที่ log มี label_rule = "floor" ทายด้วย rounding.parity_label
  actual_parity                  = กฎปัดเดิม (.50 ปัดขึ้น) คำนวณทุกแถวไว้เทียบ
  correct / baseline_correct     = กฎเดิม เฉพาะแถวเดิม (ค่าในแถวเก่าเหมือนเดิมทุกตัว)
  *_floor                        = floor เฉพาะแถว floor (แถวเก่าว่าง)
H1 (เฉพาะ A2, แถว floor): h1_correct = h1_predicted_parity == actual_parity_floor เฉพาะวันที่
  h1_status = predict · coverage = สัดส่วนวัน-หุ้นที่ไม่ abstain

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
from rounding import parity, parity_label, to_int_baht

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

    rule = log["label_rule"] if "label_rule" in log else pd.Series(np.nan, index=log.index)
    floor_rows = rule.astype(str) == "floor"
    old, new = done & ~floor_rows, done & floor_rows
    log["correct"] = np.where(old, log["predicted_parity"] == log["actual_parity"], np.nan)
    log["baseline_correct"] = np.where(old, log[baseline_col] == log["actual_parity"], np.nan)

    log["actual_parity_floor"] = np.nan
    log.loc[new, "actual_parity_floor"] = parity_label(log.loc[new, "actual_close"]).values
    log["correct_floor"] = np.where(new, log["predicted_parity"] == log["actual_parity_floor"], np.nan)
    log["baseline_correct_floor"] = np.where(
        new, log[baseline_col] == log["actual_parity_floor"], np.nan)
    if "h1_status" in log:
        h1 = new & (log["h1_status"] == "predict")
        log["h1_correct"] = np.where(
            h1, log["h1_predicted_parity"] == log["actual_parity_floor"], np.nan)
    return log


def write_outcomes(df, path):
    """คอลัมน์ที่ไฟล์เดิมมีอยู่แล้วคงลำดับเดิมไว้ข้างหน้า คอลัมน์ใหม่ต่อท้าย (แถวเก่าเป็นค่าว่าง)"""
    first = pd.read_csv(path, nrows=0).columns.tolist() if path.exists() else []
    first = [c for c in first if c in df.columns]
    df[first + [c for c in df.columns if c not in first]].to_csv(path, index=False)


def summarize(name, out, baseline_label, split_col=None):
    print(f"\n=== {name} === pending {(out['status'] == 'pending').sum()} แถว")
    for title, c, b in (("label เดิม (ปัด .50 ขึ้น)", "correct", "baseline_correct"),
                        ("label floor", "correct_floor", "baseline_correct_floor")):
        s = out[(out["status"] == "scored") & out[c].notna()]
        print(f"  --- {title}: scored {s['target_date'].nunique()} วัน ---")
        if s.empty:
            print("  (ยังไม่มี)")
            continue

        def line(label, df):
            days = df.drop_duplicates(["ticker", "target_date"])
            print(f"  {label:<22} ถูก {int(df[c].sum())}/{len(df)} "
                  f"({df[c].mean():.0%}) · baseline {baseline_label} "
                  f"{int(days[b].sum())}/{len(days)} ({days[b].mean():.0%})")

        for model, g in s.groupby("model"):
            line(model, g)
        ml = s[~s["model"].str.startswith("Baseline")]
        line("รวม ML ทุกโมเดล", ml)
        if split_col:
            for flag, g in ml.groupby(ml[split_col].astype(str)):
                line(f"{split_col}={flag}", g)


def summarize_h1(out):
    """กฎ H1 นับต่อวัน-หุ้น (ทุกแถวของวันเดียวกันมีค่า H1 ชุดเดียวกัน)"""
    if "h1_status" not in out:
        return
    days = out[(out["status"] == "scored") & out["actual_parity_floor"].notna()] \
        .drop_duplicates(["ticker", "target_date"])
    print("  --- กฎ H1 (ทายเฉพาะ tick 0.50, label floor) ---")
    if days.empty:
        print("  (ยังไม่มี)")
        return
    pred = days[days["h1_status"] == "predict"]
    msg = f"  coverage {len(pred)}/{len(days)} วัน-หุ้น ({len(pred) / len(days):.0%})"
    if len(pred):
        msg += f" · ถูก {int(pred['h1_correct'].sum())}/{len(pred)} ({pred['h1_correct'].mean():.0%})"
    print(msg)
    if not len(pred):
        print(H1_ABSTAIN_NOTE)


# coverage = 0: บอกให้ชัดว่า log จริงยังไม่ใช่หลักฐานของ H1 (ตัวเลขย้อนหลังจาก docs/floor_label_followup.md)
H1_ABSTAIN_NOTE = (
    "  หมายเหตุ: H1 abstain ทุกวัน เพราะราคา ณ 16:00 ไม่อยู่ช่วง tick 0.50 บาท (100-200 บาท)\n"
    "  -- h1_predicted_parity จึงว่าง ตราบใดที่ราคายังเกิน 200 บาท log จริงจึงยังยืนยัน H1 ไม่ได้\n"
    "  หลักฐานของ H1 มาจากข้อมูลย้อนหลังเท่านั้น (KBANK ช่วง 100-200 บาท: 62.5% ชุดพัฒนา n=424,\n"
    "  63.9% holdout n=61)"
)


def main():
    a = score(pd.read_csv(A_LOG), "prev_parity")
    write_outcomes(a, A_OUT)
    summarize("Task A (คู่/คี่ วันถัดไป)", a, "เหมือนเมื่อวาน")

    a2 = score(pd.read_csv(A2_LOG), "parity_1600")
    bar15 = {t: final_bar15(t) for t in a2["ticker"].unique()}
    final = [bar15[t].get(d, np.nan) for t, d in zip(a2["ticker"], a2["target_date"])]
    a2["bar15_final"] = final
    a2["bar15_settled"] = [None if pd.isna(f) else bool(np.isclose(p, f))
                           for p, f in zip(a2["price_1600"], final)]
    write_outcomes(a2, A2_OUT)
    summarize("Task A2 (คู่/คี่ รอบ 16:00)", a2, "เหมือนราคา 16:00",
              split_col="bar15_settled")
    summarize_h1(a2)

    print(f"\nเขียน {A_OUT.relative_to(ROOT)} และ {A2_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
