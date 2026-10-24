"""
build_daily_comparison.py — รวมคำทำนาย (log จริง + backfill) เทียบกับผลจริงรายวัน เป็น CSV ไฟล์เดียว
=====================================================================================================

    .venv\\Scripts\\python.exe backfill\\build_daily_comparison.py
    .venv\\Scripts\\python.exe backfill\\build_daily_comparison.py --start 2026-09-28 --end 2026-10-22

1 แถว = 1 คำทำนาย (วัน x หุ้น x ชุดงาน x โมเดล) · ไม่แก้ไฟล์ log/ผลใดเลย อ่านอย่างเดียว

แหล่งคำทำนาย
  live     : task_a|task_a2|task_c|task_b results/prediction_log*.csv (ไม่นับ is_dry_run)
  backfill : backfill/results/*.csv -- เอาเฉพาะแถวที่ log จริงไม่มี key เดียวกัน (log จริงมาก่อนเสมอ)
             backfill ทำนายย้อนหลังหลังรู้ผลแล้ว (README.md ในโฟลเดอร์นี้) -- แยกด้วยคอลัมน์ source
  ไม่ใช้ validation/ (ทับวันที่มี log จริง) · results_floor/ (label กฎ floor ของ branch อื่น) · experiments/

ผลจริง: ราคาปิดทางการใน task_b/raw_data/*_10Y_Cleaned.csv (แหล่งเดียวกับ evaluate_live.py / record_outcomes.py)
  A  (next_day) คู่/คี่ของราคาปิด · correct = ทายตรง · baseline = "เหมือนเมื่อวาน"
  A2 (16:00)    คู่/คี่ของราคาปิด · baseline = "เหมือนราคา 16:00" (parity_1600)
  C  next_day   ขึ้น = ปิด > ปิดวันก่อน (last_close) · 16:00 ขึ้น = ปิด > ราคา ณ cutoff 15:00 (cutoff_price)
  B  1600 รอบ 1 ผลจริง = close ของแท่ง 16:00 (Yahoo) ตามที่ record_outcomes.py ใช้ -- ต่างจากราคาปิดทางการได้ 1-2 บาท
             (actual_basis = yahoo_bar16 · ราคาปิดทางการเก็บไว้ใน official_close) · รอบ 2 (close) ใช้ราคาปิดทางการ
  B  ราคา      ไม่มี correct · วัดด้วย error เป็นบาท เทียบ Naive (ทาย return = 0) เหมือน record_outcomes.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "task_b" / "raw_data"
DEFAULT_OUT = ROOT / "backfill" / "daily_comparison_20260928_20261022.csv"

# รอบ 2 ของ task_b เริ่มรันสด 2026-10-19 · ช่วง 09-28 → 10-15 ใช้ผล backfill ที่ task_b/tools/backfill_close_r2.py
# ทำไว้ (โมเดล/config ชุดเดียวกับรอบ 2 ที่รันสด · config เลือกด้วยข้อมูล <= 2026-09-25) -- schema ต่างจาก log จึงแปลงตอนอ่าน
R2_EARLY_BACKFILL = "task_b/results/phase1d_r2/backfill_same_day_close.csv"

# (series, kind, live log, key ที่ใช้เช็คซ้ำ, ไฟล์ backfill)
SERIES = [
    ("task_a_next_day", "parity", "task_a/results/prediction_log.csv", None,
     "backfill/results/task_a_next_day.csv"),
    ("task_a2_1600", "parity", "task_a2/results/prediction_log.csv", None,
     "backfill/results/task_a2_t1600.csv"),
    ("task_b_next_day", "price", "task_b/results/prediction_log.csv", "next_day",
     "backfill/results/task_b_next_day.csv"),
    ("task_b_1600", "price", "task_b/results/prediction_log.csv", "same_day_1600",
     "backfill/results/task_b_1600_t1600.csv"),
    ("task_b_close", "price", "task_b/results/prediction_log.csv", "same_day_close",
     ["backfill/results/task_b_close_t1600.csv", R2_EARLY_BACKFILL]),
    ("task_c_next_day", "updown", "task_c/results/prediction_log.csv", None,
     "backfill/results/task_c_next_day.csv"),
    ("task_c_1600", "updown", "task_c/results/prediction_log_1600.csv", None,
     "backfill/results/task_c_1600_t1600.csv"),
]
KEY = ["series", "ticker", "target_date", "model"]
PARITY_TH = {0.0: "คู่", 1.0: "คี่"}


def official_closes(ticker):
    df = pd.read_csv(RAW_DATA / f"{ticker.replace('.BK', '')}_10Y_Cleaned.csv", thousands=",")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    return df.set_index("Date")["Price"].astype(float).sort_index()


def bar16_closes():
    """{(ticker, date): close ของแท่ง 16:00} -- snapshot 1h ของ backfill แล้วทับด้วย outcomes.csv ของ log จริง
    (สองแหล่งตรงกันทุกวันที่ซ้อนกัน · 10-14 และ 10-15 มีแค่ใน outcomes.csv)"""
    out = {}
    for path in sorted((ROOT / "backfill" / "data" / "task_b_1h_snapshots").glob("*.csv")):
        d = pd.read_csv(path)
        ts = pd.to_datetime(d["Datetime"], utc=True).dt.tz_convert("Asia/Bangkok")
        tk = path.name.split("_1h_")[0].replace("_BK", ".BK")
        for day, c in zip(ts[ts.dt.hour == 16].dt.strftime("%Y-%m-%d"), d.loc[ts.dt.hour == 16, "Close"]):
            out[(tk, day)] = float(c)
    live = pd.read_csv(ROOT / "task_b" / "results" / "outcomes.csv")
    live = live[live["prediction_type"] == "same_day_1600"]
    for t, day, c in zip(live["ticker"], live["target_date"], live["actual_close"]):
        out[(t, day)] = float(c)
    return out


def parity_floor(price):
    """คู่/คี่ของส่วนจำนวนเต็มบาท (0 = คู่) -- เหมือน rounding.parity_label"""
    return float((int(np.rint(price * 100)) // 100) % 2)


def parity_round(price):
    """กฎปัดเดิม (เศษ > .50 ปัดขึ้น) -- เหมือน rounding.to_int_baht แล้ว % 2"""
    baht, rem = divmod(int(np.rint(price * 100)), 100)
    return float((baht + (1 if rem > 50 else 0)) % 2)


def load_rows(start, end):
    parts = []
    for series, kind, live_path, ptype, bf_path in SERIES:
        bf_paths = bf_path if isinstance(bf_path, list) else [bf_path]
        for source, path in [("live", live_path)] + [("backfill", q) for q in bf_paths]:
            df = pd.read_csv(ROOT / path)
            if path == R2_EARLY_BACKFILL:
                df = df.rename(columns={"close_bar15": "prev_close"})
                df["data_cutoff"] = df["feature_window_end"]
                df["backfill_reason"] = "pre_r2_launch"
            if ptype and "prediction_type" in df:
                df = df[df["prediction_type"] == ptype]
            df = df[~df["is_dry_run"].astype(str).str.lower().isin(["true", "1"])]
            df = df[(df["target_date"] >= start) & (df["target_date"] <= end)].copy()
            df["series"], df["kind"], df["source"] = series, kind, source
            if source == "live":
                df["backfill_reason"] = ""
            parts.append(df)
    allrows = pd.concat(parts, ignore_index=True)
    # log จริงมาก่อน: ตัดแถว backfill ที่ key ซ้ำกับ log จริง
    allrows["_pri"] = (allrows["source"] != "live").astype(int)
    allrows = allrows.sort_values("_pri", kind="stable").drop_duplicates(KEY, keep="first")
    return allrows.drop(columns="_pri")


def score(rows, closes, bar16):
    out = []
    for r in rows.itertuples(index=False):
        c = closes[r.ticker]
        t = pd.Timestamp(r.target_date)
        official = c.get(t, np.nan)
        by_bar16 = r.series == "task_b_1600"
        actual = bar16.get((r.ticker, r.target_date), np.nan) if by_bar16 else official
        rec = {
            "target_date": r.target_date, "weekday": t.strftime("%a"), "ticker": r.ticker,
            "series": r.series, "kind": r.kind, "model": r.model,
            "is_baseline": str(r.model).startswith("Baseline"),
            "source": r.source, "backfill_reason": r.backfill_reason,
            "generated_at": r.generated_at, "code_commit": r.code_commit,
            "actual_basis": "yahoo_bar16" if by_bar16 else "official_close",
            "official_close": official, "actual_close": actual, "status": "scored" if pd.notna(actual) else "pending",
        }
        if r.kind == "parity":
            rec.update(_parity(r, c, t, actual))
        elif r.kind == "updown":
            rec.update(_updown(r, c, t, actual))
        else:
            rec.update(_price(r, c, actual))
        out.append(rec)
    return pd.DataFrame(out)


def _prev_close(c, cutoff):
    return c.get(pd.Timestamp(cutoff), np.nan)


def _parity(r, c, t, actual):
    is_1600 = r.series == "task_a2_1600"
    if is_1600:
        ref, ref_par, match = r.price_1600, r.parity_1600, ""    # ตรวจกับราคารายวันไม่ได้ (ราคา 15:00)
    else:
        ref = _prev_close(c, r.data_cutoff)
        ref_par = r.prev_parity
        match = bool(pd.notna(ref) and parity_floor(ref) == ref_par)
    d = {"reference_price": ref, "ref_matches_official": match,
         "predicted_proba": r.predicted_flip_proba, "prediction": r.predicted_label}
    if pd.isna(actual):
        return d
    # กฎเก่า/floor ต้องตรงกันในช่วงนี้ (ราคาปิดเป็นจำนวนเต็มบาททั้งหมด) ไม่งั้นต้องมาดูเอง
    assert parity_round(actual) == parity_floor(actual), f"{r.ticker} {t.date()}: round!=floor"
    act = parity_floor(actual)
    d.update(actual=PARITY_TH[act], correct=int(r.predicted_parity == act),
             baseline_correct=int(ref_par == act))
    return d


def _updown(r, c, t, actual):
    next_day = r.series == "task_c_next_day"
    ref = r.last_close if next_day else r.cutoff_price
    d = {"reference_price": ref,
         "ref_matches_official":
             bool(np.isclose(ref, _prev_close(c, r.data_cutoff))) if next_day else "",
         "predicted_proba": r.predicted_up_proba, "prediction": r.predicted_label}
    if pd.isna(actual):
        return d
    up = float(actual > ref)
    pred = r.predicted_updown if next_day else r.predicted_up_to_close
    d.update(actual="ขึ้น" if up else "ลง/นิ่ง", correct=int(pred == up))
    return d


def _price(r, c, actual):
    prev = float(r.prev_close)
    match = ""
    if r.series == "task_b_next_day":
        match = bool(np.isclose(_prev_close(c, r.data_cutoff), prev))
    d = {"reference_price": prev, "ref_matches_official": match,
         "predicted_close": r.predicted_close, "predicted_return": r.predicted_return,
         "prediction": f"{r.predicted_close:.2f}"}
    if pd.isna(actual):
        return d
    act_ret = actual / prev - 1
    err = r.predicted_return - act_ret
    naive = abs(act_ret)
    beat = "tie" if np.isclose(abs(err), naive, rtol=0, atol=1e-12) else (
        "win" if abs(err) < naive else "lose")
    d.update(actual=f"{actual:.2f}", actual_return=act_ret,
             error_baht=r.predicted_close - actual, abs_error_baht=abs(r.predicted_close - actual),
             naive_abs_error_baht=abs(actual - prev), beat_naive=beat)
    return d


COLUMNS = [
    "target_date", "weekday", "ticker", "series", "kind", "model", "is_baseline",
    "source", "backfill_reason", "status",
    "reference_price", "ref_matches_official",
    "prediction", "predicted_proba", "predicted_close", "predicted_return",
    "actual", "actual_basis", "actual_close", "official_close", "actual_return",
    "correct", "baseline_correct",
    "error_baht", "abs_error_baht", "naive_abs_error_baht", "beat_naive",
    "generated_at", "code_commit",
]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--start", default="2026-09-28")
    p.add_argument("--end", default="2026-10-22")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    rows = load_rows(args.start, args.end)
    closes = {t: official_closes(t) for t in rows["ticker"].unique()}
    res = score(rows, closes, bar16_closes())
    for col in COLUMNS:
        if col not in res:
            res[col] = ""
    res = res[COLUMNS].sort_values(["target_date", "ticker", "series", "model"])
    out = Path(args.out)
    res.to_csv(out, index=False, encoding="utf-8-sig")   # BOM ให้ Excel อ่านภาษาไทยถูก

    print(f"เขียน {out} · {len(res)} แถว · {res['target_date'].nunique()} วัน")
    print(res.groupby(["series", "source"]).size().unstack(fill_value=0).to_string())
    pend = res[res["status"] == "pending"]
    if len(pend):
        print(f"!! ยังไม่มีราคาปิดจริง {len(pend)} แถว: {sorted(pend['target_date'].unique())}")


if __name__ == "__main__":
    main()
