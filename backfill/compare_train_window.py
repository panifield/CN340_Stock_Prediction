"""
backfill/compare_train_window.py — เทียบผล backfill ที่เทรนด้วยข้อมูลทั้งหมด vs เทรนเฉพาะข้อมูลตั้งแต่วันหนึ่ง

    python backfill/compare_train_window.py experiments/since_20250908

ใช้เฉพาะวันที่เป็นช่องว่างของ log จริง (38 ชุด) ซึ่งตอนนี้รู้ผลจริงแล้ว จึงวัดความแม่นยำได้ แต่ตัวอย่างน้อยมาก
(~7 วัน × 2 หุ้น ต่อโมเดล) -- ใช้ดูทิศทาง/ขนาดการเปลี่ยนแปลง ไม่ใช่ข้อสรุปทางสถิติ
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import ROOT, TASKS, BACKFILL_DIR  # noqa: E402

sys.path.insert(0, str(ROOT / "task_a"))
from rounding import to_int_baht, parity  # noqa: E402

KEY = ["ticker", "target_date", "model"]
CLS = {  # task -> (pred col, proba col)
    "task_a": ("predicted_parity", "predicted_flip_proba"), "task_a2": ("predicted_parity", "predicted_flip_proba"),
    "task_c": ("predicted_updown", "predicted_up_proba"), "task_c_1600": ("predicted_up_to_close", "predicted_up_proba"),
}


def _load(f):
    d = pd.read_pickle(common.YF_CACHE_DIR / f)
    d.columns = d.columns.get_level_values(0)
    return d


def actuals():
    idx = pd.read_json(common.YF_CACHE_DIR / "index.jsonl", lines=True)
    daily, bar16 = {}, {}
    for tk in ("KBANK", "ADVANC"):
        d = _load(idx[(idx.file.str.startswith(f"{tk}_BK_1d")) & (idx.rows > 2000)].file.iloc[0])
        d.index = pd.to_datetime(d.index).normalize()
        daily[f"{tk}.BK"] = d["Close"].astype(float)
        h = _load(idx[(idx.file.str.startswith(f"{tk}_BK_1h")) & (idx.rows > 1000)].file.iloc[0])
        h.index = (h.index.tz_convert("Asia/Bangkok") if h.index.tz else h.index.tz_localize("UTC").tz_convert("Asia/Bangkok"))
        b = h[h.index.hour == 16]["Close"].astype(float)
        b.index = b.index.normalize().tz_localize(None)
        bar16[f"{tk}.BK"] = b
    return daily, bar16


def main(exp):
    exp_res = BACKFILL_DIR / exp / "results"
    daily, bar16 = actuals()
    out = []
    for task in TASKS:
        name = f"{task}_{TASKS[task]['mode']}.csv"
        full = pd.read_csv(BACKFILL_DIR / "results" / name)
        since = pd.read_csv(exp_res / name)
        m = full.merge(since, on=KEY, suffixes=("_full", "_since"))
        if m.empty:
            continue
        t = pd.to_datetime(m.target_date)
        close = np.array([daily[tk].get(d, np.nan) for tk, d in zip(m.ticker, t)])
        prev = np.array([daily[tk].loc[:d - pd.Timedelta(days=1)].iloc[-1] for tk, d in zip(m.ticker, t)])
        r = {"task": task}
        if task in CLS:
            pc, qc = CLS[task]
            if task in ("task_a", "task_a2"):
                actual = parity(to_int_baht(pd.Series(close))).to_numpy()
            elif task == "task_c":
                actual = (close > prev).astype(float)
            else:
                actual = (close > m["cutoff_price_full"].to_numpy()).astype(float)
            pf, ps = m[f"{pc}_full"].to_numpy(), m[f"{pc}_since"].to_numpy()
            r.update(n=len(m), same_label=f"{np.mean(pf == ps):.0%}",
                     mean_abs_dproba=round(float(np.mean(np.abs(m[f"{qc}_full"] - m[f"{qc}_since"]))), 3),
                     acc_full=f"{np.mean(pf == actual):.1%}", acc_since=f"{np.mean(ps == actual):.1%}",
                     base_rate=f"{max(np.mean(actual), 1 - np.mean(actual)):.1%}")
        else:
            act = close if task == "task_b" else np.array([bar16[tk].get(d, np.nan) for tk, d in zip(m.ticker, t)])
            base = m["prev_close_full"].to_numpy()
            ef = np.abs(m["predicted_close_full"] - act)
            es = np.abs(m["predicted_close_since"] - act)
            r.update(n=len(m), mean_abs_dclose=round(float(np.mean(np.abs(m["predicted_close_full"] - m["predicted_close_since"]))), 3),
                     mae_full=round(float(ef.mean()), 3), mae_since=round(float(es.mean()), 3),
                     mae_naive=round(float(np.mean(np.abs(base - act))), 3))
        out.append(r)
    df = pd.DataFrame(out)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    df.to_csv(BACKFILL_DIR / exp / "comparison.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experiments/since_20250908")
