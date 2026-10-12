"""
backfill/validate_vs_live.py — วัดว่าผล backfill ตรงกับ log จริงแค่ไหน ในวันที่มี log จริงอยู่แล้ว
(รัน run_one.py โดยตั้ง BACKFILL_RESULTS_DIR=backfill/validation ให้เขียนแยกจากผลจริงของ backfill)
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from common import ROOT, TASKS

VAL = ROOT / "backfill" / "validation"
VALCOL = {"task_a": "predicted_flip_proba", "task_a2": "predicted_flip_proba", "task_c": "predicted_up_proba",
          "task_c_1600": "predicted_up_proba", "task_b": "predicted_return", "task_b_1600": "predicted_return"}
LABEL = {"task_a": "predicted_label", "task_a2": "predicted_label", "task_c": "predicted_label",
         "task_c_1600": "predicted_label"}

rows = []
for task, col in VALCOL.items():
    p = VAL / f"{task}_{TASKS[task]['mode']}.csv"
    if not p.exists():
        continue
    bf = pd.read_csv(p)
    t = TASKS[task]
    lv = pd.read_csv(ROOT / t["live"])
    lv = lv[~lv["is_dry_run"].astype(str).str.lower().isin(["true", "1"])]
    if t["ptype"]:
        lv = lv[lv["prediction_type"] == t["ptype"]]
    m = bf.merge(lv, on=["ticker", "target_date", "model"], suffixes=("_bf", "_live"))
    if m.empty:
        continue
    d = (m[f"{col}_bf"] - m[f"{col}_live"]).abs()
    for _, r in m.iterrows():
        rows.append({"task": task, "target": r["target_date"], "ticker": r["ticker"], "model": r["model"],
                     "live": r[f"{col}_live"], "backfill": r[f"{col}_bf"],
                     "abs_diff": abs(r[f"{col}_bf"] - r[f"{col}_live"]),
                     "label_same": (r[f"{LABEL[task]}_bf"] == r[f"{LABEL[task]}_live"]) if task in LABEL else None})
out = pd.DataFrame(rows)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
print(out.to_string(index=False))
print()
print(out.groupby("task").agg(n=("abs_diff", "size"), max_abs_diff=("abs_diff", "max"),
                              label_same=("label_same", lambda s: s.dropna().mean() if s.notna().any() else None)))
out.to_csv(VAL / "validation_vs_live.csv", index=False)
