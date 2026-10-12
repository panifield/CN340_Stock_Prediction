"""
backfill/coverage.py — ตารางสรุปว่าแต่ละวัน/แต่ละ task มี log จริง (L) / backfill (B) / ทั้งคู่ (L+B) / ยังขาด (--)
เขียน backfill/COVERAGE.md และ backfill/coverage.csv
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from common import TASKS, ROWS_PER_DATE

days = common.trading_days()
rows = []
for d in days:
    iso = d.date().isoformat()
    r = {"target_date": iso, "weekday": d.strftime("%a")}
    for task in TASKS:
        l = common.live_counts(task).get(iso, 0)
        b = common.backfill_counts(task).get(iso, 0)
        if l >= ROWS_PER_DATE:
            s = "L"
        elif l and b:
            s = "L+B"      # log จริงมีไม่ครบโมเดล + backfill เติมครบ 3 โมเดล
        elif b:
            s = "B"
        elif l:
            s = "L(partial)"
        else:
            s = "--"
        r[task] = s
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(common.BACKFILL_DIR / "coverage.csv", index=False)

cols = list(TASKS)
head = ["วันที่", "วัน"] + [f"{t}<br>({TASKS[t]['mode']})" for t in cols]
md = ["# Coverage ของ log ตั้งแต่ 2026-09-28", "",
      "L = log จริงครบ (6 แถว: 2 หุ้น × 3 โมเดล) · B = มีเฉพาะ backfill · L+B = log จริงมีไม่ครบโมเดล backfill เติมให้ครบ · -- = ยังขาด", "",
      "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
for _, r in df.iterrows():
    md.append("| " + " | ".join([r["target_date"], r["weekday"]] + [r[c] for c in cols]) + " |")
n_missing = int((df[cols] == "--").sum().sum())
md += ["", f"ช่องที่ยังขาด: **{n_missing}** จาก {len(df) * len(cols)} ช่อง (วันทำการ {len(df)} วัน × {len(cols)} ชุด)"]
(common.BACKFILL_DIR / "COVERAGE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(df.to_string(index=False)); print("missing cells:", n_missing)
