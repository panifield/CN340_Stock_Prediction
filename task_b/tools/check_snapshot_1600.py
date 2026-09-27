"""
tools/check_snapshot_1600.py — ตรวจ real-time availability ของ live snapshot (Phase 1D)
=====================================================================================

    python tools/check_snapshot_1600.py --date 2026-09-29

สำหรับทุก snapshot ที่ดาวน์โหลดในวันนั้น (ทุกหุ้น) บันทึกว่า:
  - ดาวน์โหลดกี่โมง (ห่างจาก 16:00 เท่าไร)
  - มีแท่งบังคับ 10, 11, 12, 14, 15 ของวันนั้นครบไหม
    (ข้อมูลอย่างเดียวบอกไม่ได้ว่าแท่ง 15:00 จบแล้วจริง -- ต้องเทียบกับ snapshot ที่ดาวน์โหลดหลัง
     ตลาดปิดของวันเดียวกัน ว่าแท่ง 15:00 ค่าเท่ากันไหม: รันสคริปต์นี้ซ้ำหลัง 17:00 แล้วดู
     คอลัมน์ bar15_changed_vs_later)
  - มีแท่ง 16:00 ไหม (ถ้ามีก่อน 16:30 = แท่งยังไม่จบ ห้ามใช้เป็นผลจริง)
  - แท่งที่ทับไฟล์ล็อกค่าไม่ตรงกี่แท่ง
ผลต่อท้าย results/dryrun/availability_1600.csv -- ใช้เป็นหลักฐานก่อนเขียน PHASE1D_LIVE_APPROVAL.md

ไม่ fit โมเดล · ไม่ทำนาย
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intraday_1600 import BASE_DIR, FEATURE_HOURS, load_ticker_bars, time_fields   # noqa: E402
from live_1600 import LIVE_DIR, find_snapshots, merge_with_locked                   # noqa: E402

OUT = BASE_DIR / "results" / "dryrun" / "availability_1600.csv"
TICKERS = ["KBANK.BK", "ADVANC.BK"]


def check(date, root=LIVE_DIR, out=OUT):
    date = pd.Timestamp(date).normalize()
    tf = time_fields(date)
    rows = []
    for t in TICKERS:
        locked = load_ticker_bars(t)
        todays = [s for s in find_snapshots(t, root)
                  if s[1]["downloaded_at"].tz_localize(None).normalize() == date]
        latest = todays[-1][0] if todays else None
        for bars, meta in todays:
            day = bars[bars["date"] == date]
            have = sorted(day["hour"].unique())
            _, mism = merge_with_locked(locked, bars)
            # แท่ง 15:00 ใน snapshot นี้ เทียบกับ snapshot ล่าสุดของวัน (ถ้าเปลี่ยน = ตอนดาวน์โหลดยังไม่จบ)
            b15 = day[day["hour"] == 15][["Open", "High", "Low", "Close", "Volume"]]
            l15 = latest[(latest["date"] == date) & (latest["hour"] == 15)][
                ["Open", "High", "Low", "Close", "Volume"]]
            changed = (None if b15.empty or l15.empty or latest is bars
                       else not b15.reset_index(drop=True).equals(l15.reset_index(drop=True)))
            rows.append({
                "date": str(date.date()), "ticker": t, "snapshot": str(meta["path"]),
                "source": meta["source"], "downloaded_at": meta["downloaded_at"].isoformat(),
                "minutes_after_1600": (meta["downloaded_at"] - tf["feature_window_end"]).total_seconds() / 60,
                "hours_present": " ".join(map(str, have)),
                "required_bars_complete": all(h in have for h in FEATURE_HOURS),
                "has_bar16": 16 in have,
                "downloaded_before_1600": meta["downloaded_at"] < tf["feature_window_end"],
                "locked_mismatch_bars": len(mism),
                "bar15_changed_vs_later": changed,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"ไม่พบ snapshot ที่ดาวน์โหลดวันที่ {date.date()}")
        return df
    print(df.drop(columns=["snapshot"]).to_string(index=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, mode="a", header=not out.exists(), index=False)
    print(f"\n[availability] ต่อท้าย {len(df)} แถว -> {out}")
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description="ตรวจ real-time availability ของ snapshot 16:00")
    p.add_argument("--date", required=True)
    check(p.parse_args(argv).date)


if __name__ == "__main__":
    main()
