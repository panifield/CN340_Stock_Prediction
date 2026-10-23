"""
settrade_features.py — Task A2: feature จากแท่ง 1 นาทีของ Settrade (ก่อนเวลาทาย 16:00)
==================================================================================

ข้อมูล (สำรวจ 2026-10-24): data/settrade_log/{SYMBOL}_1m_{YYYY-MM-DD}.csv จาก collect_settrade_daily.py
  คอลัมน์ time, open, high, low, close, volume, dt -- แท่ง OHLCV 1 นาทีเท่านั้น
  **ไม่มี bid/ask และไม่มีรายการซื้อขายทีละ tick** · แท่งสุดท้าย 16:35 = ATC (ราคาปิดทางการ)
  Settrade เก็บแท่ง 1 นาทีย้อนหลังได้ราว 1 สัปดาห์ วันที่ไม่ได้เก็บไว้จึงดึงคืนไม่ได้

feature (ใช้เฉพาะแท่งที่ dt < 16:00 = เวลาทาย ไม่ว่า dt จะเป็นเวลาเริ่มหรือจบแท่งก็ไม่เห็นข้อมูลหลัง 16:00):
  last_tick_dir        ทิศของราคาปิดแท่งสุดท้ายเทียบแท่งก่อนหน้า +1 uptick / -1 downtick / 0 เท่าเดิม
                       (ประมาณ "tick สุดท้าย" จากแท่ง 1 นาที เพราะไม่มีข้อมูลทีละรายการ)
  last_change_dir      ทิศของการเปลี่ยนราคาครั้งล่าสุดที่ไม่ใช่ 0 (tick test)
  dticks_5m/15m/30m    ขยับกี่ tick ใน 5/15/30 นาทีสุดท้าย (นับบน tick grid ของ SET)
  close_at_bar_low/high ราคาปิดแท่งสุดท้าย = low/high ของแท่ง -> **ตัวแทน** bid/ask เท่านั้น (ราคาเด้งระหว่าง
                       2 ระดับทุกนาที = bid-ask bounce) ไม่ใช่ bid/ask จริง
  price_1559           ราคาปิดแท่งสุดท้ายก่อน 16:00

ยังไม่สรุปผล: power analysis เดิมต้องใช้ ~194 วันถึงจะยืนยันความต่าง 10 จุดได้ (MIN_DAYS_FOR_EVAL)
รัน:  python settrade_features.py      -> พิมพ์สถานะข้อมูล + เขียน results_floor/settrade/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task_a"))
sys.path.insert(0, str(HERE))

from rounding import parity_label  # noqa: E402
from dtick_model import tick_index  # noqa: E402

LOG_DIR = HERE / "data" / "settrade_log"
OUT = Path(os.environ.get("SETTRADE_FEAT_OUT") or HERE / "results_floor" / "settrade")
CUTOFF = "16:00"
WINDOWS = (5, 15, 30)
MIN_DAYS_FOR_EVAL = 200


def load_1m(symbol, log_dir=LOG_DIR):
    """แท่ง 1 นาทีทุกวันที่เก็บไว้ -> DataFrame index = dt (Asia/Bangkok)"""
    files = sorted(Path(log_dir).glob(f"{symbol}_1m_*.csv"))
    if not files:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["dt"] = pd.to_datetime(df["dt"])
    return df.drop_duplicates("dt").set_index("dt").sort_index()[["open", "high", "low", "close", "volume"]]


def day_features(bars, cutoff=CUTOFF):
    """feature ของวันเดียวจากแท่งที่ dt < cutoff เท่านั้น (คืน None ถ้ามีแท่งไม่พอ)"""
    day = bars.index[0].normalize()
    cut = day + pd.Timedelta(cutoff + ":00")
    b = bars[bars.index < cut]
    if len(b) < 31:
        return None
    close = b["close"].astype(float)
    idx = tick_index(close.values)
    diffs = np.diff(idx)
    nz = diffs[diffs != 0]
    last = b.iloc[-1]
    t_last = b.index[-1]

    def dticks(minutes):
        prior = b[b.index <= t_last - pd.Timedelta(minutes=minutes)]
        return np.nan if prior.empty else int(idx[-1] - tick_index(prior["close"].iloc[-1]))

    out = {
        "last_bar_time": t_last.strftime("%H:%M"),
        "price_1559": float(last["close"]),
        "last_tick_dir": int(np.sign(diffs[-1])) if len(diffs) else 0,
        "last_change_dir": int(np.sign(nz[-1])) if len(nz) else 0,
        "close_at_bar_low": int(last["close"] == last["low"] and last["high"] > last["low"]),
        "close_at_bar_high": int(last["close"] == last["high"] and last["high"] > last["low"]),
    }
    for m in WINDOWS:
        out[f"dticks_{m}m"] = dticks(m)
    return out


def build_features(symbol, log_dir=LOG_DIR, cutoff=CUTOFF):
    bars = load_1m(symbol, log_dir)
    rows = {}
    for day, g in bars.groupby(bars.index.date):
        f = day_features(g, cutoff)
        if f is not None:
            rows[pd.Timestamp(day)] = f
    return pd.DataFrame(rows).T.sort_index()


def main():
    import verified_daily
    OUT.mkdir(parents=True, exist_ok=True)
    status = []
    for symbol, yf_t in (("KBANK", "KBANK.BK"), ("ADVANC", "ADVANC.BK")):
        f = build_features(symbol)
        close = verified_daily.load_verified_history(yf_t)["Close"]
        f = f.join(close.rename("official_close"), how="left")
        f["parity_close"] = parity_label(f["official_close"])
        f["parity_1559"] = parity_label(f["price_1559"].astype(float))
        f.to_csv(OUT / f"{symbol}_settrade_features.csv", encoding="utf-8-sig")
        n_lab = int(f["official_close"].notna().sum())
        status.append({"symbol": symbol, "days_1m": len(f), "days_with_official_close": n_lab,
                       "first": f.index.min().date() if len(f) else None,
                       "last": f.index.max().date() if len(f) else None,
                       "min_days_for_eval": MIN_DAYS_FOR_EVAL,
                       "ready_to_evaluate": n_lab >= MIN_DAYS_FOR_EVAL})
    st = pd.DataFrame(status)
    st.to_csv(OUT / "settrade_data_status.csv", index=False, encoding="utf-8-sig")
    print(st.to_string(index=False))
    if not st["ready_to_evaluate"].all():
        print(f"\nข้อมูลยังไม่ถึง {MIN_DAYS_FOR_EVAL} วัน -- สร้าง feature ไว้แต่ยังไม่เทรน/ไม่สรุปผล")


if __name__ == "__main__":
    main()
