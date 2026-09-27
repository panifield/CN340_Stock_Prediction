"""
build_price_at_1600.py — Task A2 ขั้นเตรียมข้อมูล (ยังไม่เทรนโมเดล)
=====================================================================
target  = parity ของราคาปิดรายวัน (yfinance daily, ยืนยันแล้วว่าตรง ATC
          Settrade 100% ใน 719 วันที่ทับกัน)
feature = ราคา ณ 16:00 จาก yfinance intraday (1h, 730 วัน)

*** ประเด็นสำคัญเรื่อง bar convention ***
yfinance intraday เป็น "bar-start" (ยืนยันจากข้อมูลจริง: แท่งแรกของวัน
คือ 10:00 ตรงกับเวลาเปิดตลาดเป๊ะ) ดังนั้นแท่งที่ label "16:00" จริงๆ
ครอบคลุมช่วง [16:00, ~16:30) ซึ่งเป็นข้อมูล "หลัง 16:00" (ปนใกล้ราคาปิด
มาก ไม่ควรใช้เป็น feature ณ 16:00) แท่งที่ควรใช้แทนคือ label "15:00"
ซึ่งครอบคลุม [15:00, 16:00) เต็มไม่ชนพักเที่ยง (12:30-14:30) พอดี ->
close ของแท่ง 15:00 คือ "ราคา ณ 16:00 เป๊ะ" ที่ถูกต้องเชิง causal

รัน:
    python build_price_at_1600.py
"""

import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join("..", "task_a"))
from rounding import to_int_baht, parity, tick_size  # noqa: E402

TICKERS = {"KBANK.BK": "KBANK_BK", "ADVANC.BK": "ADVANC_BK"}
DAILY_CACHE = os.path.join("..", "task_a", "data_cache")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def load_1h(ticker_safe):
    path = os.path.join(DATA_DIR, f"{ticker_safe}_1h_730d.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_daily_close(ticker_safe):
    path = os.path.join(DAILY_CACHE, f"{ticker_safe}_2016-08-26_2026-08-28.csv")
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    s = d["Close"].copy()
    s.index = pd.to_datetime(s.index).date
    return s


def price_at_1600_from_1h(df_1h):
    """เอาแท่ง label 15:00 (ครอบคลุม [15:00,16:00)) = ราคา ณ 16:00 เป๊ะ"""
    mask = df_1h.index.time == pd.Timestamp("15:00").time()
    sub = df_1h[mask]
    s = pd.Series(sub["Close"].values, index=sub.index.date)
    return s


print("=" * 78)
print("  [3] Task A2 — เตรียม feature ราคา ณ 16:00 จาก yfinance 1h")
print("=" * 78)

results = {}
for ticker, safe in TICKERS.items():
    print(f"\n--- {ticker} ---")
    df_1h = load_1h(safe)
    n_days_1h = len(pd.unique(df_1h.index.date))
    print(f"  ข้อมูล 1h ทั้งหมด: {len(df_1h)} แท่ง, {n_days_1h} วัน "
          f"({df_1h.index[0].date()} -> {df_1h.index[-1].date()})")

    # เช็คว่ามีแท่ง 15:00 ครบทุกวันไหม
    times = sorted(set(df_1h.index.time))
    has_1500 = pd.Timestamp("15:00").time() in times
    print(f"  มีแท่ง label 15:00 (=ราคา ณ 16:00): {has_1500}")

    price_1600 = price_at_1600_from_1h(df_1h)
    print(f"  ได้ price@16:00 จำนวน {len(price_1600)} วัน")

    daily_close = load_daily_close(safe)
    common_all = sorted(set(price_1600.index) & set(daily_close.index))
    print(f"  ทับกับราคาปิดรายวัน (สำหรับสร้าง target): {len(common_all)} วัน")

    results[ticker] = {
        "price_1600": price_1600, "daily_close": daily_close,
        "common": common_all, "safe": safe,
    }

# =============================================================================
print("\n" + "=" * 78)
print("  เทียบราคา ณ 16:00 (yfinance) กับ Settrade ในช่วงที่มีข้อมูลทั้งคู่")
print("=" * 78)

for ticker in TICKERS:
    safe = TICKERS[ticker]
    settrade_path = os.path.join(DATA_DIR, f"{ticker.split('.')[0]}_1m_atc_check.csv")
    if not os.path.exists(settrade_path):
        print(f"  [{ticker}] ไม่มีไฟล์ Settrade 1m ({settrade_path}) ข้าม")
        continue
    settrade = pd.read_csv(settrade_path, parse_dates=["dt"])
    settrade["date"] = settrade["dt"].dt.date
    # เอาแท่งสุดท้ายที่ "ก่อนหรือเท่ากับ 16:00" ของแต่ละวัน (bar-start ของ
    # Settrade เอง — เท่าที่เห็น 1m ของ Settrade แท่งเวลา HH:MM คือราคา ณ
    # เวลานั้นๆ ตรงๆ ไม่ต้องเลื่อน)
    before_1600 = settrade[settrade["dt"].dt.time <= pd.Timestamp("16:00").time()]
    settrade_1600 = before_1600.groupby("date")["close"].last()

    price_1600 = results[ticker]["price_1600"]
    common = sorted(set(settrade_1600.index) & set(
        pd.to_datetime(list(price_1600.index)).date))
    print(f"\n  [{ticker}] วันที่เทียบได้: {len(common)}")
    if len(common) == 0:
        continue
    a = price_1600.reindex(common)
    b = settrade_1600.reindex(common)
    tick = tick_size(pd.Series(b.values))
    diff_ticks = (a.values - b.values) / tick.values
    for d in common:
        print(f"    {d}: yfinance@16:00={price_1600.loc[d]:.2f}  "
              f"settrade@16:00={settrade_1600.loc[d]:.2f}  "
              f"diff_ticks={(price_1600.loc[d]-settrade_1600.loc[d])/tick_size(pd.Series([settrade_1600.loc[d]])).iloc[0]:+.2f}")
    print(f"    เฉลี่ย diff = {diff_ticks.mean():+.3f} tick, "
          f"std = {diff_ticks.std():.3f} tick")

# =============================================================================
print("\n" + "=" * 78)
print("  Baseline 'Persistence-16:00' + histogram tick-move (16:00 -> ปิด)")
print("=" * 78)

for ticker in TICKERS:
    r = results[ticker]
    price_1600, daily_close, common = r["price_1600"], r["daily_close"], r["common"]

    p1600 = pd.Series(price_1600.values,
                      index=pd.to_datetime(list(price_1600.index)))
    dclose = pd.Series(daily_close.values,
                       index=pd.to_datetime(list(daily_close.index)))
    idx = pd.to_datetime(common)
    p1600_c = p1600.loc[idx]
    dclose_c = dclose.loc[idx]

    parity_1600 = parity(to_int_baht(p1600_c)).values
    parity_close = parity(to_int_baht(dclose_c)).values
    persist_acc = float(np.mean(parity_1600 == parity_close))

    tick = tick_size(dclose_c).values
    n_1600 = np.round(p1600_c.values / tick)
    n_close = np.round(dclose_c.values / tick)
    tick_move = n_close - n_1600

    print(f"\n  [{ticker}] n={len(idx)} วัน")
    print(f"    Persistence-16:00 accuracy (parity ปิด == parity ณ 16:00): "
          f"{persist_acc:.4f}")
    print(f"    ระยะที่ราคาขยับจาก 16:00 ถึงปิด (หน่วย tick):")
    vals, counts = np.unique(tick_move, return_counts=True)
    for v, c in zip(vals, counts):
        if abs(v) <= 5 or c / len(tick_move) > 0.01:
            print(f"      {int(v):+3d} tick: {c:5d} วัน ({c/len(tick_move)*100:5.2f}%)")
    print(f"    |tick_move|=0 (ไม่ขยับเลย): "
          f"{(tick_move==0).mean()*100:.2f}%")
    print(f"    tick_move mean={tick_move.mean():+.3f}  std={tick_move.std():.3f}")

    out = pd.DataFrame({"date": idx, "price_1600": p1600_c.values,
                        "daily_close": dclose_c.values,
                        "tick_move": tick_move})
    out.to_csv(os.path.join(DATA_DIR, f"{r['safe']}_price1600_vs_close.csv"),
              index=False, encoding="utf-8-sig")

print("\nเสร็จสิ้น [3] — รอดูผลก่อนเริ่มเทรนโมเดล")
