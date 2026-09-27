"""
check_intraday.py — Task A2 (ขั้นตรวจสอบข้อมูลเท่านั้น)
========================================================
ตรวจว่าดึงข้อมูล intraday ของ KBANK.BK / ADVANC.BK จาก yfinance ได้ไหม
สำหรับใช้ทำ Task A2 (ทำนายคู่/คี่ของราคาปิดวัน t ณ เวลา 16:00 ของวัน t)

*** ขั้นนี้เป็นแค่การตรวจสอบข้อมูล ไม่มีการเทรนโมเดล ไม่แก้ Task A เดิม ***

ดึง 2 ชุด:
  - interval="1h",  period="730d"  (ยาวสุดที่ yfinance ให้สำหรับ 1h)
  - interval="5m",  period="60d"   (ยาวสุดที่ yfinance ให้สำหรับ 5m)

รัน:
    python check_intraday.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

TICKERS = ["KBANK.BK", "ADVANC.BK"]
DATASETS = [
    {"interval": "1h", "period": "730d"},
    {"interval": "5m", "period": "60d"},
]
TZ = "Asia/Bangkok"
SUBPERIOD_START = "2025-09-07"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
DAILY_CACHE_DIR = os.path.join(HERE, "..", "task_a", "data_cache")
os.makedirs(DATA_DIR, exist_ok=True)


# =============================================================================
def fetch(ticker, interval, period):
    """ดึงข้อมูลดิบจาก yfinance คืน None ถ้าดึงไม่ได้/ไม่มีข้อมูล"""
    try:
        df = yf.download(ticker, interval=interval, period=period,
                         auto_adjust=False, progress=False)
    except Exception as e:
        print(f"    !! ดึง {ticker} interval={interval} period={period} "
              f"ไม่ได้: {type(e).__name__}: {e}")
        return None

    if df is None or len(df) == 0:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def to_bangkok(df):
    """แปลง timezone ของ index เป็น Asia/Bangkok"""
    idx = df.index
    if idx.tz is None:
        # yfinance ปกติคืน tz-aware อยู่แล้ว (เวลาตลาดต้นทาง) แต่กันไว้เผื่อ
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(TZ)
    out = df.copy()
    out.index = idx
    return out


def load_daily_close(ticker):
    """โหลดราคาปิดรายวันจากไฟล์ cache เดิมของ task_a เพื่อเทียบ"""
    safe = ticker.replace(".", "_")
    path = os.path.join(DAILY_CACHE_DIR, f"{safe}_2016-08-26_2026-08-28.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    s = d["Close"].copy()
    s.index = pd.to_datetime(s.index).date
    return s


def infer_timestamp_convention(df_bkk, interval):
    """
    เช็คว่า timestamp เป็นเวลาเริ่มแท่งหรือจบแท่ง โดยดูจากเอกสาร yfinance
    (ตามหลักการทั่วไป: yfinance/Yahoo คืน "bar start" — แท่งเวลา 10:00
    หมายถึงข้อมูลของช่วง 10:00-10:59 สำหรับ 1h หรือ 10:00-10:04 สำหรับ 5m)
    แล้วยืนยันซ้ำจากข้อมูลจริง: ถ้าเป็น bar start แท่งแรกของวันควรตรงกับ
    เวลาเปิดตลาด (~10:00) และแท่งสุดท้ายไม่ควรเกินเวลาปิดตลาด
    """
    first_times = df_bkk.groupby(df_bkk.index.date).apply(
        lambda g: g.index.time.min())
    last_times = df_bkk.groupby(df_bkk.index.date).apply(
        lambda g: g.index.time.max())
    common_first = pd.Series(first_times).mode()
    common_last = pd.Series(last_times).mode()
    return {
        "convention_per_yfinance_docs": "bar start (ต้นแท่ง)",
        "most_common_first_bar_time": str(common_first.iloc[0]) if len(common_first) else None,
        "most_common_last_bar_time": str(common_last.iloc[0]) if len(common_last) else None,
    }


def analyze_dataset(ticker, interval, period, daily_close):
    print(f"\n  กำลังดึง {ticker} interval={interval} period={period} ...")
    df = fetch(ticker, interval, period)

    result = {"ticker": ticker, "interval": interval, "period": period}

    if df is None:
        print("    -> ไม่มีข้อมูล / ดึงไม่ได้")
        result.update({
            "n_bars": 0, "n_days": 0, "date_start": None, "date_end": None,
            "ends_at_1600": False, "match_daily_close_pct": None,
            "covers_since_20250907": False,
            "note": "ดึงข้อมูลจาก yfinance ไม่ได้ / ไม่มีข้อมูลเลย",
        })
        return result, None

    df_bkk = to_bangkok(df)
    n_bars = len(df_bkk)
    date_list = df_bkk.index.date
    n_days = len(pd.unique(date_list))
    date_start, date_end = df_bkk.index[0], df_bkk.index[-1]

    times_found = sorted(set(t.strftime("%H:%M") for t in df_bkk.index.time))

    conv = infer_timestamp_convention(df_bkk, interval)

    # --- จำนวนแท่งต่อวัน + วันที่ผิดปกติ ---
    bars_per_day = pd.Series(date_list).value_counts().sort_index()
    typical = int(bars_per_day.mode().iloc[0]) if len(bars_per_day) else 0
    irregular = bars_per_day[bars_per_day != typical]

    # --- มีแท่งจบตรง 16:00 ไหม + ไม่มีแท่งไหนเกิน 16:00 ---
    ends_at_1600 = "16:00" in times_found
    max_time = max(df_bkk.index.time)
    exceeds_1600 = max_time > pd.Timestamp("16:00").time()

    # --- เทียบราคาปิดแท่งสุดท้ายของแต่ละวัน vs ราคาปิดรายวันเดิม ---
    match_pct = None
    n_compared = 0
    if daily_close is not None:
        last_bar_close = df_bkk.groupby(date_list)["Close"].last()
        common_dates = sorted(set(last_bar_close.index) & set(daily_close.index))
        n_compared = len(common_dates)
        if n_compared > 0:
            a = last_bar_close.loc[common_dates].values
            b = daily_close.loc[common_dates].values
            matches = np.isclose(a, b, atol=0.01)
            match_pct = float(matches.mean() * 100)

    # --- ครอบคลุมตั้งแต่ 2025-09-07 ไหม ---
    sub_start = pd.Timestamp(SUBPERIOD_START).date()
    covers_since = date_start.date() <= sub_start <= date_end.date() \
        if n_bars > 0 else False
    # ต้องมีแท่งจริงในช่วงนั้นด้วย ไม่ใช่แค่ date range ครอบ
    has_bars_after_sub = (pd.Series(date_list) >= sub_start).any()

    print(f"    n_bars={n_bars}  n_days={n_days}  "
          f"{date_start.date()} -> {date_end.date()}")
    print(f"    เวลาแท่งที่พบ: {times_found[:5]}...{times_found[-5:]} "
          f"(รวม {len(times_found)} เวลาที่ต่างกัน)")
    print(f"    แท่งแรกของวันปกติ: {conv['most_common_first_bar_time']}, "
          f"แท่งสุดท้ายของวันปกติ: {conv['most_common_last_bar_time']}")
    print(f"    มีแท่งจบตรง 16:00: {ends_at_1600}  "
          f"มีแท่งเกิน 16:00: {exceeds_1600} (เวลาสุดท้ายที่พบ={max_time})")
    print(f"    วันที่จำนวนแท่งผิดปกติ: {len(irregular)}/{len(bars_per_day)} วัน "
          f"(ปกติ={typical} แท่ง/วัน)")
    if match_pct is not None:
        print(f"    ตรงกับราคาปิดรายวันเดิม: {match_pct:.1f}% "
              f"(เทียบได้ {n_compared} วัน)")
    else:
        print("    ไม่มีวันที่ทับซ้อนกับไฟล์ราคาปิดรายวันเดิม เทียบไม่ได้")
    print(f"    ครอบคลุมตั้งแต่ {SUBPERIOD_START}: {has_bars_after_sub}")

    result.update({
        "n_bars": n_bars, "n_days": n_days,
        "date_start": date_start.date(), "date_end": date_end.date(),
        "n_distinct_bar_times": len(times_found),
        "first_bar_time_typical": conv["most_common_first_bar_time"],
        "last_bar_time_typical": conv["most_common_last_bar_time"],
        "ends_at_1600": ends_at_1600,
        "exceeds_1600": exceeds_1600,
        "n_irregular_days": int(len(irregular)),
        "n_total_days": int(len(bars_per_day)),
        "typical_bars_per_day": typical,
        "match_daily_close_pct": round(match_pct, 1) if match_pct is not None else None,
        "n_days_compared": n_compared,
        "covers_since_20250907": bool(has_bars_after_sub),
        "note": "",
    })

    # บันทึกดิบ
    safe = ticker.replace(".", "_")
    out_path = os.path.join(DATA_DIR, f"{safe}_{interval}_{period}.csv")
    df_bkk.to_csv(out_path, encoding="utf-8-sig")
    print(f"    [saved] {out_path}")

    return result, {"times_found": times_found, "irregular_days": irregular}


def main():
    print("=" * 78)
    print("  Task A2 — ตรวจสอบข้อมูล intraday จาก yfinance")
    print("  (ยังไม่มีการเทรนโมเดล / ไม่แก้ Task A เดิม)")
    print("=" * 78)

    rows = []
    details = {}
    for ticker in TICKERS:
        daily_close = load_daily_close(ticker)
        if daily_close is None:
            print(f"\n!! ไม่พบไฟล์ราคาปิดรายวันเดิมของ {ticker} "
                  f"(เทียบราคาปิดไม่ได้)")

        for ds in DATASETS:
            r, d = analyze_dataset(ticker, ds["interval"], ds["period"],
                                   daily_close)
            rows.append(r)
            details[(ticker, ds["interval"])] = d

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("  ตารางสรุป")
    print("=" * 78)
    show_cols = ["ticker", "interval", "n_days", "date_start", "date_end",
                "ends_at_1600", "match_daily_close_pct",
                "covers_since_20250907"]
    print(summary[show_cols].to_string(index=False))

    summary_path = os.path.join(DATA_DIR, "..", "intraday_check_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[csv] {summary_path}")

    print("\n" + "=" * 78)
    print("  ความเห็น")
    print("=" * 78)
    for _, r in summary.iterrows():
        if r["n_bars"] == 0:
            print(f"  [{r['ticker']} / {r['interval']}] ใช้ไม่ได้ — "
                  f"{r['note']}")
            continue
        usable = r["ends_at_1600"] and not r["exceeds_1600"]
        print(f"  [{r['ticker']} / {r['interval']}] "
              f"{'ใช้ตัดที่ 16:00 ได้แม่นยำ' if usable else 'ตัดที่ 16:00 ไม่แม่นยำ/ไม่มีแท่งจบพอดี'}"
              f" — แท่งสุดท้ายทั่วไปของวัน = "
              f"{r['last_bar_time_typical']}, "
              f"ตรงกับราคาปิดรายวัน {r['match_daily_close_pct']}%")

    return summary


if __name__ == "__main__":
    main()
