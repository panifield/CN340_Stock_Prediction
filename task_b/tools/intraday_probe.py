"""
tools/intraday_probe.py — สำรวจไฟล์รายชั่วโมง (Yahoo) สำหรับโมเดล 16:00 (§6.5)
=================================================================================

    python tools/intraday_probe.py
    python tools/intraday_probe.py --yahoo-daily KBANK_BK_1d.csv ADVANC_BK_1d.csv

สร้างตัวเลขใน PHASE_1C_SPEC §6.1–6.3 ขึ้นมาใหม่จากไฟล์ใน raw_data_intraday/
(ไม่ใช้อินเทอร์เน็ต) แล้วเขียนผลลง results/intraday_probe.csv (full precision)

*** probe + เอกสารเท่านั้น -- ไม่ใช่โมเดล 16:00 (นั่นคือ Phase 1D) ***
- ไม่แตะ pipeline รายวัน · ไม่ต่อข้อมูล Yahoo เข้า raw_data/
- ไฟล์ Yahoo ถูกอ่านด้วย pd.read_csv ตรงนี้เท่านั้น (data_loader ปฏิเสธไฟล์ intraday)
- ราคาปิดของแท่ง 16:00 เรียกว่า close_bar16 -- ยังไม่ใช่ "ราคาปิดทางการ"
  จนกว่าจะยืนยันด้วย --yahoo-daily

สมมติฐาน timestamp: เวลาของแท่ง = เวลา "เริ่ม" ของช่วงชั่วโมง (convention ของ Yahoo)
    แท่ง 15:00 = 15:00 -> 16:00 = รู้แล้ว ณ 16:00
    แท่ง 16:00 = 16:00 -> ปิด   = ยังไม่เกิด ณ 16:00

นิยามที่ใช้ในตาราง §6.3 (ใช้เฉพาะวันที่มีทั้งแท่ง 15:00 และ 16:00 = "วันที่ครบ")
    A  : close_bar16(t) / close_bar15(t) - 1                 naive = ไม่เปลี่ยนจาก bar15
    B2 : close_bar16(t+1) / close_bar15(t) - 1
    อ้างอิง รายวัน ปิด->ปิด : close_bar16(t+1) / close_bar16(t) - 1
    t+1 = วันทำการถัดไปในไฟล์ (ทุกวัน ไม่ใช่ "วันที่ครบถัดไป") แล้วตัดคู่ที่ขาดแท่งทิ้ง
          -> ไม่มีคู่ที่คร่อมวันที่ถูกตัด (horizon เป็น 1 วันจริงทุกคู่ ห้าม ffill)
    naive ทาย return = 0 ทุกวัน -> naive MAE = mean|return|
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BASE_DIR, OUTPUT_DIR, SPLIT_BY_DATE, TICKERS
from data_loader import load_stock
from trading_costs import cost_round_trip

INTRADAY_DIR = BASE_DIR / "raw_data_intraday"
EXPECTED_SHA256 = {
    "KBANK.BK":  "b192ad2c0bc3ae2a75b732752369c56b7dd27d043900eb6e8fb2d5fb08912060",
    "ADVANC.BK": "2097bb53ce1c6ad49023f432128b39363187f4fdaebcb12bbd5ee7e5a295e79c",
}
MISMATCH_WARN_BAHT = 1.0      # ใช้แค่พิมพ์วันที่ต่างมากให้ตรวจด้วยตา


def intraday_path(ticker):
    return INTRADAY_DIR / f"{ticker.replace('.', '_')}_1h_730d.csv"


def read_hourly(path):
    """อ่านไฟล์รายชั่วโมงของ Yahoo -> คอลัมน์ date / hour + OHLCV (เวลาไทย)"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    ts = pd.to_datetime(df["Datetime"])            # มี +07:00 ติดมา -> ใช้เวลาไทยตรง ๆ
    df["date"] = pd.to_datetime(ts.dt.date)
    df["hour"] = ts.dt.hour
    return df


def bars_by_hour(df):
    """ตาราง date x hour ของราคาปิดแต่ละแท่ง (NaN = ไม่มีแท่งนั้น)"""
    return df.pivot_table(index="date", columns="hour", values="Close", aggfunc="last")


def move_stats(ret, cost_rt):
    """naive MAE / std / % ไม่ขยับ / ÷ต้นทุน / % |ขยับ| เกินต้นทุน / % ขึ้นเกินต้นทุน"""
    ret = np.asarray(ret, dtype=float)
    mae = float(np.mean(np.abs(ret)))
    return {
        "n": len(ret),
        "naive_MAE": mae,
        "std": float(np.std(ret, ddof=1)),
        "pct_no_move": float(np.mean(ret == 0)),
        "naive_MAE_over_cost": mae / cost_rt,
        "pct_abs_move_above_cost": float(np.mean(np.abs(ret) > cost_rt)),   # magnitude
        "pct_up_above_cost": float(np.mean(ret > cost_rt)),                 # long-only
    }


def probe_ticker(ticker, cost_rt, yahoo_daily=None):
    rows = []

    def add(section, metric, value):
        rows.append({"ticker": ticker, "section": section,
                     "metric": metric, "value": value})

    path = intraday_path(ticker)
    print("\n" + "#" * 78)
    print(f"#  {ticker}  ({path.name})")
    print("#" * 78)

    # --- SHA-256 ---------------------------------------------------------
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    ok = sha == EXPECTED_SHA256[ticker]
    print(f"[sha256] {sha}  {'ตรง' if ok else '*** ไม่ตรงค่าที่คาดไว้ ***'}")
    add("file", "sha256_match", ok)
    if not ok:
        print("   ตัวเลขทั้งหมดของสเปค §6 วัดจากไฟล์ชุดที่คาดไว้ -- หยุด")
        return rows, False

    df = read_hourly(path)
    bars = bars_by_hour(df)
    days = bars.index
    print(f"[range] {df['Datetime'].iloc[0]} -> {df['Datetime'].iloc[-1]}")
    print(f"        แถว (แท่ง) = {len(df)} · วันทำการ = {len(days)}")
    add("6.1", "n_rows", len(df))
    add("6.1", "n_days", len(days))
    add("6.1", "first_date", str(days[0].date()))
    add("6.1", "last_date", str(days[-1].date()))

    # --- แท่งต่อวันแยกตามปี --------------------------------------------------
    has = bars.notna()
    by_year = has.groupby(days.year).sum()
    by_year.insert(0, "days", has.groupby(days.year).size())
    print("\n[bars] จำนวนวันที่มีแท่งแต่ละชั่วโมง แยกตามปี")
    print(by_year.to_string())
    for year, r in by_year.iterrows():
        for col, v in r.items():
            add("6.1_bars_by_year", f"{year}_{col}", int(v))

    # --- วันที่ไม่ครบ -----------------------------------------------------------
    complete = has.get(15, False) & has.get(16, False)
    n_complete = int(complete.sum())
    incomplete = days[~complete]
    print(f"\n[complete] วันที่มีทั้งแท่ง 15:00 และ 16:00 = {n_complete}")
    print(f"[incomplete] {len(incomplete)} วัน (ตัดทิ้งทั้งวัน ห้าม ffill):")
    for d in incomplete:
        hrs = [h for h in bars.columns if has.loc[d, h]]
        print(f"   {d.date()}  มีแท่ง {hrs}")
        add("6.1_incomplete", str(d.date()), str(hrs))
    add("6.1", "n_complete_days", n_complete)

    # --- Volume = 0 แยกตามชั่วโมง ------------------------------------------
    vol0 = df[df["Volume"] == 0].groupby("hour").size()
    vol0 = vol0.reindex(sorted(df["hour"].unique()), fill_value=0)
    print("\n[volume=0] จำนวนแท่งที่ Volume = 0 แยกตามชั่วโมง")
    print("   " + " · ".join(f"{h}:00={n}" for h, n in vol0.items()))
    for h, n in vol0.items():
        add("6.2_volume0_by_hour", f"{h}:00", int(n))

    # --- ราคาปิดรายวันจาก raw_data/ (investing.com) ----------------------------
    # ใช้เพื่อ (1) หาขอบของ daily data (2) วัดความต่างของสองแหล่งเท่านั้น
    # ไม่ใช่การประเมินโมเดล -- ไม่มีโมเดลใดถูกเทรนหรือทำนายในไฟล์นี้
    daily = load_stock(ticker, verbose=False)
    daily_end = daily.index[-1]

    # --- partition ตาม SPLIT_BY_DATE ของโมเดลรายวัน ---------------------------
    train_end = pd.Timestamp(SPLIT_BY_DATE["train_end"])
    val_end = pd.Timestamp(SPLIT_BY_DATE["val_end"])
    part = {
        "TRAIN": int((days <= train_end).sum()),
        "VAL": int(((days > train_end) & (days <= val_end)).sum()),
        "TEST": int(((days > val_end) & (days <= daily_end)).sum()),
        f"AFTER_DAILY(>{daily_end.date()})": int((days > daily_end).sum()),
    }
    print(f"\n[split] จำนวนวันใน partition ของ SPLIT_BY_DATE: {part}  "
          f"(รวม {sum(part.values())})")
    for k, v in part.items():
        add("6.2_partition", k, v)

    # --- close_bar16 (Yahoo) vs ราคาปิดใน raw_data/ (investing) ---------------
    c16 = bars[16].dropna()
    both = pd.concat([c16.rename("close_bar16"), daily["Close"].rename("investing")],
                     axis=1, join="inner")
    diff = (both["close_bar16"] - both["investing"]).abs()
    src = {"n_overlap": len(both), "pct_exact": float((diff == 0).mean()),
           "mean_abs_diff_baht": float(diff.mean()), "max_abs_diff_baht": float(diff.max())}
    print(f"\n[source] close_bar16 (Yahoo) vs raw_data/ (investing.com) "
          f"บน {src['n_overlap']} วันที่ทับกัน")
    print(f"   ตรงเป๊ะ {src['pct_exact']:.1%} · mean|ต่าง| {src['mean_abs_diff_baht']:.4f} บาท "
          f"· max|ต่าง| {src['max_abs_diff_baht']:.4f} บาท")
    for k, v in src.items():
        add("6.2_source_mismatch", k, v)

    only_daily = daily.index[(daily.index >= days[0]) & (daily.index <= days[-1])]
    only_daily = only_daily.difference(days)
    print(f"   วันที่ daily มีแต่ hourly ไม่มี: {[str(d.date()) for d in only_daily]}")
    for d in only_daily:
        add("6.1_daily_not_hourly", str(d.date()), 1)

    # --- ตาราง §6.3 -----------------------------------------------------------
    # shift บนทุกวันในไฟล์ แล้วค่อย dropna -- คู่ที่คร่อมวันไม่ครบจะหายไปเอง
    ret_a = (bars[16] / bars[15] - 1).dropna()           # = วันที่ครบ 720 วัน
    ret_b2 = (bars[16].shift(-1) / bars[15] - 1).dropna()
    ret_cc = (bars[16].shift(-1) / bars[16] - 1).dropna()
    stats = {
        "A (same-day close_bar16 vs bar15)": move_stats(ret_a, cost_rt),
        "B2 (next close_bar16 vs bar15)":    move_stats(ret_b2, cost_rt),
        "ref daily close->close (Yahoo)":    move_stats(ret_cc, cost_rt),
    }
    tbl = pd.DataFrame(stats).T
    print(f"\n[6.3] ความยากของโจทย์ (ต้นทุนไป-กลับ = {cost_rt:.5%})")
    fmt = tbl.copy()
    for c in fmt.columns:
        if c == "n":
            fmt[c] = fmt[c].astype(int)
        elif c == "naive_MAE_over_cost":
            fmt[c] = fmt[c].map(lambda v: f"{v:.2f}x")
        else:
            fmt[c] = fmt[c].map(lambda v: f"{v:.3%}")
    print(fmt.to_string())
    for name, s in stats.items():
        for k, v in s.items():
            add("6.3_" + name.split(" ")[0], k, v)

    # --- --yahoo-daily: ยืนยันความหมายของ close_bar16 ------------------------
    if yahoo_daily is None:
        print("\n[confirm] close_bar16 vs Yahoo interval 1d : ยังไม่ได้ยืนยัน "
              "(ไม่มีไฟล์ --yahoo-daily)")
        add("6.2_close_bar16_confirm", "status", "ยังไม่ได้ยืนยัน")
    else:
        yd = pd.read_csv(yahoo_daily, encoding="utf-8-sig")
        date_col = "Date" if "Date" in yd.columns else yd.columns[0]
        yd.index = pd.to_datetime(pd.to_datetime(yd[date_col]).dt.date)
        j = pd.concat([c16.rename("close_bar16"), yd["Close"].rename("yahoo_1d")],
                      axis=1, join="inner")
        d = (j["close_bar16"] - j["yahoo_1d"]).abs()
        print(f"\n[confirm] close_bar16 vs Yahoo 1d ({Path(yahoo_daily).name}) "
              f"บน {len(j)} วัน: ตรงเป๊ะ {(d == 0).mean():.1%} · mean|ต่าง| {d.mean():.4f} บาท")
        add("6.2_close_bar16_confirm", "n_overlap", len(j))
        add("6.2_close_bar16_confirm", "pct_exact", float((d == 0).mean()))
        add("6.2_close_bar16_confirm", "mean_abs_diff_baht", float(d.mean()))

    return rows, True


def main():
    p = argparse.ArgumentParser(description="probe ไฟล์รายชั่วโมง (Yahoo) -- §6.5")
    p.add_argument("--yahoo-daily", nargs="+", type=Path, default=None,
                   help="ไฟล์ราคาปิดรายวันจาก Yahoo (interval 1d) "
                        "จับคู่กับ ticker จากชื่อไฟล์ เช่น KBANK_BK_1d.csv")
    args = p.parse_args()

    cost_rt = cost_round_trip()                   # §5 -- ห้าม hardcode
    all_rows, all_ok = [], True
    for ticker in TICKERS:
        yd = None
        if args.yahoo_daily:
            stem = ticker.replace(".BK", "")
            match = [f for f in args.yahoo_daily if f.name.upper().startswith(stem)]
            yd = match[0] if match else None
        rows, ok = probe_ticker(ticker, cost_rt, yd)
        all_rows += rows
        all_ok &= ok

    out = OUTPUT_DIR / "intraday_probe.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[probe] บันทึก {out.relative_to(BASE_DIR)} ({len(all_rows)} แถว)")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
