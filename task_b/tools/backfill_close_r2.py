"""
tools/backfill_close_r2.py — backfill รอบ 2 (same_day_close) ช่วง 2026-09-28 → 2026-10-15 (PHASE1D_PLAN_R2.md ข้อ 19)
=====================================================================================================================

    python tools/backfill_close_r2.py --run          # ทำนายย้อนหลัง 13 วันทำการ + ผลจริง -> results/phase1d_r2/backfill_same_day_close.csv
    python tools/backfill_close_r2.py --leak-test    # วางยาข้อมูลหลังเวลาตัด -> ผลต้องไม่เปลี่ยน (ต่างได้ไม่เกิน noise ของระบบ) -> backfill_leak_test.csv

**ไม่ใช่ผลทำนายสด** -- สร้างหลังรู้ผลแล้ว (generated_at = เวลารัน) ติด is_backfill = True ห้ามนำไปปนกับ prediction_log.csv
ห้ามนำผลช่วงนี้ย้อนกลับไปเลือก config (config ถูกเลือกด้วยข้อมูล <= 2026-09-25 เท่านั้น)

ใช้ผู้ชนะชุดเดียวกับ live (results/phase1d_r2/phase1d_scores.csv) และ predict_close.predict_ticker ตัวเดิม
แท่ง 1h ที่ใช้ต่อวัน (bar_source):
  snapshot_1602         snapshot จริงที่ดาวน์โหลดในวันนั้นก่อน 16:30 และเริ่มก่อน 16:10 (เหมือนที่ระบบสดเห็น)
  snapshot_before_1630  snapshot ล่าสุดของวันนั้นก่อน 16:30 ที่เริ่มหลัง 16:10
  snapshot_after_close  ไม่มี snapshot ก่อน 16:30 ของวันนั้น -> ใช้ snapshot หลังตลาดปิดของวันเดียวกัน (builder ตัดแท่ง >= 16:00 ทิ้ง)
  yahoo_final           ไม่มี snapshot ของวันนั้นเลย -> ไฟล์ Yahoo 1h ที่แคชไว้ใน backfill/data/task_b_1h_snapshots/
ราคาปิดรายวัน: snapshot Yahoo 1d ที่ดึงหลังเหตุการณ์ (raw_data_daily_live/) ใช้เฉพาะวัน < target (builder ตัด index >= target ทิ้งเสมอ)
ผลจริง: raw_data/ (เหมือน next_day / record_outcomes) -- วันที่ raw_data ยังไม่มีราคาปิดเว้นว่าง
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import daily_close as DC                                                # noqa: E402
import predict_close as PC                                              # noqa: E402
from intraday_1600 import BANGKOK, BASE_DIR, load_ticker_bars           # noqa: E402
from live_1600 import find_snapshots, merge_with_locked, prepare_bars   # noqa: E402
from tune_1600 import TICKERS                                           # noqa: E402

START, END = pd.Timestamp("2026-09-28"), pd.Timestamp("2026-10-15")
OUT_DIR = BASE_DIR / "results" / "phase1d_r2"
OUT = OUT_DIR / "backfill_same_day_close.csv"
LEAK_OUT = OUT_DIR / "backfill_leak_test.csv"
FINAL_DIR = BASE_DIR.parent / "backfill" / "data" / "task_b_1h_snapshots"
EXTRA = ["is_backfill", "bar_source", "bar_snapshot_path", "daily_snapshot_path",
         "daily_close_mismatch_rows", "daily_close_dropped_rows"]
OUTCOME = ["actual_close", "actual_return", "error_return", "abs_error", "naive_abs_error", "beat_naive",
           "c15_final", "prev_close_match"]
EXACT_COLS = ["close_bar15", "train_days", "n_features"]          # ต้องเท่ากันเป๊ะ (ข้อมูลนำเข้า / จำนวน)
FLOAT_COLS = ["predicted_return", "predicted_close"]               # ทศนิยมจากโมเดล: RF/XGB รวมผลแบบขนาน (n_jobs=-1) ปัดเศษต่างได้ ~1e-16
FLOAT_TOL = 1e-9                                                   # ผ่านถ้าต่าง <= ค่านี้ · เทียบ noise ที่วัดได้จากการรันซ้ำ (mode=rerun) และ control (>= หลาย 0.001)
CONTROL_MIN = 1e-6                                                 # control ต้องต่างมากกว่าค่านี้ จึงถือว่าการวางยาไวพอ


# ---------------------------------------------------------------
# ปฏิทิน / เลือกแหล่งข้อมูล
# ---------------------------------------------------------------
def backfill_days(start=START, end=END, holidays=None):
    holidays = DC.read_holidays() if holidays is None else holidays
    return [d for d in pd.date_range(start, end) if d.dayofweek < 5 and d not in holidays]


def pick_bar_source(snapshots, day):
    """
    snapshots = [(bars, meta)] เรียงตามเวลาดาวน์โหลด · คืน ((bars, meta) หรือ None, label)
    ไม่มี snapshot ของวันนั้นเลย -> (None, "yahoo_final")
    """
    day = pd.Timestamp(day).normalize()
    same = [s for s in snapshots if s[1]["downloaded_at"].tz_localize(None).normalize() == day]
    cutoff = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=16, minute=30, tz=BANGKOK)
    early = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=16, minute=10, tz=BANGKOK)
    before = [s for s in same if s[1]["downloaded_at"] < cutoff]
    if before:
        s = before[-1]
        return s, ("snapshot_1602" if s[1]["downloaded_at"] < early else "snapshot_before_1630")
    after = [s for s in same if s[1]["downloaded_at"] >= cutoff]
    if after:
        return after[0], "snapshot_after_close"
    return None, "yahoo_final"


def final_bars_path(ticker):
    files = sorted(FINAL_DIR.glob(f"{ticker.replace('.', '_')}_1h_*.csv"))
    if len(files) != 1:
        raise RuntimeError(f"ต้องมีไฟล์ yahoo_final ของ {ticker} ไฟล์เดียวใน {FINAL_DIR} (พบ {len(files)})")
    return files[0]


def load_inputs(ticker):
    """ข้อมูลที่ใช้ซ้ำทุกวันของหุ้นหนึ่งตัว"""
    locked = load_ticker_bars(ticker)
    snaps = find_snapshots(ticker)
    final_path = final_bars_path(ticker)
    final = prepare_bars(pd.read_csv(final_path, encoding="utf-8-sig"))
    dsnap = DC.find_snapshots(ticker)
    if not dsnap:
        raise RuntimeError(f"ไม่มี snapshot ราคาปิดรายวันของ {ticker} -- รัน tools/fetch_yahoo_daily_close.py ก่อน")
    return {"locked": locked, "snaps": snaps, "final": final, "final_path": final_path, "dsnap": dsnap[-1]}


def bars_for_day(ticker, inp, day):
    """คืน (bars รวมไฟล์ล็อก+snapshot, info, label, path)"""
    picked, label = pick_bar_source(inp["snaps"], day)
    if picked is None:
        snap_bars, path, dl = inp["final"], inp["final_path"], ""
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    else:
        snap_bars, meta = picked
        path, sha, dl = meta["path"], meta["sha256"], meta["downloaded_at"].isoformat()
    bars, mism = merge_with_locked(inp["locked"], snap_bars)
    info = {"sha256": sha, "downloaded_at": dl, "source": "yahoo", "path": str(path), "mismatches": len(mism)}
    return bars, info, label, str(path)


def daily_for_day(ticker, inp, day, holidays):
    snap_df, meta = inp["dsnap"]
    locked = DC.load_locked(ticker)
    close, mism, dropped = DC.merge_daily(locked, snap_df)
    DC.require_previous_close(close, day, holidays)              # ไม่มีราคาปิดวันทำการก่อนหน้า = หยุด
    prior = close[close.index < pd.Timestamp(day).normalize()]
    info = {"sha256": meta["sha256"], "source": meta["source"], "path": str(meta["path"]),
            "mismatches": len(mism), "dropped_rows": dropped, "last_date": str(prior.index.max().date())}
    return close, info


# ---------------------------------------------------------------
# ผลจริง
# ---------------------------------------------------------------
def outcome_fields(row, actual_closes, c15_final):
    t = pd.Timestamp(row["target_date"])
    ac = actual_closes.get(t)
    out = {k: np.nan for k in OUTCOME}
    out["beat_naive"] = ""
    c15 = c15_final.get(t)
    if c15 is not None and not np.isnan(c15):
        out["c15_final"] = float(c15)
        out["prev_close_match"] = bool(np.isclose(c15, row["close_bar15"], rtol=0, atol=1e-9))
    if ac is None or np.isnan(ac):
        return out                                                # raw_data ยังไม่มีราคาปิดวันนั้น
    prev = float(row["close_bar15"])
    ar = float(ac) / prev - 1
    err = float(row["predicted_return"]) - ar
    naive = abs(ar)
    out.update(actual_close=float(ac), actual_return=ar, error_return=err, abs_error=abs(err),
               naive_abs_error=naive,
               beat_naive=("tie" if np.isclose(abs(err), naive, rtol=0, atol=1e-12) else
                           ("win" if abs(err) < naive else "lose")))
    return out


def c15_final_series(final_bars):
    d = final_bars[final_bars["hour"] == 15].set_index("date")["Close"]
    d.index = pd.DatetimeIndex(d.index).normalize()
    return d


# ---------------------------------------------------------------
# run
# ---------------------------------------------------------------
def run(days=None):
    from data_loader import load_stock          # ใช้เฉพาะดึงผลจริง (เครื่องมือ ไม่ใช่โมดูล Phase 1D)
    if OUT.exists():
        raise SystemExit(f"{OUT} มีอยู่แล้ว -- ไม่ทับ (ลบเองถ้าต้องการสร้างใหม่)")
    holidays = DC.read_holidays()
    days = days or backfill_days(holidays=holidays)
    winners = PC.load_winners(PC.SCORES_PATH)
    inputs = {t: load_inputs(t) for t in TICKERS}
    actual = {t: load_stock(t, verbose=False)["Close"] for t in TICKERS}
    commit = PC._git("rev-parse", "HEAD")
    dirty = bool(PC._git("status", "--porcelain", "--", ".", ":!results"))
    generated_at = pd.Timestamp.now(tz=BANGKOK).isoformat(timespec="seconds")
    rows = []
    for day in days:
        print(f"\n=== backfill {day.date()}", flush=True)
        for t in TICKERS:
            inp = inputs[t]
            bars, info, label, bpath = bars_for_day(t, inp, day)
            daily, dinfo = daily_for_day(t, inp, day, holidays)
            part = PC.predict_ticker(t, day, winners, bars, info, daily, dinfo)
            c15f = c15_final_series(inp["final"])
            for r in part:
                r.update(generated_at=generated_at, prediction_type=PC.PREDICTION_TYPE,
                         target_date=str(day.date()), code_commit=commit, worktree_dirty=dirty,
                         is_dry_run=False, is_backfill=True, bar_source=label, bar_snapshot_path=bpath,
                         daily_snapshot_path=dinfo["path"])
                r.update(outcome_fields(r, actual[t], c15f))
                rows.append(r)
    cols = PC.COLUMNS + EXTRA + OUTCOME
    df = pd.DataFrame(rows)[cols]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n[backfill] เขียน {len(df)} แถว -> {OUT}")
    return df


# ---------------------------------------------------------------
# leak test
# ---------------------------------------------------------------
def poison_future(bars, daily, day, rng):
    """ข้อมูลที่ไม่ควรเห็น: แท่ง >= 16:00 ของ target + ทุกแท่งหลัง target + ราคาปิดรายวัน >= target (ราคา x0.6..1.5, volume x0.2..6)"""
    b = bars.copy()
    b["Volume"] = b["Volume"].astype(float)                      # กันชนิดข้อมูล int64 ถูกเขียนทับด้วย float
    m = ((b["date"] > day) | ((b["date"] == day) & (b["hour"] >= 16))).to_numpy()
    n = int(m.sum())
    for c in ("Open", "High", "Low", "Close", "Adj Close"):
        b.loc[m, c] = b.loc[m, c].to_numpy() * rng.uniform(0.6, 1.5, n)
    b.loc[m, "Volume"] = b.loc[m, "Volume"].to_numpy() * rng.uniform(0.2, 6.0, n)
    d = daily.copy()
    dm = np.asarray(d.index >= day)
    d[dm] = d[dm].to_numpy() * rng.uniform(0.6, 1.5, int(dm.sum()))
    return b, d, n + int(dm.sum())


def poison_control(bars, daily, day, rng):
    """ข้อมูลที่ควรเห็น: แท่ง 15:00 ของ target และราคาปิดวันทำการก่อนหน้า -> ผลต้องเปลี่ยน"""
    b = bars.copy()
    b["Volume"] = b["Volume"].astype(float)
    m = ((b["date"] == day) & (b["hour"] == 15)).to_numpy()
    for c in ("Open", "High", "Low", "Close", "Adj Close"):
        b.loc[m, c] = b.loc[m, c].to_numpy() * 1.05
    d = daily.copy()
    prev = d.index[d.index < day].max()
    d.loc[prev] = d.loc[prev] * 1.10
    return b, d, int(m.sum()) + 1


def identity(bars, daily, day, rng):
    """ไม่วางยา -- ใช้วัด noise ของระบบเอง (รันซ้ำด้วยข้อมูลเดิม)"""
    return bars.copy(), daily.copy(), 0


def leak_test(days=None):
    if not OUT.exists():
        raise SystemExit("ต้องรัน --run ก่อน (เทียบกับผล backfill ที่เขียนไว้)")
    if LEAK_OUT.exists():
        raise SystemExit(f"{LEAK_OUT} มีอยู่แล้ว -- ไม่ทับ")
    holidays = DC.read_holidays()
    base = pd.read_csv(OUT)
    days = days or backfill_days(holidays=holidays)
    winners = PC.load_winners(PC.SCORES_PATH)
    inputs = {t: load_inputs(t) for t in TICKERS}
    rng = np.random.default_rng(20261018)
    rows = []
    for day in days:
        for mode, poison in (("rerun", identity), ("future", poison_future), ("control", poison_control)):
            for t in TICKERS:
                inp = inputs[t]
                bars, info, _, _ = bars_for_day(t, inp, day)
                daily, dinfo = daily_for_day(t, inp, day, holidays)
                pb, pd_, n = poison(bars, daily, day, rng)
                part = pd.DataFrame(PC.predict_ticker(t, day, winners, pb, info, pd_, dinfo))
                ref = base[(base["target_date"] == str(day.date())) & (base["ticker"] == t)]
                m = part.merge(ref, on=["ticker", "model"], suffixes=("_new", "_ref"))
                assert len(m) == len(part) == len(ref) == 3, (day, t, len(m))
                ex = max(float((m[f"{c}_new"] - m[f"{c}_ref"]).abs().max()) for c in EXACT_COLS)
                fl = max(float((m[f"{c}_new"] - m[f"{c}_ref"]).abs().max()) for c in FLOAT_COLS)
                if mode == "control":
                    res = "OK (ผลเปลี่ยนตามที่คาด)" if max(ex, fl) > CONTROL_MIN else "ใช้ไม่ได้ (ไม่เปลี่ยน)"
                else:
                    res = "PASS (ไม่มี leak)" if (ex == 0 and fl <= FLOAT_TOL) else "FAIL (พบ leak?)"
                rows.append({"target_date": str(day.date()), "ticker": t, "mode": mode, "poisoned_cells": n,
                             "max_diff_exact_cols": ex, "max_diff_float_cols": fl, "result": res})
                print(rows[-1], flush=True)
    rep = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rep.to_csv(LEAK_OUT, index=False)
    print(rep.groupby(["mode", "result"]).size())
    print("noise ของระบบ (rerun) สูงสุด:", rep[rep["mode"] == "rerun"]["max_diff_float_cols"].max(),
          "| future สูงสุด:", rep[rep["mode"] == "future"]["max_diff_float_cols"].max(),
          "| control ต่ำสุด:", rep[rep["mode"] == "control"]["max_diff_float_cols"].min())
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description="backfill รอบ 2 (same_day_close) -- ไม่ใช่ผลทำนายสด")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument("--leak-test", action="store_true")
    ap.add_argument("--only-dates", nargs="*", help="YYYY-MM-DD ... (ทดสอบ)")
    a = ap.parse_args(argv)
    days = [pd.Timestamp(d) for d in a.only_dates] if a.only_dates else None
    if a.run:
        run(days)
    else:
        leak_test(days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
