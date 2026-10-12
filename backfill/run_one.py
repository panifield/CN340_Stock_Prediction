"""
backfill/run_one.py — ทำนายย้อนหลัง 1 task × 1 วัน (หรือเตรียมข้อมูลของ task_b)
=================================================================================

    python backfill/run_one.py --task task_a --target-date 2026-10-07 [--reason missing]
    python backfill/run_one.py --prepare-b-raw --through 2026-10-12

ไม่แก้ไฟล์ใน task_*/ เลย -- ใช้วิธี "ครอบ" ฟังก์ชันดึงข้อมูลของสคริปต์จริงให้ตัดข้อมูลไว้ ณ เวลาตัด
ของวันนั้น แล้วเรียก predict_one_ticker() ตัวเดิมของ task นั้น (โค้ดโมเดลชุดเดียวกับระบบจริง)
ข้าม guard เวลา/ตลาดปิด/clean-tree ซึ่งออกแบบมาสำหรับรันสด และเขียนผลลง backfill/results/ เท่านั้น
ต้องรันด้วย Python ที่ลง requirements ครบ (ใช้ .venv ของโปรเจกต์)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import common  # noqa: E402
from common import ROOT, TASKS, DATA_DIR  # noqa: E402

B_RAW_DIR = DATA_DIR / "task_b_raw_data"


def _finish(task, rows, cols, reason, extra_cols=None):
    out = pd.DataFrame(rows)
    out = out[cols + [c for c in (extra_cols or []) if c not in cols]]
    common.write_rows(task, out, reason)
    show = [c for c in ("ticker", "target_date", "model", "predicted_label", "predicted_return",
                        "predicted_close") if c in out.columns]
    print(out[show].to_string(index=False))


TRAIN_START = os.environ.get("BACKFILL_TRAIN_START")      # ทดลอง: ตัดข้อมูลเทรนให้เริ่มวันนี้ (ปกติไม่ตั้ง = ใช้ทั้งหมด)


def _since(*objs):
    """ตัดแถว index < TRAIN_START ออกจาก pandas object ทุกตัว (ถ้าไม่ตั้งค่า คืนของเดิม)"""
    if not TRAIN_START:
        return objs if len(objs) > 1 else objs[0]
    out = []
    for o in objs:
        idx = pd.DatetimeIndex(o.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        out.append(o[idx >= pd.Timestamp(TRAIN_START)])
    return tuple(out) if len(out) > 1 else out[0]


def _announce_train(label, idx):
    if os.environ.get("BACKFILL_CAPTURE_TRAIN"):
        idx = pd.DatetimeIndex(idx)
        print(f"TRAIN_WINDOW {label} first={idx.min().date()} last={idx.max().date()} n={len(idx)}", flush=True)


def _enter(task):
    d = ROOT / TASKS[task]["dir"]
    os.chdir(d)
    sys.path.insert(0, str(d))
    common.install_yf_cache()
    return d


# ---------------------------------------------------------------
# task_a / task_c (next_day): ตัดข้อมูลรายวันให้เหลือก่อนวัน target
# ---------------------------------------------------------------

def run_next_day_ac(task, target, reason):
    _enter(task)
    import predict_live as pl
    tgt = pd.Timestamp(target)
    orig = pl.fetch_live
    pl.fetch_live = lambda t: orig(t)[lambda d: d.index < tgt]
    orig_fit = pl.fit_all_models

    def fit_and_announce(X_all, *a, **k):
        if TRAIN_START:
            # โค้ดจริงเรียก walk-forward CV (min_train=250) ซึ่งข้อมูลสั้นแบบนี้ไม่พอ -> ข้าม (OOF เป็นแค่ข้อมูลอ้างอิง)
            if os.environ.get("BACKFILL_ORIG_FIT"):
                X_all, *a = _since(X_all, *a)
                return orig_fit(X_all, *a, **k)
            ys = _since(X_all, *[x for x in a if hasattr(x, "index")])
            X_all, ys = ys[0], ys[1:]
            _announce_train(task, X_all.index)
            models = pl.get_classifiers()
            for m in models.values():
                m.fit(X_all, ys[0])
            return models
        _announce_train(task, X_all.index)
        return orig_fit(X_all, *a, **k)
    pl.fit_all_models = fit_and_announce

    rows = []
    for t in pl.TICKERS:
        rows.extend(pl.predict_one_ticker(t, target))
    out = pd.DataFrame(rows)
    assert (pd.to_datetime(out["data_cutoff"]) < tgt).all(), "data_cutoff ต้องก่อน target"
    out["generated_at"] = common.now_iso()
    out["code_commit"] = pl.get_code_commit()
    out["worktree_dirty"] = pl.worktree_dirty()
    out["is_dry_run"] = False
    _finish(task, out.to_dict("records"), pl.LOG_COLUMNS, reason)


# ---------------------------------------------------------------
# task_a2 (t1600)
# ---------------------------------------------------------------

def run_t1600_a2(task, target, reason):
    _enter(task)
    import predict_live as pl
    tgt = pd.Timestamp(target)

    o1h, od = pl.fetch_live_1h, pl.fetch_live_daily

    def trim_1h(tk):
        df = o1h(tk)
        keep = (df.index.date < tgt.date()) | ((df.index.date == tgt.date()) & (df.index.hour <= 15))
        if TRAIN_START:
            keep = keep & (df.index.tz_localize(None) >= pd.Timestamp(TRAIN_START))
        return df[keep]

    def trim_daily(tk):                      # เก็บแถววัน target ไว้ (feature ทุกตัว shift(1) แล้ว ไม่มี same-day leak)
        df = od(tk)
        return df[df.index <= tgt]

    pl.fetch_live_1h, pl.fetch_live_daily = trim_1h, trim_daily

    rows = []
    for t, short in pl.TICKERS.items():
        tb = trim_1h(t)
        tb = tb[tb.index.date == tgt.date()]
        if not (tb.index.hour == 15).any():
            raise RuntimeError(f"{t}: ไม่มีแท่ง 15:00 ของ {target} ใน Yahoo -- ทำ backfill ไม่ได้")
        rows.extend(pl.predict_one_ticker(t, short, target, True))
    out = pd.DataFrame(rows)
    out["generated_at"] = common.now_iso()
    out["code_commit"] = pl.get_code_commit()
    out["worktree_dirty"] = pl.worktree_dirty()
    out["is_dry_run"] = False
    _finish(task, out.to_dict("records"), pl.LOG_COLUMNS, reason)


# ---------------------------------------------------------------
# task_c (t1600)
# ---------------------------------------------------------------

def run_t1600_c(task, target, reason):
    _enter(task)
    import predict_live_1600 as pl
    tgt = pd.Timestamp(target)
    o1h = pl.fetch_live_1h

    def trim_1h(tk):
        df = o1h(tk)
        d = df["Date"].dt.tz_localize(None) if df["Date"].dt.tz is not None else df["Date"]
        return df[(d < tgt) | ((d == tgt) & (df["Hour"] <= 15))]

    def daily_close_nowrite(ticker, start, end, use_cache=True):
        # เหมือน intraday_task2.load_daily_close แต่ (1) ไม่อ่าน/เขียน data_cache/ ของ production
        # (cache อาจมีวันหลัง target -> prev_close ผิด) (2) ตัดให้ไม่เกิน end
        import yfinance as yf
        s0 = pd.Timestamp(start).tz_localize(None) if pd.Timestamp(start).tz else pd.Timestamp(start)
        e0 = pd.Timestamp(end).tz_localize(None) if pd.Timestamp(end).tz else pd.Timestamp(end)
        df = yf.download(ticker, start=s0.strftime("%Y-%m-%d"),
                         end=(e0 + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), interval="1d",
                         auto_adjust=False, progress=False)
        if df is None or len(df) == 0:
            raise RuntimeError(f"ดึงราคาปิดรายวันของ {ticker} ไม่ได้")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"].dropna().copy()
        s.index = pd.to_datetime(s.index).normalize()
        return s[s.index <= e0]

    pl.fetch_live_1h, pl.load_daily_close = trim_1h, daily_close_nowrite
    if TRAIN_START:
        orig_b = pl.build_task2_dataset

        def b_since(*a, **k):
            X, y, z = orig_b(*a, **k)
            return _since(X, y, z)
        pl.build_task2_dataset = b_since

    rows = []
    for t in pl.TICKERS:
        tb = trim_1h(t)
        tb = tb[tb["Date"].dt.tz_localize(None) == tgt]
        if not (tb["Hour"] == 15).any():
            raise RuntimeError(f"{t}: ไม่มีแท่ง 15:00 ของ {target} ใน Yahoo -- ทำ backfill ไม่ได้")
        rows.extend(pl.predict_one_ticker(t, target, True))
    out = pd.DataFrame(rows)
    out["generated_at"] = common.now_iso()
    out["code_commit"] = pl.get_code_commit()
    out["worktree_dirty"] = pl.worktree_dirty()
    out["is_dry_run"] = False
    _finish(task, out.to_dict("records"), pl.LOG_COLUMNS, reason)


# ---------------------------------------------------------------
# task_b next_day: ใช้ --as-of ที่มีอยู่แล้วของ predict_live.py (ปกติจำกัดแค่ dry-run) กับ raw_data
# สำเนาใน backfill/data/ ที่ต่อข้อมูล Yahoo ให้ทันวันที่ต้องการ (ไม่แตะ task_b/raw_data/)
# ---------------------------------------------------------------

def prepare_b_raw(through_target):
    """สร้างสำเนา raw_data ของ task_b + ต่อแถว Yahoo ที่ขาด (ข้อมูลถึงวันก่อน through_target)"""
    _enter("task_b")
    from tools.fetch_yahoo_daily import plan_fetch
    from config import TICKERS, RAW_DATA_DIR, RAW_DATA_SUFFIX
    B_RAW_DIR.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp(f"{through_target} 06:00", tz=common.BANGKOK).to_pydatetime()
    plans = plan_fetch(now, raw_dir=RAW_DATA_DIR, tickers=TICKERS)   # ตรวจ OHLC ทับของเดิมเป๊ะ เหมือนระบบจริง
    for p in plans:
        name = p["ticker"] + RAW_DATA_SUFFIX
        b = (RAW_DATA_DIR / name).read_bytes()
        prev, lines = p["prev_close"], []
        for d, r in p["new"].iterrows():
            chg = (r["Close"] / prev - 1) * 100
            lines.append(f"{d:%m/%d/%Y},{float(r['Close'])},{float(r['Open'])},{float(r['High'])},"
                         f"{float(r['Low'])},{float(r['Volume']) / 1000},{chg:.2f}%")
            prev = float(r["Close"])
        add = ("\r\n".join(lines) + "\r\n").encode("ascii") if lines else b""
        (B_RAW_DIR / name).write_bytes(b + add)
        last = p["new"].index[-1].date() if len(p["new"]) else p["last"].date()
        print(f"[b-raw] {name}: +{len(lines)} แถว (ถึง {last})")


def run_next_day_b(task, target, reason):
    _enter(task)
    import data_loader
    data_loader.RAW_DATA_DIR = B_RAW_DIR          # raw_data_path() อ่านตัวแปรนี้ตอนเรียก
    import predict_live as pl
    from config import TICKERS, CONFIG_TAG
    if TRAIN_START:
        orig_fl = pl.fit_live_models
        def fit_live_since(X, y, *a, **k):
            X, y = _since(X, y)
            _announce_train(task, X.index)
            return orig_fl(X, y, *a, **k)
        pl.fit_live_models = fit_live_since

    tgt = pd.Timestamp(target)
    df = data_loader.load_stock(TICKERS[0], verbose=False)
    prior = df.index[df.index < tgt]
    if len(prior) == 0 or prior[-1] < tgt - pd.Timedelta(days=5):
        raise RuntimeError(f"raw_data สำเนาไม่ครอบคลุมก่อน {target} -- รัน --prepare-b-raw ก่อน")
    as_of = prior[-1]

    rows = []
    for t in TICKERS:
        rows.extend(pl.predict_one_ticker(t, target, as_of=as_of, expected_cutoff=as_of.date().isoformat()))
    out = pd.DataFrame(rows)
    out["generated_at"] = common.now_iso()
    out["prediction_type"] = "next_day"
    out["target_date"] = tgt.date().isoformat()
    out["code_commit"] = pl.get_code_commit(True)
    out["half_life"] = out["half_life"].astype("Int64")
    out["config_tag"] = CONFIG_TAG
    out["is_dry_run"] = False
    _finish(task, out.to_dict("records"), pl.LOG_COLUMNS, reason)


# ---------------------------------------------------------------
# task_b t1600: สร้าง "snapshot" จาก Yahoo 1h ที่ดึงตอนนี้ (รูปแบบเดียวกับ tools/fetch_yahoo_intraday.py)
# แล้วใช้ bars_for_prediction/predict_ticker ตัวเดิม (validate_for_prediction ถูกข้ามเพราะบังคับ
# ว่าต้องดาวน์โหลดวันเดียวกัน ณ/หลัง 16:00 ซึ่ง backfill ทำไม่ได้โดยนิยาม)
# ---------------------------------------------------------------

def run_t1600_b(task, target, reason):
    _enter(task)
    import predict_1600 as p16
    import live_1600
    from tools import fetch_yahoo_intraday as fy
    tgt = pd.Timestamp(target).normalize()
    snap_dir = DATA_DIR / "task_b_1h_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    orig_fp = p16.fit_predict

    def fp_announce(model, X_train, y_train, X_live):
        if TRAIN_START:
            X_train, y_train = _since(X_train, y_train)
        _announce_train(task, X_train.index)
        return orig_fp(model, X_train, y_train, X_live)
    p16.fit_predict = fp_announce

    winners = p16.load_winners()
    commit = p16._git("rev-parse", "HEAD")
    rows = []
    for t in p16.TICKERS:
        raw = fy.to_locked_format(fy.download_hourly(t))            # yfinance ผ่านแคช -> ข้อมูลชุดเดียวกันทุก run
        csv = snap_dir / f"{t.replace('.', '_')}_1h_{hashlib.sha1(raw.to_csv().encode()).hexdigest()[:10]}.csv"
        if not csv.exists():
            raw.to_csv(csv)
        sha = hashlib.sha256(csv.read_bytes()).hexdigest()
        snap_bars = live_1600.prepare_bars(pd.read_csv(csv, encoding="utf-8-sig"))
        bars, mism = live_1600.merge_with_locked(live_1600.load_ticker_bars(t), snap_bars)
        info = {"sha256": sha, "downloaded_at": "", "source": "yahoo", "path": str(csv.relative_to(ROOT)),
                "mismatches": len(mism)}
        rows += p16.predict_ticker(t, tgt, winners, bars, info)

    gen = common.now_iso()
    out = pd.DataFrame([{
        "generated_at": gen, "prediction_type": p16.PREDICTION_TYPE, "ticker": r["ticker"],
        "target_date": str(tgt.date()), "data_cutoff": r["feature_window_end"],
        "train_rows": r["train_days"], "half_life": "", "model": r["model"],
        "predicted_return": r["predicted_return"], "prev_close": r["close_bar15"],
        "predicted_close": r["predicted_close_bar16"], "n_features": r["n_features"],
        "dataset_sha256": r["dataset_sha256"], "code_commit": commit,
        "config_tag": p16.CONFIG_TAG_1600, "is_dry_run": False,
        "snapshot_path": r["snapshot_path"], "locked_mismatch_bars": r["locked_mismatch_bars"],
        "historical_x_match": r["historical_x_match"],
    } for r in rows])
    _finish(task, out.to_dict("records"), p16.LOG_COLUMNS, reason,
            extra_cols=["snapshot_path", "locked_mismatch_bars", "historical_x_match"])


RUNNERS = {
    "task_a": run_next_day_ac, "task_c": run_next_day_ac, "task_b": run_next_day_b,
    "task_a2": run_t1600_a2, "task_b_1600": run_t1600_b, "task_c_1600": run_t1600_c,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(RUNNERS))
    ap.add_argument("--target-date")
    ap.add_argument("--reason", default="missing", choices=["missing", "partial_models"])
    ap.add_argument("--prepare-b-raw", action="store_true")
    ap.add_argument("--through", help="YYYY-MM-DD วัน target ท้ายสุดที่จะ backfill (ใช้กับ --prepare-b-raw)")
    a = ap.parse_args()
    if a.prepare_b_raw:
        prepare_b_raw(a.through)
        return
    if not (a.task and a.target_date):
        ap.error("ต้องระบุ --task และ --target-date")
    RUNNERS[a.task](a.task, a.target_date, a.reason)


if __name__ == "__main__":
    main()
