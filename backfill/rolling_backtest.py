"""
backfill/rolling_backtest.py — rolling backtest: เทรนด้วยข้อมูลทั้งหมด vs ถ่วงน้ำหนักข้อมูลล่าสุด vs หน้าต่างสั้น
===================================================================================================

    python backfill/rolling_backtest.py --task task_a            # รัน 1 task (บันทึก predictions)
    python backfill/rolling_backtest.py --analyze                # สรุปผลทุก task

ไม่แก้ไฟล์ใดของ task_*/ หรือ backfill/ ที่มีอยู่ — อ่านข้อมูลจากแคชใน backfill/data/ (ออฟไลน์) และเรียก
ฟังก์ชัน feature/target/model ตัวเดิมของแต่ละ task ผลเขียนที่ backfill/rolling_backtest/ เท่านั้น

วิธีจำลอง: ทุกวันทำการ t ในช่วงทดสอบ เทรนด้วยแถวที่ index < t เท่านั้น (label ของแถวเหล่านั้นรู้แล้ว ณ เช้าวัน t)
แล้วทำนายแถว t ด้วย feature ของวัน t ตัวเดียวกับที่ระบบสดใช้ (feature ทุกตัว shift แล้ว / ตัดที่ 15:00)
เพื่อประหยัดเวลา เทรนใหม่ทุก REFIT_EVERY วันทำการ (ทำนาย block ถัดไปด้วยโมเดลเดียว) ทุก config ใช้ block เดียวกัน

config ที่เทียบ (กำหนดก่อนรัน):
  F     = ข้อมูลทั้งหมด น้ำหนักเท่ากัน (ของเดิม)
  H500  = ถ่วงน้ำหนัก half-life 500 แถว    H250 = 250 แถว    H125 = 125 แถว
        w = 0.5 ** (อายุแถว/half_life) แล้ว normalize ให้ mean = 1 (สูตรเดียวกับ task_b/weighting.py)
  W265  = ใช้เฉพาะ 265 แถวล่าสุด (ขนาดเท่าข้อมูลตั้งแต่ 2025-09-08 ถึงตอนนี้)

กติกาตัดสิน (ตั้งไว้ก่อนดูผล):
  เมตริกหลัก  - classification: accuracy ต่อวัน (เฉลี่ยทุกหุ้น×โมเดล) · ราคา: |ราคาที่ทาย - ราคาจริง| เฉลี่ยต่อวัน
  เทียบ config กับ F ด้วย paired bootstrap ข้ามวัน (2,000 รอบ) -> ช่วงเชื่อมั่น 95% ของส่วนต่าง
  config ถือว่า "ดีกว่า F" เมื่อช่วงเชื่อมั่นทั้งช่วงอยู่ฝั่งที่ดีกว่า (ไม่คร่อม 0) เท่านั้น
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402
from common import ROOT, DATA_DIR  # noqa: E402

OUT_DIR = HERE / "rolling_backtest"
TEST_START, TEST_END = pd.Timestamp("2025-12-22"), pd.Timestamp("2026-10-09")
REFIT_EVERY = 5
REFIT_OVERRIDE = {"task_b": 10, "task_b_1600": 10}      # ANN หลายตัว/หลาย seed ช้า -> เทรนใหม่ทุก 10 วัน
CONFIGS = {"F": ("none", None), "H500": ("hl", 500), "H250": ("hl", 250), "H125": ("hl", 125), "W265": ("win", 265)}
TASKS = ["task_a", "task_c", "task_b", "task_a2", "task_c_1600", "task_b_1600"]
TICKERS = ["KBANK.BK", "ADVANC.BK"]


def recency_weights(n, half_life):
    age = np.arange(n - 1, -1, -1, dtype=float)
    w = 0.5 ** (age / float(half_life))
    return w / w.mean()


def _cache_big(ticker, interval, min_rows):
    idx = pd.read_json(DATA_DIR / "yf_cache" / "index.jsonl", lines=True)
    stem = ticker.replace(".", "_")
    f = idx[(idx.file.str.startswith(f"{stem}_{interval}")) & (idx.rows > min_rows)].file.iloc[0]
    d = pd.read_pickle(DATA_DIR / "yf_cache" / f)
    d.columns = d.columns.get_level_values(0)
    return d


def daily_df(ticker, clean):
    d = _cache_big(ticker, "1d", 2000)[["Open", "High", "Low", "Close", "Volume"]].copy()
    d.index = pd.to_datetime(d.index)
    d.index.name = "Date"
    return clean(d, verbose=False)


# ---------------------------------------------------------------
# adapters: คืน {ticker: dict(X, y, aux, kind, make_models, fit_kw)}
# aux = คอลัมน์ที่ใช้ประเมิน · kind = cls_parity | cls | reg
# ---------------------------------------------------------------

def _enter(d):
    os.chdir(ROOT / d)
    sys.path.insert(0, str(ROOT / d))


def adapter_task_a():
    _enter("task_a")
    from data_loader import clean
    from features import build_features
    from targets import build_targets
    from models import get_classifiers
    from main import _prepare
    out = {}
    for t in TICKERS:
        df = daily_df(t, clean)
        X = build_features(df, verbose=False)
        tg = build_targets(df, verbose=False)
        Xa, yf, ex = _prepare(X, tg["y_flip"], tg[["prev_parity", "y_parity"]])
        out[t] = dict(X=Xa, y=yf, aux=ex.rename(columns={"y_parity": "actual"}), kind="cls_parity", make=get_classifiers)
    return out


def adapter_task_c():
    _enter("task_c")
    from data_loader import clean
    from features import build_features
    from targets import build_targets
    from models import get_classifiers
    from main import _prepare
    out = {}
    for t in TICKERS:
        df = daily_df(t, clean)
        X = build_features(df, verbose=False)
        tg = build_targets(df, verbose=False)
        Xa, y = _prepare(X, tg["y_updown"])
        out[t] = dict(X=Xa, y=y, aux=pd.DataFrame({"actual": y}), kind="cls", make=get_classifiers)
    return out


def adapter_task_b():
    _enter("task_b")
    import data_loader
    data_loader.RAW_DATA_DIR = DATA_DIR / "task_b_raw_data"
    from features import build_features
    from targets import build_targets
    from splits import prepare_xy
    from models import get_regressors
    out = {}
    for t in TICKERS:
        df = data_loader.load_stock(t, verbose=False)
        X = build_features(df, verbose=False)
        tg = build_targets(df, verbose=False)
        Xp, yp = prepare_xy(X, tg["y_return"], verbose=False)
        aux = pd.DataFrame({"prev_close": df["Close"].shift(1), "close": df["Close"]}).loc[Xp.index]
        out[t] = dict(X=Xp, y=yp, aux=aux, kind="reg", make=get_regressors, wrap_target=True)
    return out


def adapter_task_a2():
    _enter("task_a2")
    sys.path.insert(0, str(ROOT / "task_a"))
    from data_loader import clean
    from rounding import to_int_baht, parity as parity_fn
    import features_a2
    from main_a2 import get_classifiers
    out = {}
    for t in TICKERS:
        short = t.replace(".BK", "")
        h = _cache_big(t, "1h", 1000)
        idx = h.index
        idx = idx.tz_localize("UTC") if idx.tz is None else idx
        h.index = idx.tz_convert("Asia/Bangkok")
        df_d = daily_df(t, clean)
        intraday = features_a2.build_intraday_features(t, short, df_1h=h)
        ctx = features_a2.build_daily_context_features(t, df=df_d)
        close = df_d["Close"]
        dates = [d for d in intraday.index.intersection(ctx.index) if d in close.index]
        X = pd.concat([intraday.loc[dates].drop(columns=["price_1600"]), ctx.loc[dates]], axis=1)
        X = X.replace([np.inf, -np.inf], np.nan)
        y_par = parity_fn(to_int_baht(close.loc[dates])).values
        p16 = parity_fn(to_int_baht(intraday.loc[dates, "price_1600"])).values
        yflip = pd.Series((y_par + p16) % 2, index=dates)
        aux = pd.DataFrame({"prev_parity": p16, "actual": y_par}, index=dates)
        out[t] = dict(X=X, y=yflip, aux=aux, kind="cls_parity", make=get_classifiers)
    return out


def adapter_task_c_1600():
    _enter("task_c")
    import yfinance  # noqa: F401
    common.install_yf_cache()
    import predict_live_1600 as pl
    from intraday_task2 import build_task2_dataset, _prepare
    from models import get_classifiers
    out = {}
    for t in TICKERS:
        hourly = pl.fetch_live_1h(t)                      # แคช (ไม่มีส่วนที่ขึ้นกับวันที่)
        dc = _cache_big(t, "1d", 2000)["Close"].dropna()
        dc.index = pd.to_datetime(dc.index).normalize()
        X, y, z = build_task2_dataset(hourly, daily_close=dc)
        X, y, _ = _prepare(X, y, z)
        X.index = pd.DatetimeIndex(X.index).tz_localize(None)
        y.index = X.index
        out[t] = dict(X=X, y=y, aux=pd.DataFrame({"actual": y}), kind="cls", make=get_classifiers)
    return out


def adapter_task_b_1600():
    _enter("task_b")
    import live_1600
    from intraday_1600 import build_dataset
    import predict_1600 as p16
    from tune_1600 import GRIDS, make_live_model
    winners = p16.load_winners()
    out = {}
    for t in TICKERS:
        stem = t.replace(".", "_")
        snaps = sorted((DATA_DIR / "task_b_1h_snapshots").glob(f"{stem}_1h_*.csv"), key=os.path.getmtime)
        snap = live_1600.prepare_bars(pd.read_csv(snaps[-1], encoding="utf-8-sig"))
        bars, _ = live_1600.merge_with_locked(live_1600.load_ticker_bars(t), snap)
        X, y, info, _ = build_dataset(bars)
        aux = pd.DataFrame({"C15": info["C15"]}).loc[X.index]
        aux["close"] = aux["C15"] * (1 + y.loc[X.index])
        aux["prev_close"] = aux["C15"]
        make = (lambda w=winners: {f: make_live_model(f, GRIDS[f][c]) for f, c in w.items()})
        out[t] = dict(X=X, y=y, aux=aux, kind="reg", make=make, wrap_target=True)
    return out


ADAPTERS = {"task_a": adapter_task_a, "task_c": adapter_task_c, "task_b": adapter_task_b, "task_a2": adapter_task_a2,
            "task_c_1600": adapter_task_c_1600, "task_b_1600": adapter_task_b_1600}


# ---------------------------------------------------------------
# backtest
# ---------------------------------------------------------------

def fit_predict(d, name, model, Xtr, ytr, w, Xte):
    kw = {} if w is None else {"model__sample_weight": w}
    if d.get("wrap_target"):
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.preprocessing import StandardScaler
        m = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
        m.fit(Xtr, ytr, **kw)
        return m.predict(Xte), None
    model.fit(Xtr, ytr, **kw)
    pred = model.predict(Xte)
    try:
        proba = model.predict_proba(Xte)[:, 1]
    except Exception:
        proba = np.full(len(Xte), np.nan)
    return pred, proba


def run(task, limit=None):
    t0 = time.time()
    data = ADAPTERS[task]()
    every = REFIT_OVERRIDE.get(task, REFIT_EVERY)
    rows = []
    for tk, d in data.items():
        X, y = d["X"], d["y"]
        test = X.index[(X.index >= TEST_START) & (X.index <= TEST_END)]
        if limit:
            test = test[:limit]
        for i in range(0, len(test), every):
            blk = test[i:i + every]
            s = blk[0]
            tr = X.index < s
            for cfg, (kind, p) in CONFIGS.items():
                Xtr, ytr = X[tr], y[tr]
                w = None
                if kind == "win":
                    Xtr, ytr = Xtr.iloc[-p:], ytr.iloc[-p:]
                elif kind == "hl":
                    w = recency_weights(len(Xtr), p)
                for name, model in d["make"]().items():
                    pred, proba = fit_predict(d, name, model, Xtr, ytr, w, X.loc[blk])
                    for j, day in enumerate(blk):
                        r = {"task": task, "ticker": tk, "config": cfg, "model": name, "date": day.date().isoformat(),
                             "pred": float(pred[j]), "proba": float(proba[j]) if proba is not None else np.nan,
                             "train_rows": len(Xtr), "train_start": Xtr.index[0].date().isoformat()}
                        r.update({k: float(v) for k, v in d["aux"].loc[day].items()})
                        rows.append(r)
            print(f"[{task}] {tk} {blk[0].date()}  ({i + len(blk)}/{len(test)})  {time.time() - t0:.0f}s", flush=True)
    OUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / f"{task}_preds.csv", index=False)
    print(f"[{task}] เสร็จ {len(rows)} แถว {time.time() - t0:.0f}s")


# ---------------------------------------------------------------
# analyze
# ---------------------------------------------------------------

def per_day_metric(df, task):
    """คืน DataFrame index=date คอลัมน์=config ของเมตริกหลักต่อวัน (เฉลี่ยทุกหุ้น×โมเดล) -- สูงดีกว่า (acc) / ต่ำดีกว่า (err)"""
    if task in ("task_a", "task_a2"):
        pp = (df["prev_parity"] + df["pred"]) % 2
        df = df.assign(score=(pp == df["actual"]).astype(float))
    elif task in ("task_c", "task_c_1600"):
        df = df.assign(score=(df["pred"] == df["actual"]).astype(float))
    else:
        df = df.assign(score=(df["prev_close"] * (1 + df["pred"]) - df["close"]).abs())
    return df.groupby(["date", "config"])["score"].mean().unstack()


def naive_and_base(df, task):
    if task in ("task_b", "task_b_1600"):
        n = (df["prev_close"] - df["close"]).abs()
        return float(n.groupby(df["date"]).mean().mean())
    return None


def analyze():
    rng = np.random.default_rng(0)
    rows, extra = [], []
    for task in TASKS:
        p = OUT_DIR / f"{task}_preds.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        m = per_day_metric(df, task)
        higher_better = task not in ("task_b", "task_b_1600")
        nd = len(m)
        boots = rng.integers(0, nd, size=(2000, nd))
        for cfg in m.columns:
            diff = (m[cfg] - m["F"]).to_numpy()
            if not higher_better:
                diff = -diff                       # ให้ + = ดีกว่า F เสมอ
            bs = diff[boots].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            verdict = "-" if cfg == "F" else ("ดีกว่า F" if lo > 0 else "แย่กว่า F" if hi < 0 else "ไม่ต่างจาก F")
            rows.append({"task": task, "config": cfg, "days": nd, "metric": "accuracy" if higher_better else "MAE_baht",
                         "value": round(float(m[cfg].mean()), 4 if higher_better else 3),
                         "diff_vs_F(+=ดีกว่า)": round(float(diff.mean()), 4), "CI95": f"[{lo:+.4f}, {hi:+.4f}]", "verdict": verdict})
        nb = naive_and_base(df, task)
        if nb is not None:
            extra.append((task, round(nb, 3)))
    rep = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(rep.to_string(index=False))
    print("naive MAE (ทายว่าราคาเท่าเดิม):", extra)
    rep.to_csv(OUT_DIR / "rolling_backtest_report.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--limit", type=int, help="จำกัดจำนวนวันทดสอบ (ทดสอบความเร็ว)")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze()
    elif a.task:
        common.install_yf_cache()
        os.environ["BACKFILL_OFFLINE"] = "1"
        run(a.task, a.limit)
    else:
        ap.error("ระบุ --task หรือ --analyze")
