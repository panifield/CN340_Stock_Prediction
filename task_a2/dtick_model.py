"""
dtick_model.py — Task A2: ทายจำนวน tick ที่ราคาขยับจาก 16:00 ถึงราคาปิด แล้วแปลงเป็น P(คี่)
===========================================================================================

target : Δtick = ตำแหน่งบน tick grid ของราคาปิดทางการ - ของราคา ณ 16:00 ตัดเป็น 5 คลาส
         {<=-2, -1, 0, +1, >=+2}  (ราคาปิดจาก verified_daily ไม่ใช้ Yahoo ตรงๆ)
         นับ tick ข้ามช่วงราคาได้ถูก (เช่น 199.50 -> 201 = +2: 199.50->200 ครึ่งบาท 1 tick, 200->201 1 tick)
P(คี่) : แต่ละคลาสแปลงเป็นราคาปิดที่เป็นไปได้ แล้ว parity_label (label floor)
         คลาสปลายเปิด (<=-2, >=+2) ใช้สัดส่วนของ Δtick จริงในคลาสนั้นจากชุด train ของ fold นั้นเท่านั้น
feature: หน่วย tick ทั้งหมด จากแท่ง 1h ถึงแท่ง 15:00 (= ราคา ณ 16:00) + feature รายวันของงาน A (ถึงวัน t-1)
การแบ่ง: แถวและการแบ่งเดียวกับ main_a2.py (dev pool 80% แรก walk-forward 5 fold, holdout 20%)
         เลือกโมเดลบน OOF ของ dev pool แล้วดู holdout ครั้งเดียว

ตาราง tick ของ SET (rounding.tick_size) ตรวจแล้ว 2026-10-24: ตรงกับหน้า Price bands and tick sizes ของ SET
(25-100 = 0.25, 100-200 = 0.50, 200-400 = 1.00, >=400 = 2.00) และราคา OHLC ของ KBANK/ADVANC ตั้งแต่ 2016
อยู่บน grid นี้ทุกแถว

รัน:  cd task_a2 ; python dtick_model.py      -> results_floor/dtick/
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

HERE = Path(__file__).resolve().parent
TASK_A = HERE.parent / "task_a"
sys.path.insert(0, str(TASK_A))
sys.path.insert(0, str(HERE))

from config import ANN_PARAMS, LGBM_PARAMS, LOGREG_PARAMS, RANDOM_STATE  # noqa: E402
from rounding import parity_label  # noqa: E402
from splits import walk_forward_splits  # noqa: E402
from eval_stats import binom_ci, mcnemar_test  # noqa: E402
import features as task_a_features  # noqa: E402
import features_a2  # noqa: E402
import verified_daily  # noqa: E402

OUT = Path(os.environ.get("DTICK_OUT") or HERE / "results_floor" / "dtick")
TICKERS = {"KBANK.BK": "KBANK_BK", "ADVANC.BK": "ADVANC_BK"}
TEST_FRACTION, N_SPLITS, MIN_TRAIN = 0.20, 5, 150
CLASSES = [-2, -1, 0, 1, 2]               # -2 = <=-2, +2 = >=+2
BUCKETS = {"100-200 (tick 0.5)": (100, 200), "200-400 (tick 1)": (200, 400)}

# ---------------------------------------------------------------
# tick grid (หน่วยสตางค์ กัน float)
# ---------------------------------------------------------------
_BOUNDS = [0, 200, 500, 1000, 2500, 10000, 20000, 40000]         # สตางค์ ตามตาราง SET
_TICKS = [1, 2, 5, 10, 25, 50, 100, 200]                          # สตางค์
_START_IDX = np.concatenate([[0], np.cumsum(np.diff(_BOUNDS) // np.array(_TICKS[:-1]))])


def tick_index(price):
    """ตำแหน่งบน tick grid ของ SET (จำนวนเต็ม) -- ต่างกัน 1 = ห่างกัน 1 tick ข้ามช่วงราคาได้"""
    s = np.rint(np.asarray(price, dtype=float) * 100)
    seg = np.searchsorted(_BOUNDS, s, side="right") - 1
    off = s - np.asarray(_BOUNDS)[seg]
    tick = np.asarray(_TICKS)[seg]
    if np.any(np.mod(off, tick) != 0):
        raise ValueError("ราคาไม่อยู่บน tick grid ของ SET")
    return (_START_IDX[seg] + off // tick).astype(np.int64)


def price_from_index(idx):
    """กลับจาก tick_index เป็นราคา (บาท)"""
    idx = np.asarray(idx, dtype=np.int64)
    seg = np.searchsorted(_START_IDX, idx, side="right") - 1
    satang = np.asarray(_BOUNDS)[seg] + (idx - _START_IDX[seg]) * np.asarray(_TICKS)[seg]
    return satang / 100.0


def dtick_class(dtick):
    return np.clip(np.asarray(dtick), -2, 2).astype(int)


# ---------------------------------------------------------------
# feature (หน่วย tick ทั้งหมด)
# ---------------------------------------------------------------
DAILY_TICK_COLS = [f"{c}_prev" for c in [
    "dtick_lag1", "dtick_lag2", "dtick_lag3", "dtick_parity_lag1", "dtick_parity_lag2",
    "zero_rate_20", "atr14_ticks", "std20_ticks", "range_ticks_lag1", "tick_regime", "n_mod2"]]


def intraday_tick_features(df_1h, dates):
    """แท่ง 1h ถึงแท่ง 15:00 ของแต่ละวัน (ไม่ใช้แท่ง 16:00 ขึ้นไป) -> feature หน่วย tick"""
    df_1h = df_1h.sort_index()
    rows = {}
    for date, g in df_1h.groupby(df_1h.index.date):
        ts = pd.Timestamp(date)
        if ts not in dates:
            continue
        g = g[[t.hour < 16 for t in g.index]]                # ห้ามเห็นข้อมูลตั้งแต่ 16:00
        g = g.set_index(g.index.hour)
        if 15 not in g.index:
            continue
        p = g.loc[15, "Close"]
        i15 = int(tick_index(p))
        tick = float(features_a2.tick_size(pd.Series([p])).iloc[0])
        frac_satang = np.rint(p * 100) % 100
        ticks_above_int = (frac_satang / 100) / tick                 # ลงกี่ tick ถึง x.00 (จุดเปลี่ยนคู่/คี่)
        ticks_to_next_int = ((100 - frac_satang) / 100) / tick        # ขึ้นกี่ tick ถึงบาทถัดไป
        rows[ts] = {
            "tick_1600": tick,
            "ticks_above_int": ticks_above_int,
            "ticks_to_next_int": ticks_to_next_int,
            "dtick_last1h": i15 - int(tick_index(g.loc[14, "Close"])) if 14 in g.index else np.nan,
            "dtick_last2h": i15 - int(tick_index(g.loc[13, "Close"])) if 13 in g.index else np.nan,
            "dtick_since_open": i15 - int(tick_index(g["Open"].iloc[0])),
            "range_ticks_so_far": int(tick_index(g["High"].max()) - tick_index(g["Low"].min())),
            "high_minus_1600_ticks": int(tick_index(g["High"].max())) - i15,
            "the_1600_minus_low_ticks": i15 - int(tick_index(g["Low"].min())),
            "n_mod2_1600": i15 % 2,
            "n_bars": len(g),
        }
    return pd.DataFrame(rows).T.sort_index()


def build_dataset(ticker, safe):
    """แถว = วันเดียวกับ A2 dataset (มีแท่ง 15:00 + ราคาปิดทางการ + feature รายวัน)"""
    daily = verified_daily.load_verified_history(ticker)
    df_1h = features_a2.load_1h(safe)
    intra_ref = features_a2.build_intraday_features(ticker, safe, df_1h=df_1h)
    ctx = task_a_features.build_features(daily, verbose=False)
    common = intra_ref.index.intersection(daily.index).intersection(ctx.index)
    feats = intraday_tick_features(df_1h, set(common))
    common = common.intersection(feats.index)

    p1600 = intra_ref.loc[common, "price_1600"].astype(float)
    close = daily.loc[common, "Close"].astype(float)
    open_today = daily.loc[common, "Open"].astype(float)
    prev_close = daily["Close"].shift(1).loc[common].astype(float)
    X = pd.concat([feats.loc[common].astype(float), ctx.loc[common, DAILY_TICK_COLS]], axis=1)
    X["gap_ticks"] = tick_index(open_today) - tick_index(prev_close)
    X = X.replace([np.inf, -np.inf], np.nan)

    i1600 = tick_index(p1600)
    dtick = tick_index(close) - i1600
    return {
        "X": X, "dtick": pd.Series(dtick, index=common), "i1600": pd.Series(i1600, index=common),
        "p1600": p1600, "close": close,
        "y": pd.Series(parity_label(close).values, index=common),
        "base": pd.Series(parity_label(p1600).values, index=common),
    }


# ---------------------------------------------------------------
# P(คี่) จาก P(คลาส)
# ---------------------------------------------------------------
def open_class_weights(dtick_train):
    """สัดส่วน Δtick จริงภายในคลาสปลายเปิด จาก train เท่านั้น -> {คลาส: {k: น้ำหนัก}}"""
    d = np.asarray(dtick_train)
    out = {}
    for cls, mask in ((-2, d <= -2), (2, d >= 2)):
        if mask.sum() == 0:
            out[cls] = {cls: 1.0}
        else:
            ks, n = np.unique(d[mask], return_counts=True)
            out[cls] = dict(zip(ks.tolist(), (n / n.sum()).tolist()))
    return out


def p_odd_from_classes(proba, classes_present, i1600, weights):
    """proba: (n, len(classes_present)) -> P(ราคาปิดเป็นคี่) แต่ละแถว"""
    i1600 = np.asarray(i1600)
    p_odd = np.zeros(len(i1600))
    for j, cls in enumerate(classes_present):
        ks = weights[cls] if cls in (-2, 2) else {cls: 1.0}
        for k, w in ks.items():
            odd = parity_label(price_from_index(i1600 + k)).values
            p_odd += proba[:, j] * w * odd
    return p_odd


# ---------------------------------------------------------------
# โมเดล
# ---------------------------------------------------------------
def _scaled(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", est)])


def _unscaled(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", est)])


MODELS = {
    "Δtick LogReg": lambda: _scaled(LogisticRegression(**{**LOGREG_PARAMS, "max_iter": 3000})),
    "Δtick LightGBM": lambda: _unscaled(LGBMClassifier(**{**LGBM_PARAMS, "objective": "multiclass"})),
    "Δtick MLP": lambda: _scaled(MLPClassifier(**ANN_PARAMS)),
}


def fit_predict_podd(ctor, X_tr, cls_tr, dtick_tr, X_te, i1600_te):
    m = ctor()
    m.fit(X_tr, cls_tr)
    proba = m.predict_proba(X_te)
    return p_odd_from_classes(proba, list(m.classes_), i1600_te, open_class_weights(dtick_tr))


def run_ticker(ticker, safe, flip_oof=None, flip_hold=None):
    d = build_dataset(ticker, safe)
    X, y, base = d["X"], d["y"], d["base"]
    cls = pd.Series(dtick_class(d["dtick"]), index=X.index)
    n = len(X)
    n_pool = n - int(round(n * TEST_FRACTION))
    Xp, Xh = X.iloc[:n_pool], X.iloc[n_pool:]

    oof = {k: np.full(n_pool, np.nan) for k in MODELS}
    for tr, te in walk_forward_splits(Xp, n_splits=N_SPLITS, min_train=MIN_TRAIN):
        for name, ctor in MODELS.items():
            oof[name][te] = fit_predict_podd(ctor, Xp.iloc[tr], cls.iloc[tr].values,
                                             d["dtick"].iloc[tr].values, Xp.iloc[te],
                                             d["i1600"].iloc[te].values)
    oof = pd.DataFrame(oof, index=Xp.index).dropna()
    hold = pd.DataFrame({name: fit_predict_podd(ctor, Xp, cls.iloc[:n_pool].values,
                                                d["dtick"].iloc[:n_pool].values, Xh,
                                                d["i1600"].iloc[n_pool:].values)
                         for name, ctor in MODELS.items()}, index=Xh.index)
    return d, cls, n_pool, oof, hold


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(TASK_A))
    import five_analyses as FA     # โมเดล "พลิก/ไม่พลิก" เดิมบนแถว/การแบ่งเดียวกัน (label floor)

    rows, dist_rows, choice = [], [], []
    m_holdout = 2 * len(TICKERS)    # holdout: Δtick ที่เลือก + flip ที่เลือก ต่อหุ้น -> Bonferroni
    for ticker, safe in TICKERS.items():
        t0 = time.time()
        d, cls, n_pool, oof, hold = run_ticker(ticker, safe)
        fd, f_oof, f_hold = FA.cached("A2", ticker, "floor")
        assert fd.X.index.equals(d["X"].index), "แถวไม่ตรงกับ A2 dataset เดิม"

        y, base, p = d["y"], d["base"], d["p1600"]
        bucket = pd.Series("other", index=y.index)
        for b, (lo, hi) in BUCKETS.items():
            bucket[(p >= lo) & (p < hi)] = b

        # การกระจายของ Δtick แยกช่วงราคา (ข้อมูลทั้งหมด)
        for b in BUCKETS:
            m = bucket == b
            if m.sum():
                vc = cls[m].value_counts(normalize=True).reindex(CLASSES, fill_value=0)
                dist_rows.append({"ticker": ticker, "price_range": b, "n": int(m.sum()),
                                  **{f"class_{c:+d}": round(v, 4) for c, v in vc.items()},
                                  "flip_rate": round(float((y[m] != base[m]).mean()), 4)})

        preds_oof = {name: (oof[name].values >= 0.5).astype(float) for name in MODELS}
        preds_oof.update({f"Flip {n}": FA.parity_pred(f_oof[n].values, base.loc[f_oof.index].values)
                          for n in FA.MODEL_NAMES})
        preds_hold = {name: (hold[name].values >= 0.5).astype(float) for name in MODELS}
        preds_hold.update({f"Flip {n}": FA.parity_pred(f_hold[n].values, base.loc[f_hold.index].values)
                           for n in FA.MODEL_NAMES})
        assert f_oof.index.equals(oof.index) and f_hold.index.equals(hold.index)

        yo, bo = y.loc[oof.index].values, base.loc[oof.index].values
        yh, bh = y.loc[hold.index].values, base.loc[hold.index].values
        best_dt = max(MODELS, key=lambda k: FA.acc(preds_oof[k], yo))
        best_fl = max([f"Flip {n}" for n in FA.MODEL_NAMES], key=lambda k: FA.acc(preds_oof[k], yo))

        for split, idx, preds, yt, bt in (("dev OOF", oof.index, preds_oof, yo, bo),
                                          ("holdout", hold.index, preds_hold, yh, bh)):
            bk = bucket.loc[idx].values
            names = (["Baseline: Persistence-1600"] + list(preds)) if split == "dev OOF" else \
                ["Baseline: Persistence-1600", best_dt, best_fl]
            for name in names:
                pr = bt if name.startswith("Baseline") else preds[name]
                for b in ["all"] + list(BUCKETS):
                    m = np.ones(len(yt), bool) if b == "all" else bk == b
                    if m.sum() == 0:
                        continue
                    k = int(np.sum(pr[m] == yt[m]))
                    lo, hi = binom_ci(k, int(m.sum()))
                    r = {"ticker": ticker, "split": split, "model": name, "price_range": b,
                         "n": int(m.sum()), "accuracy": round(k / m.sum(), 4),
                         "ci": f"{lo:.3f}-{hi:.3f}",
                         "persistence": round(FA.acc(bt[m], yt[m]), 4)}
                    if not name.startswith("Baseline"):
                        mc = mcnemar_test(yt[m], pr[m], bt[m])
                        r["mcnemar_p"] = round(mc["pvalue"], 4)
                        if split == "holdout" and b == "all":
                            r["p_bonferroni"] = round(min(1.0, mc["pvalue"] * m_holdout), 4)
                    rows.append(r)
        choice.append({"ticker": ticker, "selected_dtick_model": best_dt,
                       "selected_flip_model": best_fl, "n_pool": n_pool, "n_holdout": len(hold)})
        print(f"  {ticker}: เลือก {best_dt} / {best_fl} จาก dev OOF ({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "dtick_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dist_rows).to_csv(OUT / "dtick_class_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(choice).to_csv(OUT / "dtick_model_choice.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(pd.DataFrame(dist_rows).to_string(index=False))
    print(res.to_string(index=False))
    print(f"\nผลอยู่ที่ {OUT}")


if __name__ == "__main__":
    main()
