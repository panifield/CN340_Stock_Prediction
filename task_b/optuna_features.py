"""
optuna_features.py — ค้น feature group + hyperparameter ด้วย Optuna (feedback ข้อ 3)
=====================================================================================

*** ไฟล์นี้ทำอะไร ***
ใช้ Optuna (TPE — R13 / R14) ค้น **feature group 8 กลุ่ม (เปิด/ปิด)** พร้อมกับ
**hyperparameter ที่ช่วงกว้างกว่า grid ของ `TUNING_PLAN.md` ทั้งสองฝั่ง** โดยเห็นเฉพาะ
fold 1-2 แล้วยืนยันผู้ชนะบน fold 3 ที่ Optuna ไม่เคยเห็น

*** ทำไมต้องมี / ตอบ feedback ข้อไหน ***
feedback ข้อ 3 (14 ก.ย. 2026) ขอให้ใช้ HyperOpt หา feature · `tune.py` เดิมเป็น grid search
hyperparameter อย่างเดียว feature fix 29 ตัวตายตัว ไม่มีกลไกเปิด/ปิด feature เลย

*** แผนประกาศไว้ก่อนรันใน results/feedback/OPTUNA_PLAN.md ***
SHA-256 ของแผนถูกบันทึกลงทุกแถวของผลลัพธ์ (`plan_sha256`) · โค้ดนี้ต้องทำตามไฟล์นั้นเป๊ะ

*** ขอบเขต ***
- ข้อมูลตัดที่ SPLIT_BY_DATE["val_end"] ก่อนสร้าง feature แล้ว assert -> ไม่มีทางแตะ test
- ไม่แตะ fixed val ของ main.py
- **ผลรายงานเท่านั้น ห้ามนำผู้ชนะไปใส่ config.py / features.py** (OPTUNA_PLAN ข้อ 7)

    python optuna_features.py --run
    python optuna_features.py --run --smoke     # 3 trials/โมเดล ลงโฟลเดอร์ชั่วคราว

ผลลัพธ์: results/feedback/optuna_trials.csv · optuna_summary.csv ·
         optuna_group_frequency.csv · optuna_ablation.csv · optuna_null.csv ·
         optuna_history.png · optuna_group_frequency.png · optuna_fold3.png
"""

import argparse
import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

import optuna

from config import (
    TICKERS, OUTPUT_DIR, BASE_DIR, SPLIT_BY_DATE, ANN_SEEDS, ANN_PARAMS,
    RF_PARAMS, XGB_PARAMS, CONFIG_TAG,
)
from baselines import baseline_naive_return
from evaluate import regression_metrics, prediction_shape
from features_experimental import (
    GROUPS, GROUP_NAMES, CROSS_GROUP, build_grouped_features, load_pair,
    select_groups, verify_no_leak_experimental,
)
from splits import prepare_xy, walk_forward_splits
from stats_tests import diebold_mariano
from targets import build_targets
from tune import make_model, MODEL_NAMES, N_SPLITS, MIN_TRAIN, TUNING_ANN_SEEDS

FEEDBACK_DIR = OUTPUT_DIR / "feedback"
PLAN_PATH = FEEDBACK_DIR / "OPTUNA_PLAN.md"

# ---------------------------------------------------------------
# ค่าที่ประกาศไว้ใน OPTUNA_PLAN.md -- ห้ามแก้หลังเห็นผล
# ---------------------------------------------------------------
N_TRIALS = {"Random Forest": 60, "XGBoost": 60, "ANN (MLP)": 40}
SAMPLER_SEED = 0
PENALTY = 10.0                       # objective ของ trial ที่ไม่ผ่าน guard
GUARD_MIN_STDRATIO = 0.05
SEARCH_FOLDS = (1, 2)                # Optuna เห็นเฉพาะ fold 1-2 (นับเริ่มที่ 1)
CONFIRM_FOLD = 3                     # fold 3 = confirmation ไม่ถูกค้นเห็น
ALPHA = 0.05
TOP_K = 10                           # top-10 trials สำหรับ feature group frequency
N_PERMUTATIONS = 3                   # null test
PHASE1C_GROUPS = [g for g in GROUP_NAMES if g != CROSS_GROUP]   # = 29 feature เดิม

ANN_HIDDEN_CHOICES = ["(4, 2)", "(8, 4)", "(16, 8)", "(32, 16)"]
RF_MAX_FEATURES_CHOICES = [0.3, 0.5, 0.7, 1.0]


def plan_sha256():
    if not PLAN_PATH.exists():
        raise SystemExit(f"ไม่พบ {PLAN_PATH} -- ต้องเขียนแผนก่อนรัน (OPTUNA_PLAN ข้อ 0)")
    return hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()


def _rel(path):
    """path สั้น ๆ สำหรับพิมพ์ -- โหมด --smoke เขียนลงโฟลเดอร์ชั่วคราวนอก BASE_DIR"""
    try:
        return Path(path).relative_to(BASE_DIR)
    except ValueError:
        return Path(path)


def n_parallel():
    """ไม่เกิน cpu_count() - 2 process และไม่เกินจำนวน eval ต่อ trial (4)"""
    return min(len(TICKERS) * len(SEARCH_FOLDS), max(1, (os.cpu_count() or 2) - 2))


# ---------------------------------------------------------------
# ข้อมูล -- ตัดที่ val_end ก่อนสร้าง feature เหมือน tune.load_dev_data()
# ---------------------------------------------------------------
def load_dev_data_grouped(verbose=True):
    """
    คืน {ticker: (X_all, y, folds, groups)} -- X_all มีทุกคอลัมน์ของทุกกลุ่ม (33 ตัว)

    ใช้ `splits.prepare_xy` + `splits.walk_forward_splits` ตัวเดียวกับ `tune.py`
    -> fold เหมือนกันเป๊ะ · ไม่เขียน logic เตรียมข้อมูลใหม่
    """
    val_end = pd.Timestamp(SPLIT_BY_DATE["val_end"])
    data = {}
    for ticker in TICKERS:
        df, partner = load_pair(ticker, verbose=False)
        # 1) ตัดข้อมูล **ก่อน** สร้าง feature -- ทั้งหุ้นหลักและหุ้นคู่
        df, partner = df.loc[:val_end], partner.loc[:val_end]
        assert df.index[-1] <= val_end and partner.index[-1] <= val_end
        X_all, groups = build_grouped_features(df, partner, verbose=False)
        verify_no_leak_experimental(df, partner, X_all, sample_idx=100)
        targets = build_targets(df, verbose=False)
        X_all, y = prepare_xy(X_all, targets["y_return"], verbose=False)
        # 2) assert เชิงโครงสร้าง (เหมือน tune.load_dev_data)
        assert X_all.index[-1] <= val_end, "optuna_features เห็นแถวที่เลย val_end -- หยุด"
        folds = list(walk_forward_splits(X_all, n_splits=N_SPLITS, min_train=MIN_TRAIN))
        assert len(folds) == N_SPLITS
        for tr, ev in folds:
            assert tr[-1] < ev[0], "fold ข้ามเวลา"
        if verbose:
            print(f"[data] {ticker}: {X_all.shape} "
                  f"({X_all.index[0].date()} -> {X_all.index[-1].date()}) · folds: "
                  + " | ".join(f"train[0:{len(tr)}] eval[{ev[0]}:{ev[-1] + 1}]"
                               for tr, ev in folds))
        data[ticker] = (X_all, y, folds, groups)
    return data


# ---------------------------------------------------------------
# งานหนึ่งชิ้น = หนึ่ง (ticker, fold) -- ต้องอยู่ระดับ module เพื่อให้ pickle ได้
# ---------------------------------------------------------------
def eval_task(name, override, X_sub, y, tr, ev, ann_seeds):
    """
    เทรน + ประเมินหนึ่ง evaluation · คืน (MAE_return, StdRatio)

    ใช้ `tune.make_model` + `TransformedTargetRegressor(StandardScaler())` ชุดเดียวกับ
    `tune.py` -> scaler ของ X (ใน Pipeline) และของ y fit จาก train ของ fold นั้นเท่านั้น
    ทุกโมเดล n_jobs = 1 (บังคับจาก RF_PARAMS / XGB_PARAMS)
    """
    X_tr, y_tr = X_sub.iloc[tr], y.iloc[tr]
    X_ev, y_ev = X_sub.iloc[ev], y.iloc[ev]
    assert X_tr.index[-1] < X_ev.index[0], "fold ข้ามเวลา"
    wrapped = TransformedTargetRegressor(
        regressor=make_model(name, override, ann_seeds=ann_seeds),
        transformer=StandardScaler(),
    )
    wrapped.fit(X_tr, y_tr)
    pred = wrapped.predict(X_ev)
    return (regression_metrics(y_ev, pred)["MAE_return"],
            prediction_shape(y_ev, pred)["StdRatio"])


def eval_task_pred(name, override, X_sub, y, tr, ev, ann_seeds, perm_seed=None):
    """เหมือน eval_task แต่คืน prediction ด้วย (ใช้กับ fold 3 / null test)"""
    X_tr, y_tr = X_sub.iloc[tr], y.iloc[tr]
    X_ev, y_ev = X_sub.iloc[ev], y.iloc[ev]
    assert X_tr.index[-1] < X_ev.index[0], "fold ข้ามเวลา"
    if perm_seed is not None:
        # null test: สลับ y **ใน train เท่านั้น** · y ของ eval ไม่ถูกแตะ
        rng = np.random.default_rng(perm_seed)
        y_tr = pd.Series(rng.permutation(y_tr.to_numpy()), index=y_tr.index)
    wrapped = TransformedTargetRegressor(
        regressor=make_model(name, override, ann_seeds=ann_seeds),
        transformer=StandardScaler(),
    )
    wrapped.fit(X_tr, y_tr)
    pred = wrapped.predict(X_ev)
    return (regression_metrics(y_ev, pred)["MAE_return"],
            prediction_shape(y_ev, pred)["StdRatio"], pred)


def eval_config(name, override, active_groups, data, fold_ids, ann_seeds, parallel):
    """
    ประเมินหนึ่ง config บนทุก (ticker, fold) ที่ระบุ -- ขนานระดับ evaluation

    คืน (mean_mae, maes, stdratios, keys) · joblib คืนผลตามลำดับที่ส่งเข้าไปเสมอ
    -> ค่าเฉลี่ย deterministic
    """
    jobs, keys = [], []
    for ticker, (X_all, y, folds, groups) in data.items():
        X_sub = select_groups(X_all, active_groups, groups)
        for fid in fold_ids:
            tr, ev = folds[fid - 1]
            jobs.append(delayed(eval_task)(name, override, X_sub, y, tr, ev, ann_seeds))
            keys.append((ticker, fid))
    out = parallel(jobs)
    maes = [o[0] for o in out]
    stds = [o[1] for o in out]
    return float(np.mean(maes)), maes, stds, keys


# ---------------------------------------------------------------
# Search space (OPTUNA_PLAN ข้อ 3)
# ---------------------------------------------------------------
def suggest_params(trial, name):
    """คืน override dict ของ hyperparameter -- ช่วงตามแผน ห้ามแก้หลังเห็นผล"""
    if name == "ANN (MLP)":
        hidden = trial.suggest_categorical("hidden_layer_sizes", ANN_HIDDEN_CHOICES)
        return {
            # log-uniform 0.1-50 : ผู้ชนะเดิม alpha 7.0 อยู่ที่ขอบ grid [0.3..7.0]
            "alpha": trial.suggest_float("alpha", 0.1, 50.0, log=True),
            # เก็บเป็น str ใน Optuna (categorical ไม่รับ tuple) แล้วแปลงกลับ
            "hidden_layer_sizes": tuple(int(v) for v in
                                        hidden.strip("()").split(", ")),
        }
    if name == "Random Forest":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "max_features": trial.suggest_categorical("max_features",
                                                      RF_MAX_FEATURES_CHOICES),
        }
    if name == "XGBoost":
        return {
            "max_depth": trial.suggest_int("max_depth", 1, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        }
    raise KeyError(name)


def suggest_groups(trial):
    """8 พารามิเตอร์ bool -- suggest ทุกตัวเสมอ เพื่อให้ param space คงที่ทุก trial"""
    flags = {g: trial.suggest_categorical(f"use_{g}", [False, True])
             for g in GROUP_NAMES}
    return [g for g in GROUP_NAMES if flags[g]]


# ---------------------------------------------------------------
# Objective
# ---------------------------------------------------------------
def make_objective(name, data, rows, sha, parallel):
    def objective(trial):
        t0 = time.time()
        override = suggest_params(trial, name)
        active = suggest_groups(trial)
        row = {"plan_sha256": sha, "model": name, "number": trial.number,
               "n_groups": len(active),
               "active_groups": "|".join(active),
               **{f"use_{g}": (g in active) for g in GROUP_NAMES},
               **{k: (str(v) if isinstance(v, tuple) else v)
                  for k, v in override.items()},
               "n_features": None, "ann_seeds": len(
                   TUNING_ANN_SEEDS if name == "ANN (MLP)" else [1])}

        if not active:
            # OPTUNA_PLAN ข้อ 2: ต้องเปิดอย่างน้อย 1 กลุ่ม -> penalty (ไม่ลบ trial)
            row.update({"objective": PENALTY, "min_StdRatio": np.nan,
                        "passed_guard": False, "no_groups": True,
                        "seconds": round(time.time() - t0, 2)})
            rows.append(row)
            print(f"[opt][{name}][trial {trial.number}] ไม่เปิดกลุ่มใดเลย -> penalty",
                  flush=True)
            return PENALTY

        seeds = TUNING_ANN_SEEDS if name == "ANN (MLP)" else (0,)
        mean_mae, maes, stds, keys = eval_config(
            name, override, active, data, SEARCH_FOLDS, seeds, parallel)
        min_std = float(np.nanmin(stds)) if len(stds) else np.nan
        passed = bool(np.all(np.asarray(stds, dtype=float) >= GUARD_MIN_STDRATIO))
        obj = mean_mae if passed else PENALTY

        first = list(data)[0]
        row["n_features"] = select_groups(data[first][0], active,
                                          data[first][3]).shape[1]
        for (ticker, fid), m, s in zip(keys, maes, stds):
            t = ticker.replace(".BK", "")
            row[f"MAE_{t}_f{fid}"] = m
            row[f"StdRatio_{t}_f{fid}"] = s
        row.update({"mean_MAE_return": mean_mae, "objective": obj,
                    "min_StdRatio": min_std, "passed_guard": passed,
                    "no_groups": False, "seconds": round(time.time() - t0, 2)})
        rows.append(row)
        print(f"[opt][{name}][trial {trial.number}/{N_TRIALS[name]}] "
              f"mean MAE={mean_mae:.7f} minStd={min_std:.4f} "
              f"guard={'ok' if passed else 'FAIL'} "
              f"n_feat={row['n_features']} groups={len(active)} "
              f"({row['seconds']:.0f}s)", flush=True)
        return obj
    return objective


# ---------------------------------------------------------------
# fold 3 confirmation + null test + ablation
# ---------------------------------------------------------------
def phase1c_override(name):
    """
    config Phase 1C = ค่าที่อยู่ใน config.py ตอนนี้ -> override = {} (make_model ใช้ค่า config)

    คืน dict ว่าง ไม่ใช่การคัดลอกค่า เพื่อให้ไม่มีค่าซ้ำสองที่
    """
    assert name in MODEL_NAMES
    return {}


def confirm_on_fold3(name, winner_override, winner_groups, data, parallel, sha):
    """
    ประเมินบน fold 3 (Optuna ไม่เคยเห็น): ผู้ชนะ (ANN 10 seeds) · Phase 1C · Naive

    *** fold 3 ไม่ใช่ pristine *** มันทับช่วงที่ tune.py ใช้เลือก Phase 1C มาก่อน
    (TUNING_PLAN ใช้ทั้ง 3 folds) -> เป็นแค่ "ชุดที่ไม่ถูกใช้ในการค้นรอบนี้"
    ความเปรียบจึงเอนเข้าข้าง Phase 1C บน fold 3
    """
    rows = []
    for ticker, (X_all, y, folds, groups) in data.items():
        tr, ev = folds[CONFIRM_FOLD - 1]
        y_ev = y.iloc[ev]
        naive = baseline_naive_return(y_ev)
        mae_naive = regression_metrics(y_ev, naive)["MAE_return"]

        specs = [("optuna_winner", winner_override, winner_groups),
                 ("phase1c", phase1c_override(name), PHASE1C_GROUPS)]
        jobs = [delayed(eval_task_pred)(name, ov,
                                        select_groups(X_all, gs, groups),
                                        y, tr, ev, ANN_SEEDS)
                for _, ov, gs in specs]
        out = parallel(jobs)
        preds = {}
        for (label, _, gs), (mae, std, pred) in zip(specs, out):
            preds[label] = pred
            rows.append({"plan_sha256": sha, "model": name, "fold": CONFIRM_FOLD,
                         "ticker": ticker, "config": label,
                         "n_groups": len(gs), "active_groups": "|".join(gs),
                         "MAE_return": mae, "StdRatio": std,
                         "MAE_naive": mae_naive, "relMAE": mae / mae_naive,
                         "ann_seeds": len(ANN_SEEDS)})
        # Naive เป็นแถวในตารางเดียวกัน (กฎของ task_b: baseline อยู่ในตารางเดียวกับโมเดล)
        rows.append({"plan_sha256": sha, "model": name, "fold": CONFIRM_FOLD,
                     "ticker": ticker, "config": "naive", "n_groups": 0,
                     "active_groups": "", "MAE_return": mae_naive,
                     "StdRatio": prediction_shape(y_ev, naive)["StdRatio"],
                     "MAE_naive": mae_naive, "relMAE": 1.0, "ann_seeds": 0})

        # DM: ผู้ชนะ vs Naive และ ผู้ชนะ vs Phase 1C
        for label, ref_pred, ref_name in (("optuna_winner", naive, "naive"),
                                          ("optuna_winner", preds["phase1c"],
                                           "phase1c"),
                                          ("phase1c", naive, "naive")):
            dm = diebold_mariano(y_ev - preds[label], y_ev - ref_pred, loss="abs")
            for r in rows:
                if (r["model"] == name and r["ticker"] == ticker
                        and r["config"] == label):
                    r[f"dm_vs_{ref_name}_stat_hln"] = dm["dm_stat_hln"]
                    r[f"dm_vs_{ref_name}_p_one_sided"] = \
                        dm["p_one_sided_model_better"]
        print(f"[opt][{name}][{ticker}][fold 3] winner relMAE="
              f"{rows[-3]['relMAE']:.4f} phase1c relMAE={rows[-2]['relMAE']:.4f}",
              flush=True)
    return rows


def ablation(name, winner_override, winner_groups, data, parallel, sha, winner_obj):
    """ปิดทีละกลุ่มของผู้ชนะ ประเมินบน fold 1-2 (สนามเดียวกับที่ค้น)"""
    rows = []
    seeds = TUNING_ANN_SEEDS if name == "ANN (MLP)" else (0,)
    for g in winner_groups:
        rest = [x for x in winner_groups if x != g]
        if not rest:
            print(f"[opt][{name}] ablation ข้าม {g}: ผู้ชนะเปิดกลุ่มเดียว")
            continue
        mae, maes, stds, _ = eval_config(name, winner_override, rest, data,
                                         SEARCH_FOLDS, seeds, parallel)
        rows.append({"plan_sha256": sha, "model": name, "removed_group": g,
                     "remaining_groups": "|".join(rest),
                     "mean_MAE_return": mae,
                     "winner_mean_MAE_return": winner_obj,
                     "delta": mae - winner_obj,
                     "pct_worse": (mae - winner_obj) / winner_obj,
                     "min_StdRatio": float(np.nanmin(stds)),
                     "passed_guard": bool(np.all(np.asarray(stds) >= GUARD_MIN_STDRATIO))})
        print(f"[opt][{name}][ablation] ปิด {g:12s} -> {mae:.7f} "
              f"({rows[-1]['pct_worse']:+.2%})", flush=True)
    return rows


def null_test(name, winner_override, winner_groups, data, parallel, sha):
    """
    รัน config ผู้ชนะด้วย y ที่สลับสุ่มใน train (3 permutations) บน fold 1-2 และ fold 3

    y ของ eval **ไม่ถูกสลับ** -> วัดว่า "ถ้าไม่มีสัญญาณเลย MAE จะอยู่ที่เท่าไหร่"
    """
    rows = []
    for perm in range(N_PERMUTATIONS):
        for fid in list(SEARCH_FOLDS) + [CONFIRM_FOLD]:
            jobs, keys = [], []
            for ticker, (X_all, y, folds, groups) in data.items():
                X_sub = select_groups(X_all, winner_groups, groups)
                tr, ev = folds[fid - 1]
                jobs.append(delayed(eval_task_pred)(
                    name, winner_override, X_sub, y, tr, ev, ANN_SEEDS,
                    perm_seed=perm))
                keys.append(ticker)
            out = parallel(jobs)
            for ticker, (mae, std, _) in zip(keys, out):
                rows.append({"plan_sha256": sha, "model": name,
                             "permutation_seed": perm, "fold": fid,
                             "ticker": ticker, "MAE_return": mae,
                             "StdRatio": std})
        print(f"[opt][{name}][null] permutation {perm} เสร็จ", flush=True)
    return rows


# ---------------------------------------------------------------
# รูป
# ---------------------------------------------------------------
def make_figures(trials, freq, fold3, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"ANN (MLP)": "#4C78A8", "Random Forest": "#59A14F",
              "XGBoost": "#E45756"}

    # 1) optimization history
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for ax, name in zip(axes, MODEL_NAMES):
        g = trials[(trials["model"] == name)].sort_values("number")
        ok = g[g["passed_guard"]]
        bad = g[~g["passed_guard"]]
        ax.scatter(ok["number"], ok["mean_MAE_return"], s=22,
                   color=colors[name], label="passed guard")
        if len(bad):
            # trial ที่ตก guard ได้ objective = PENALTY แต่พล็อตที่ค่า MAE ดิบ
            # เพื่อให้เห็นว่ามันไปอยู่ตรงไหน (ถ้าพล็อตที่ 10.0 จะบีบแกนจนอ่านอะไรไม่ได้)
            ax.scatter(bad["number"], bad["mean_MAE_return"], s=26, marker="x",
                       color="#999999",
                       label="guard FAIL (objective = penalty; plotted at raw MAE)")
        running = ok.sort_values("number")["mean_MAE_return"].cummin()
        ax.plot(ok.sort_values("number")["number"], running, color="black", lw=1.5,
                label="best so far")
        ax.set_title(f"{name} — {len(g)} trials", fontsize=11)
        ax.set_xlabel("trial number")
        ax.set_ylabel("mean MAE_return (folds 1-2)")
        ax.grid(alpha=0.3, lw=0.6)
        ax.legend(fontsize=8)
    fig.suptitle("task_b feedback item 3 — Optuna optimization history "
                 "(search folds 1-2 only, dev data, report only)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "optuna_history.png", dpi=150)
    plt.close(fig)

    # 2) feature group frequency in top-10
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    width = 0.26
    xs = np.arange(len(GROUP_NAMES))
    for i, name in enumerate(MODEL_NAMES):
        g = freq[freq["model"] == name].set_index("group").reindex(GROUP_NAMES)
        ax.bar(xs + (i - 1) * width, g["n_open_in_top10"], width,
               color=colors[name], label=name, edgecolor="white")
    ax.axhline(TOP_K / 2, color="black", ls="--", lw=1.3,
               label=f"random baseline = {TOP_K // 2}/{TOP_K}")
    ax.set_xticks(xs)
    ax.set_xticklabels(GROUP_NAMES, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(f"times group was ON among top-{TOP_K} trials")
    ax.set_ylim(0, TOP_K + 0.6)
    ax.set_title(f"task_b feedback item 3 — feature group frequency in top-{TOP_K} "
                 "trials per model", fontsize=12)
    ax.grid(axis="y", alpha=0.3, lw=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "optuna_group_frequency.png", dpi=150)
    plt.close(fig)

    # 3) fold 3: winner vs phase1c vs naive
    fig, axes = plt.subplots(1, len(TICKERS), figsize=(12.5, 5.0))
    labels = ["optuna_winner", "phase1c", "naive"]
    lcolor = {"optuna_winner": "#4C78A8", "phase1c": "#59A14F", "naive": "#555555"}
    for ax, ticker in zip(np.atleast_1d(axes), TICKERS):
        sub = fold3[fold3["ticker"] == ticker]
        xs = np.arange(len(MODEL_NAMES))
        for i, lab in enumerate(labels):
            vals = [float(sub[(sub["model"] == m) & (sub["config"] == lab)]
                          ["relMAE"].iloc[0]) for m in MODEL_NAMES]
            ax.bar(xs + (i - 1) * 0.27, vals, 0.27, color=lcolor[lab], label=lab,
                   edgecolor="white")
        ax.axhline(1.0, color="black", ls="--", lw=1.3)
        ax.set_xticks(xs)
        ax.set_xticklabels(MODEL_NAMES, fontsize=9)
        ax.set_ylabel("relMAE on fold 3 (= MAE / MAE_naive)")
        lo = min(0.99, fold3["relMAE"].min() - 0.005)
        ax.set_ylim(lo, max(1.01, float(fold3["relMAE"].max()) + 0.005))
        ax.set_title(f"{ticker} — fold 3 (not seen during this search)", fontsize=11)
        ax.grid(axis="y", alpha=0.3, lw=0.6)
        ax.legend(fontsize=9)
    fig.suptitle("task_b feedback item 3 — Optuna winner vs Phase 1C on the "
                 "confirmation fold (below 1.0 = beats Naive)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "optuna_fold3.png", dpi=150)
    plt.close(fig)
    print(f"[opt] บันทึกรูป 3 ไฟล์ -> {_rel(out_dir)}")


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def group_frequency(trials, importances, sha):
    rows = []
    for name in MODEL_NAMES:
        g = trials[(trials["model"] == name) & trials["passed_guard"]]
        top = g.nsmallest(min(TOP_K, len(g)), "objective")
        imp = importances.get(name, {})
        for grp in GROUP_NAMES:
            rows.append({
                "plan_sha256": sha, "model": name, "group": grp,
                "n_top_trials": len(top),
                "n_open_in_top10": int(top[f"use_{grp}"].sum()),
                "frac_open_in_top10": float(top[f"use_{grp}"].mean()),
                "frac_open_all_trials": float(g[f"use_{grp}"].mean()),
                "optuna_importance": imp.get(f"use_{grp}", np.nan),
            })
    return pd.DataFrame(rows)


def run(out_dir, trial_budget=None):
    sha = plan_sha256()
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = trial_budget or N_TRIALS
    print("=" * 78)
    print("  optuna_features.py (feedback ข้อ 3) -- ผลรายงานเท่านั้น")
    print(f"  plan_sha256 = {sha}")
    print(f"  optuna {optuna.__version__} · TPESampler(seed={SAMPLER_SEED}) "
          f"· ไม่ใช้ pruner · งบ {budget}")
    print(f"  Optuna เห็น fold {SEARCH_FOLDS} · fold {CONFIRM_FOLD} = confirmation")
    print(f"  ขนานระดับ evaluation: {n_parallel()} process (loky) · โมเดล n_jobs=1")
    print(f"  *** ห้ามนำผู้ชนะไปใส่ config.py / features.py (OPTUNA_PLAN ข้อ 7) ***")
    print("=" * 78)

    data = load_dev_data_grouped()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    trial_rows, fold3_rows, abl_rows, null_rows, sum_rows = [], [], [], [], []
    importances = {}
    t_all = time.time()

    with Parallel(n_jobs=n_parallel(), backend="loky") as parallel:
        for name in MODEL_NAMES:
            db = out_dir / f"optuna_{name.split()[0].lower()}.db"
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED),
                storage=f"sqlite:///{db.as_posix()}",
                study_name=f"taskb_{name}_{sha[:12]}",
                load_if_exists=True,
            )
            done = len(study.trials)
            todo = max(0, budget[name] - done)
            print(f"\n[opt] {name}: มีอยู่แล้ว {done} trials · จะรันอีก {todo} "
                  f"(db = {db.name})")
            if done:
                # รายงานต้องมี "ทุก trial ทุกโมเดล" (OPTUNA_PLAN ข้อ 7.1) แต่คะแนนราย
                # evaluation / StdRatio / passed_guard ของ trial เก่าไม่ได้ถูกเก็บใน
                # sqlite -- มันอยู่ใน rows_here ของการรันครั้งนั้นเท่านั้น
                # ถ้าปล่อยให้ resume ต่อ optuna_trials.csv จะขาด trial เก่าไปเงียบ ๆ
                # (และถ้า todo = 0 จะไปตกที่ HARD STOP ของ guard ซึ่งให้เหตุผลผิด)
                raise SystemExit(
                    f"\n!! HARD STOP: {db.name} มี {done} trials ค้างอยู่\n"
                    "   การรันต่อจะทำให้ optuna_trials.csv ขาด trial เก่า\n"
                    "   (คะแนนราย evaluation ไม่ได้ถูกเก็บใน sqlite)\n"
                    f"   -> ลบไฟล์ .db แล้วรันใหม่ทั้งหมด:  del {db}\n"
                    "   (OPTUNA_PLAN ข้อ 6 ห้าม resume ข้ามการแก้โค้ดอยู่แล้ว)")
            rows_here = []
            if todo:
                study.optimize(make_objective(name, data, rows_here, sha, parallel),
                               n_trials=todo)
            trial_rows += rows_here

            g = pd.DataFrame(rows_here)
            ok = g[g["passed_guard"]] if len(g) else g
            if not len(ok):
                raise SystemExit(
                    f"\n!! HARD STOP (OPTUNA_PLAN ข้อ 5): ทุก trial ของ {name} "
                    f"ไม่ผ่าน guard StdRatio >= {GUARD_MIN_STDRATIO}\n"
                    "   ห้ามลดเกณฑ์หลังเห็นผล -- รายงานว่าโมเดลนี้ยุบเป็นค่าคงที่ทุกค่า")
            best = ok.loc[ok["objective"].idxmin()]
            win_groups = best["active_groups"].split("|")
            win_override = study.trials[int(best["number"])].params
            win_override = suggest_params(
                optuna.trial.FixedTrial(win_override), name)
            print(f"[opt] {name} ผู้ชนะ = trial {int(best['number'])} "
                  f"objective {best['objective']:.7f} · groups {win_groups} "
                  f"· params {win_override}")

            try:
                importances[name] = optuna.importance.get_param_importances(study)
            except Exception as e:                    # noqa: BLE001
                print(f"[opt] {name}: คำนวณ importance ไม่ได้ ({e})")
                importances[name] = {}

            # Phase 1C บน fold 1-2 (สนามเดียวกับที่ค้น) เพื่อเทียบ objective ตรง ๆ
            seeds = TUNING_ANN_SEEDS if name == "ANN (MLP)" else (0,)
            p1c_obj, _, p1c_stds, _ = eval_config(
                name, phase1c_override(name), PHASE1C_GROUPS, data,
                SEARCH_FOLDS, seeds, parallel)

            fold3_rows += confirm_on_fold3(name, win_override, win_groups, data,
                                           parallel, sha)
            abl_rows += ablation(name, win_override, win_groups, data, parallel,
                                 sha, float(best["objective"]))
            null_rows += null_test(name, win_override, win_groups, data, parallel,
                                   sha)

            sum_rows.append({
                "plan_sha256": sha, "model": name,
                "n_trials": len(g), "n_failed_guard": int((~g["passed_guard"]).sum()),
                "winner_trial": int(best["number"]),
                "winner_params": str(win_override),
                "winner_groups": best["active_groups"],
                "winner_n_groups": int(best["n_groups"]),
                "winner_n_features": int(best["n_features"]),
                "winner_obj_folds12": float(best["objective"]),
                "winner_min_StdRatio": float(best["min_StdRatio"]),
                "phase1c_obj_folds12": p1c_obj,
                "phase1c_min_StdRatio": float(np.nanmin(p1c_stds)),
                "delta_folds12": float(best["objective"]) - p1c_obj,
                "pct_folds12": (float(best["objective"]) - p1c_obj) / p1c_obj,
                "config_tag_phase1c": CONFIG_TAG,
            })

    elapsed = time.time() - t_all
    trials = pd.DataFrame(trial_rows)
    fold3 = pd.DataFrame(fold3_rows)
    summary = pd.DataFrame(sum_rows)
    freq = group_frequency(trials, importances, sha)

    # เติมผล fold 3 ลง summary (ต่อโมเดล เฉลี่ยสองหุ้น + ต่อหุ้น)
    for i, r in summary.iterrows():
        f = fold3[fold3["model"] == r["model"]]
        w = f[f["config"] == "optuna_winner"]
        p = f[f["config"] == "phase1c"]
        summary.loc[i, "winner_MAE_fold3"] = w["MAE_return"].mean()
        summary.loc[i, "phase1c_MAE_fold3"] = p["MAE_return"].mean()
        summary.loc[i, "winner_relMAE_fold3"] = w["relMAE"].mean()
        summary.loc[i, "phase1c_relMAE_fold3"] = p["relMAE"].mean()
        summary.loc[i, "delta_fold3"] = (w["MAE_return"].mean()
                                         - p["MAE_return"].mean())
        summary.loc[i, "pct_fold3"] = (summary.loc[i, "delta_fold3"]
                                       / p["MAE_return"].mean())
        summary.loc[i, "winner_n_sig_better_than_naive_fold3"] = int(
            (w["dm_vs_naive_p_one_sided"] < ALPHA).sum())
        summary.loc[i, "winner_n_sig_better_than_phase1c_fold3"] = int(
            (w["dm_vs_phase1c_p_one_sided"] < ALPHA).sum())
        summary.loc[i, "phase1c_n_sig_better_than_naive_fold3"] = int(
            (p["dm_vs_naive_p_one_sided"] < ALPHA).sum())
        nl = pd.DataFrame(null_rows)
        nl = nl[(nl["model"] == r["model"]) & (nl["fold"] == CONFIRM_FOLD)]
        summary.loc[i, "null_MAE_fold3_mean"] = nl["MAE_return"].mean()
        summary.loc[i, "null_MAE_fold3_min"] = nl["MAE_return"].min()
        summary.loc[i, "null_MAE_fold3_max"] = nl["MAE_return"].max()

    for df, fn in ((trials, "optuna_trials.csv"), (summary, "optuna_summary.csv"),
                   (freq, "optuna_group_frequency.csv"),
                   (fold3, "optuna_fold3.csv"),
                   (pd.DataFrame(abl_rows), "optuna_ablation.csv"),
                   (pd.DataFrame(null_rows), "optuna_null.csv")):
        df.to_csv(out_dir / fn, index=False, encoding="utf-8-sig")
        print(f"[opt] บันทึก {_rel(out_dir / fn)} ({len(df)} แถว)")
    make_figures(trials, freq, fold3, out_dir)

    pd.set_option("display.width", 260)
    print("\n" + summary[["model", "n_trials", "n_failed_guard", "winner_trial",
                          "winner_n_groups", "winner_obj_folds12",
                          "phase1c_obj_folds12", "pct_folds12",
                          "winner_MAE_fold3", "phase1c_MAE_fold3", "pct_fold3",
                          "winner_n_sig_better_than_naive_fold3"]].to_string(index=False))
    print(f"\n[opt] เวลารวม {elapsed:.0f}s")
    return trials, summary, freq, fold3, elapsed


def main():
    p = argparse.ArgumentParser(
        description="Optuna feature + hyperparameter search -- dev data เท่านั้น")
    p.add_argument("--run", action="store_true", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="3 trials/โมเดล ลงโฟลเดอร์ชั่วคราว แล้วลบทิ้ง (จับเวลา)")
    args = p.parse_args()      # ห้ามมี flag ใดที่เกี่ยวกับ test

    if args.smoke:
        tmp = Path(tempfile.mkdtemp(prefix="taskb_optuna_smoke_"))
        try:
            budget = {k: 3 for k in N_TRIALS}
            _, _, _, _, elapsed = run(tmp, trial_budget=budget)
            n_smoke = sum(budget.values())
            per_trial = elapsed / n_smoke
            est = per_trial * sum(N_TRIALS.values())
            print(f"\n[opt] SMOKE: {n_smoke} trials ใน {elapsed:.0f}s "
                  f"({per_trial:.1f}s/trial รวม overhead ของ fold3/ablation/null)")
            print(f"[opt] ประมาณเวลาเต็ม ({sum(N_TRIALS.values())} trials) "
                  f"= {est:.0f}s = {est / 60:.1f} นาที")
            if est > 3600:
                print("[opt] !! ประมาณเกิน 60 นาที -> เงื่อนไขหยุดข้อ 9.3 "
                      "ต้องถามผู้ใช้ก่อนรันเต็ม")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"[opt] ลบโฟลเดอร์ชั่วคราว {tmp}")
        return
    run(FEEDBACK_DIR)


if __name__ == "__main__":
    main()
