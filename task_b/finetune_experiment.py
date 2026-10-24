"""
finetune_experiment.py — pretrain ด้วยข้อมูลเก่า แล้ว fine-tune ด้วยข้อมูลใหม่ (feedback ข้อ 7)
================================================================================================

*** ไฟล์นี้ทำอะไร ***
ทดลองการเทรน 2 ขั้น (old -> recent) เทียบกับการเทรนรวมครั้งเดียว (pooled) บน walk-forward
3 folds x 2 หุ้น = 6 evaluation ตามแผนที่ประกาศไว้ก่อนรันใน
`results/feedback/FINETUNE_PLAN.md` (SHA-256 ของแผนถูกบันทึกลงทุกแถวของผลลัพธ์)

*** ทำไมต้องมี / ตอบ feedback ข้อไหน ***
feedback ข้อ 7 (14 ก.ย. 2026) ขอ "pretrain ด้วยข้อมูลเก่า แล้ว fine-tune ด้วยข้อมูลใหม่"
ก่อนหน้านี้ repo มีแค่ recency weighting (`weighting.py`) ซึ่ง docstring ของมันเขียนไว้ว่า
เลือกวิธีถ่วงน้ำหนักแทน "เพราะแบบ 2 ขั้นเพิ่ม hyperparameter และเสี่ยงให้ลืมข้อมูลเก่า"
= เหตุผลเชิงทฤษฎีที่ยังไม่เคยทดลองจริง ไฟล์นี้ทดลองจริง

*** ขอบเขต ***
- ข้อมูลตัดที่ SPLIT_BY_DATE["val_end"] ก่อนสร้าง feature (ผ่าน `tune.load_dev_data`
  ซึ่ง assert ให้เอง) -> ไม่มีทางแตะ test · ไม่แตะ fixed val ของ main.py
- **ผลรายงานเท่านั้น ห้ามนำผู้ชนะไปใส่ config.py** (FINETUNE_PLAN ข้อ 4)

    python finetune_experiment.py --run
    python finetune_experiment.py --run --smoke     # 1 fold 1 หุ้น เพื่อจับเวลา

ผลลัพธ์: results/feedback/finetune_results.csv · finetune_summary.csv · finetune.png
"""

import argparse
import copy
import hashlib
import time

import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import (
    OUTPUT_DIR, BASE_DIR, ANN_PARAMS, ANN_SEEDS, RF_PARAMS, XGB_PARAMS,
)
from baselines import baseline_naive_return
from evaluate import regression_metrics, prediction_shape
from weighting import recency_weights
from stats_tests import diebold_mariano
from tune import load_dev_data, make_model, MODEL_NAMES, N_SPLITS

FEEDBACK_DIR = OUTPUT_DIR / "feedback"
PLAN_PATH = FEEDBACK_DIR / "FINETUNE_PLAN.md"
RESULTS_CSV = FEEDBACK_DIR / "finetune_results.csv"
SUMMARY_CSV = FEEDBACK_DIR / "finetune_summary.csv"
FIG_PATH = FEEDBACK_DIR / "finetune.png"
TUNING_SCORES = OUTPUT_DIR / "tuning_scores.csv"

# ---------------------------------------------------------------
# ค่าที่ประกาศไว้ใน FINETUNE_PLAN.md -- ห้ามแก้หลังเห็นผล
# ---------------------------------------------------------------
R_GRID = [250, 500]              # ขนาดของ "recent" (แถวท้ายของ train)
ANN_EPOCH_GRID = [50, 200]       # จำนวน partial_fit ตอน fine-tune
ANN_LR_MULT_GRID = [1.0, 0.1]    # ตัวคูณ learning_rate_init ตอน fine-tune
XGB_ROUND_GRID = [50, 150]       # จำนวน boosting round ที่เทรนต่อบน recent
RF_TREE_GRID = [100, 400]        # จำนวนต้นไม้ใหม่ที่เทรนบน recent (analogue ไม่ใช่ fine-tune)
GUARD_MIN_STDRATIO = 0.05
ALPHA = 0.05
PIPELINE_RTOL = 1e-6             # เกณฑ์ว่าเส้นทาง manual เทียบเท่า Pipeline


def plan_sha256():
    """SHA-256 ของไฟล์แผน -- บันทึกลงทุกแถว เพื่อผูกผลกับแผนฉบับที่ใช้จริง"""
    if not PLAN_PATH.exists():
        raise SystemExit(f"ไม่พบ {PLAN_PATH} -- ต้องเขียนแผนก่อนรัน (FINETUNE_PLAN ข้อ 0)")
    return hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()


# ---------------------------------------------------------------
# half-life จาก stage 1 -- อ่านไฟล์ ไม่จูนใหม่ (FINETUNE_PLAN ข้อ 2.2)
# ---------------------------------------------------------------
def stage1_half_lives():
    """
    คืน (winner, best_finite) ต่อโมเดล จาก results/tuning_scores.csv

    winner      = half-life ที่ชนะ stage 1 (อาจเป็น None)
    best_finite = half-life ที่ "ไม่ใช่ None" ที่คะแนนดีที่สุด -- arm ข้อมูลประกอบ
                  (ประกาศไว้ใน FINETUNE_PLAN ข้อ 2.2 ก่อนรัน)
    """
    if not TUNING_SCORES.exists():
        raise SystemExit(f"ไม่พบ {TUNING_SCORES} -- ต้องมีผล tune.py stage 1 ก่อน")
    s = pd.read_csv(TUNING_SCORES)
    s = s[s["stage"].astype(str) == "1"]
    out = {}
    for name in MODEL_NAMES:
        sub = s[s["model"] == name]
        win = sub[sub["winner"].astype(str).isin(["True", "1", "1.0"])]
        assert len(win) == 1, f"stage 1 ของ {name} ต้องมีผู้ชนะหนึ่งค่า (ได้ {len(win)})"
        w = win["half_life"].iloc[0]
        winner = None if pd.isna(w) else int(w)
        finite = sub[sub["half_life"].notna()].sort_values("primary_score")
        out[name] = (winner, int(finite["half_life"].iloc[0]))
    return out


# ---------------------------------------------------------------
# เส้นทาง manual scaling -- จำเป็นเพราะ partial_fit ต้องเข้าถึงข้อมูลที่ scale แล้ว
# ---------------------------------------------------------------
class ScaledFrame:
    """
    scaler ของ X และ y fit จาก train ทั้ง fold (old + recent) ครั้งเดียว
    แล้วใช้ค่าเดียวกันทุกขั้นทุก arm ของ fold นั้น (FINETUNE_PLAN ข้อ 1)

    ถ้า fit scaler แยกต่อขั้น ความต่างระหว่าง arm จะมาจาก scaler ไม่ใช่จากกลยุทธ์
    scaler เห็นแต่ข้อมูล train ของ fold นั้น ไม่เคยเห็นแถว eval -> ไม่ leak

    *** X scaler ใช้เฉพาะโมเดลที่ pipeline เดิมใช้ ***
    `models._scaled` (ANN) = Pipeline(impute, **scale**, model)
    `models._unscaled` (RF / XGB) = Pipeline(impute, model) -- **ไม่มี StandardScaler**
    ถ้าที่นี่ scale X ให้ tree ด้วย จะไม่เทียบเท่า pipeline เดิม: tree ไม่สนใจ scale
    ในเชิงทฤษฎี (การแบ่งเป็น monotone transform) แต่ค่า threshold ที่คำนวณได้ต่างกัน
    ในระดับ floating point -> ต้นไม้คนละต้น -> MAE ต่างกันราว 1e-5 ซึ่ง guard ข้อ 6
    ของแผนจับได้จริง (RF fold 3 ของ KBANK รอบแรก)
    y scaler ใช้กับทุกโมเดล เพราะ `main.py` / `tune.py` ห่อทุกตัวด้วย
    TransformedTargetRegressor(StandardScaler())

    imputer ไม่ถูกคัดลอกมา เพราะ `splits.prepare_xy` ตัดแถวที่มี NaN ทิ้งหมดแล้ว
    (ในทางปฏิบัติ imputer เป็น no-op) -- มี assert ยืนยันด้านล่าง
    """

    # โมเดลที่ pipeline เดิม scale X ให้ (ต้องตรงกับ models.get_regressors())
    SCALES_X = {"ANN (MLP)"}

    def __init__(self, X_train, y_train):
        assert np.isfinite(np.asarray(X_train, dtype=float)).all(), "X_train มี NaN/inf"
        self.xs = StandardScaler().fit(np.asarray(X_train, dtype=float))
        self.ys = StandardScaler().fit(np.asarray(y_train, dtype=float).reshape(-1, 1))

    def X(self, name, X):
        """คืน X ที่พร้อมเข้าโมเดล -- scale เฉพาะโมเดลที่ pipeline เดิม scale"""
        arr = np.asarray(X, dtype=float)
        return self.xs.transform(arr) if name in self.SCALES_X else arr

    def y(self, y):
        return self.ys.transform(np.asarray(y, dtype=float).reshape(-1, 1)).ravel()

    def inv_y(self, y):
        return self.ys.inverse_transform(np.asarray(y, dtype=float).reshape(-1, 1)).ravel()


def _ann_estimator(seed):
    return MLPRegressor(**{**ANN_PARAMS, "random_state": seed})


def _set_finetune_lr(mlp, lr_mult):
    """
    ตั้ง learning rate ของขั้น fine-tune

    Adam ของ sklearn คำนวณ `learning_rate` ใหม่จาก `learning_rate_init` ทุก iteration
    (`learning_rate = learning_rate_init * sqrt(1-beta2^t)/(1-beta1^t)`) จึงต้องแก้
    `learning_rate_init` ของทั้ง estimator และของ `_optimizer` ไม่ใช่แก้ `learning_rate`
    (แก้ `learning_rate` ตรง ๆ จะถูกเขียนทับใน iteration ถัดไปเงียบ ๆ)
    """
    lr = ANN_PARAMS["learning_rate_init"] * lr_mult
    mlp.learning_rate_init = lr
    if hasattr(mlp, "_optimizer") and hasattr(mlp._optimizer, "learning_rate_init"):
        mlp._optimizer.learning_rate_init = lr
    return lr


def _fit_tree(name, X, y, override=None):
    """
    สร้าง + fit โมเดล tree-based (X ที่ส่งเข้ามาไม่ถูก scale -- ดู ScaledFrame.SCALES_X)

    ใช้เฉพาะ arm ที่ไม่ต้องถ่วงน้ำหนัก (old_only / recent_only)
    ส่วน `predict_pooled` เขียนแยกเพราะต้องส่ง `sample_weight` ให้ได้ (arm
    recency_weighted) ซึ่งเป็น argument ของ `fit` ไม่ใช่ของ constructor
    """
    params = {"Random Forest": RF_PARAMS, "XGBoost": XGB_PARAMS}[name]
    cls = {"Random Forest": RandomForestRegressor, "XGBoost": XGBRegressor}[name]
    est = cls(**{**params, **(override or {})})
    est.fit(X, y)
    return est


# ---------------------------------------------------------------
# แต่ละ arm: คืน prediction บน eval (หน่วย return จริง ไม่ใช่ scaled)
# ---------------------------------------------------------------
def predict_pooled(name, sc, Xtr, ytr, Xev, half_life=None):
    """เทรนครั้งเดียวบน train ทั้ง fold (+ ถ่วงน้ำหนัก recency ถ้าใส่ half_life)"""
    Xs, ys, Xe = sc.X(name, Xtr), sc.y(ytr), sc.X(name, Xev)
    w = recency_weights(len(Xs), half_life)
    if w is not None:
        assert len(w) == len(Xs) and np.isclose(w.mean(), 1.0) and w[-1] == w.max()
    if name == "ANN (MLP)":
        preds = []
        for seed in ANN_SEEDS:
            m = _ann_estimator(seed)
            m.fit(Xs, ys) if w is None else m.fit(Xs, ys, sample_weight=w)
            preds.append(m.predict(Xe))
        return sc.inv_y(np.mean(preds, axis=0))
    params = {"Random Forest": RF_PARAMS, "XGBoost": XGB_PARAMS}[name]
    cls = {"Random Forest": RandomForestRegressor, "XGBoost": XGBRegressor}[name]
    est = cls(**params)
    est.fit(Xs, ys) if w is None else est.fit(Xs, ys, sample_weight=w)
    return sc.inv_y(est.predict(Xe))


def predict_subset(name, sc, Xsub, ysub, Xev):
    """เทรนบน subset เดียว (old_only / recent_only)"""
    Xs, ys, Xe = sc.X(name, Xsub), sc.y(ysub), sc.X(name, Xev)
    if name == "ANN (MLP)":
        preds = []
        for seed in ANN_SEEDS:
            m = _ann_estimator(seed)
            m.fit(Xs, ys)
            preds.append(m.predict(Xe))
        return sc.inv_y(np.mean(preds, axis=0))
    return sc.inv_y(_fit_tree(name, Xs, ys).predict(Xe))


def pretrain_ann(sc, Xold, yold):
    """
    fit ANN บน old ครั้งเดียวต่อ seed -> คืน list ของ MLP ที่ fit แล้ว

    แยกออกมาเพื่อให้ config ที่มี R เดียวกันแต่ E / lr ต่างกัน ใช้ผล pretrain ร่วมกันได้
    (deepcopy ก่อน partial_fit) -- ผลเหมือนการ pretrain ใหม่ทุกครั้งเป๊ะ เพราะ
    partial_fit ไม่ shuffle และ deepcopy เก็บ state ของ optimizer/_random_state ครบ
    """
    Xs, ys = sc.X("ANN (MLP)", Xold), sc.y(yold)
    return [_ann_estimator(seed).fit(Xs, ys) for seed in ANN_SEEDS]


def predict_ann_finetune(sc, pretrained, Xrec, yrec, Xev, epochs, lr_mult):
    """ANN: partial_fit ต่อบน recent E ครั้ง · 10 seeds เฉลี่ยเหมือนนิยามโมเดลเดิม"""
    Xr, yr, Xe = sc.X("ANN (MLP)", Xrec), sc.y(yrec), sc.X("ANN (MLP)", Xev)
    preds, lr_used = [], None
    for m0 in pretrained:
        m = copy.deepcopy(m0)
        lr_used = _set_finetune_lr(m, lr_mult)
        for _ in range(epochs):
            m.partial_fit(Xr, yr)
        preds.append(m.predict(Xe))
    return sc.inv_y(np.mean(preds, axis=0)), lr_used


def predict_xgb_finetune(sc, Xold, yold, Xrec, yrec, Xev, rounds):
    """XGBoost: train บน old (300 rounds) แล้วเทรนต่อบน recent อีก K rounds ด้วย xgb_model="""
    n = "XGBoost"
    Xo, yo = sc.X(n, Xold), sc.y(yold)
    Xr, yr = sc.X(n, Xrec), sc.y(yrec)
    Xe = sc.X(n, Xev)
    base = XGBRegressor(**XGB_PARAMS).fit(Xo, yo)
    cont = XGBRegressor(**{**XGB_PARAMS, "n_estimators": rounds})
    cont.fit(Xr, yr, xgb_model=base.get_booster())
    return sc.inv_y(cont.predict(Xe))


def predict_rf_warmstart(sc, Xold, yold, Xrec, yrec, Xev, extra_trees):
    """
    Random Forest analogue -- *** ไม่ใช่ fine-tune จริง ***

    RF ปรับน้ำหนักต้นไม้ที่สร้างแล้วไม่ได้ · ที่ทำคือ warm_start=True แล้วเพิ่มต้นไม้ใหม่
    K ต้นที่เทรนบน recent ต่อจาก 400 ต้นเดิมที่เทรนบน old
    การทำนาย = ค่าเฉลี่ยของต้นไม้ทั้ง 400 + K ต้น -> เป็นการผสมแบบถ่วงน้ำหนักคงที่
    (400/(400+K) : K/(400+K)) ระหว่างโมเดล old กับโมเดล recent
    ไม่ใช่การปรับพารามิเตอร์เดิมด้วยข้อมูลใหม่ ซึ่งเป็นความหมายของคำว่า fine-tune
    """
    n = "Random Forest"
    Xo, yo = sc.X(n, Xold), sc.y(yold)
    Xr, yr = sc.X(n, Xrec), sc.y(yrec)
    Xe = sc.X(n, Xev)
    rf = RandomForestRegressor(**{**RF_PARAMS, "warm_start": True})
    rf.fit(Xo, yo)
    n0 = rf.n_estimators
    rf.set_params(n_estimators=n0 + extra_trees)
    rf.fit(Xr, yr)                       # ต้นไม้ใหม่ K ต้นเทรนบน recent เท่านั้น
    assert len(rf.estimators_) == n0 + extra_trees
    return sc.inv_y(rf.predict(Xe)), n0


# ---------------------------------------------------------------
# สร้างรายการ config ต่อโมเดล
# ---------------------------------------------------------------
def arm_configs(name, half_lives):
    """คืน list ของ dict config ตาม FINETUNE_PLAN ข้อ 2 -- ลำดับคงที่"""
    winner_hl, finite_hl = half_lives[name]
    cfgs = [{"arm": "pooled", "label": "pooled"}]
    for R in R_GRID:
        cfgs.append({"arm": "old_only", "R": R, "label": f"old_only R={R}"})
    for R in R_GRID:
        cfgs.append({"arm": "recent_only", "R": R, "label": f"recent_only R={R}"})
    if name == "ANN (MLP)":
        for R in R_GRID:
            for E in ANN_EPOCH_GRID:
                for lm in ANN_LR_MULT_GRID:
                    cfgs.append({"arm": "pretrain_finetune", "R": R, "epochs": E,
                                 "lr_mult": lm,
                                 "label": f"pretrain_finetune R={R} E={E} lr x{lm:g}"})
    elif name == "XGBoost":
        for R in R_GRID:
            for K in XGB_ROUND_GRID:
                cfgs.append({"arm": "pretrain_finetune", "R": R, "rounds": K,
                             "label": f"pretrain_finetune R={R} K={K} rounds"})
    else:
        for R in R_GRID:
            for K in RF_TREE_GRID:
                cfgs.append({"arm": "pretrain_finetune", "R": R, "extra_trees": K,
                             "label": f"warm_start_analogue R={R} K={K} trees"})
    cfgs.append({"arm": "recency_weighted", "half_life": winner_hl,
                 "label": f"recency_weighted hl={winner_hl}"})
    cfgs.append({"arm": "recency_weighted_best_finite", "half_life": finite_hl,
                 "label": f"recency_weighted_best_finite hl={finite_hl}"})
    return cfgs


# ---------------------------------------------------------------
# รันหนึ่ง evaluation ครบทุก config ของหนึ่งโมเดล
# ---------------------------------------------------------------
def run_eval(name, ticker, fold, X, y, tr, ev, cfgs, sha):
    X_tr, y_tr = X.iloc[tr], y.iloc[tr]
    X_ev, y_ev = X.iloc[ev], y.iloc[ev]
    assert X_tr.index[-1] < X_ev.index[0], "fold ข้ามเวลา"
    sc = ScaledFrame(X_tr, y_tr)

    naive = baseline_naive_return(y_ev)
    mae_naive = regression_metrics(y_ev, naive)["MAE_return"]

    # pretrain ครั้งเดียวต่อ R (ANN) -- ใช้ซ้ำกับทุก (E, lr) ที่ R เดียวกัน
    pre_cache, split_cache = {}, {}

    def splits_for(R):
        if R not in split_cache:
            assert R < len(X_tr), f"R ({R}) ต้องน้อยกว่า train ของ fold ({len(X_tr)})"
            Xo, yo = X_tr.iloc[:-R], y_tr.iloc[:-R]
            Xr, yr = X_tr.iloc[-R:], y_tr.iloc[-R:]
            assert len(Xo) + len(Xr) == len(X_tr)
            assert Xo.index[-1] < Xr.index[0], "old กับ recent ซ้อนทับกัน"
            split_cache[R] = (Xo, yo, Xr, yr)
        return split_cache[R]

    rows = []
    pooled_pred = None
    for cfg in cfgs:
        t0 = time.time()
        extra = {}
        arm = cfg["arm"]
        if arm == "pooled":
            pred = predict_pooled(name, sc, X_tr, y_tr, X_ev)
            pooled_pred = pred
        elif arm in ("recency_weighted", "recency_weighted_best_finite"):
            pred = predict_pooled(name, sc, X_tr, y_tr, X_ev,
                                  half_life=cfg["half_life"])
        elif arm == "old_only":
            Xo, yo, _, _ = splits_for(cfg["R"])
            pred = predict_subset(name, sc, Xo, yo, X_ev)
        elif arm == "recent_only":
            _, _, Xr, yr = splits_for(cfg["R"])
            pred = predict_subset(name, sc, Xr, yr, X_ev)
        elif arm == "pretrain_finetune":
            R = cfg["R"]
            Xo, yo, Xr, yr = splits_for(R)
            if name == "ANN (MLP)":
                if R not in pre_cache:
                    pre_cache[R] = pretrain_ann(sc, Xo, yo)
                pred, lr = predict_ann_finetune(sc, pre_cache[R], Xr, yr, X_ev,
                                                cfg["epochs"], cfg["lr_mult"])
                extra["finetune_lr"] = lr
            elif name == "XGBoost":
                pred = predict_xgb_finetune(sc, Xo, yo, Xr, yr, X_ev, cfg["rounds"])
            else:
                pred, n0 = predict_rf_warmstart(sc, Xo, yo, Xr, yr, X_ev,
                                                cfg["extra_trees"])
                extra["base_trees"] = n0
        else:
            raise KeyError(arm)

        m = regression_metrics(y_ev, pred)
        shape = prediction_shape(y_ev, pred)
        dm_naive = diebold_mariano(y_ev - pred, y_ev - naive, loss="abs")
        row = {
            "plan_sha256": sha, "model": name, "arm": arm, "label": cfg["label"],
            "ticker": ticker, "fold": fold,
            "R": cfg.get("R"), "epochs": cfg.get("epochs"),
            "lr_mult": cfg.get("lr_mult"), "rounds": cfg.get("rounds"),
            "extra_trees": cfg.get("extra_trees"), "half_life": cfg.get("half_life"),
            "train_rows": len(tr), "eval_rows": len(ev),
            "eval_start": X_ev.index[0].date().isoformat(),
            "eval_end": X_ev.index[-1].date().isoformat(),
            "MAE_return": m["MAE_return"], "RMSE_return": m["RMSE_return"],
            "R2_return": m["R2_return"],
            "MAE_naive": mae_naive, "relMAE": m["MAE_return"] / mae_naive,
            "StdRatio": shape["StdRatio"], "Bias": shape["Bias"], "Rho": shape["Rho"],
            "dm_vs_naive_stat_hln": dm_naive["dm_stat_hln"],
            "dm_vs_naive_p_one_sided": dm_naive["p_one_sided_model_better"],
            "seconds": round(time.time() - t0, 2), **extra,
        }
        # DM vs pooled -- pooled เป็น config แรกเสมอ จึงมีค่าแล้วทุกครั้ง
        assert pooled_pred is not None, "pooled ต้องถูกรันเป็น config แรก"
        if arm == "pooled":
            row["dm_vs_pooled_stat_hln"] = 0.0
            row["dm_vs_pooled_p_one_sided"] = 1.0
        else:
            dmp = diebold_mariano(y_ev - pred, y_ev - pooled_pred, loss="abs")
            row["dm_vs_pooled_stat_hln"] = dmp["dm_stat_hln"]
            row["dm_vs_pooled_p_one_sided"] = dmp["p_one_sided_model_better"]
        rows.append(row)
        print(f"[ft][{name}][{ticker}][fold {fold}] {cfg['label']:46s} "
              f"MAE={m['MAE_return']:.7f} relMAE={row['relMAE']:.4f} "
              f"StdRatio={shape['StdRatio']:.4f} ({row['seconds']:.0f}s)", flush=True)

    # --- ตรวจว่าเส้นทาง manual เทียบเท่า Pipeline (FINETUNE_PLAN ข้อ 6) ---
    wrapped = TransformedTargetRegressor(regressor=make_model(name, None,
                                                              ann_seeds=ANN_SEEDS),
                                         transformer=StandardScaler())
    wrapped.fit(X_tr, y_tr)
    mae_pipe = regression_metrics(y_ev, wrapped.predict(X_ev))["MAE_return"]
    mae_manual = rows[0]["MAE_return"]
    ok = bool(np.isclose(mae_manual, mae_pipe, rtol=PIPELINE_RTOL, atol=0))
    for r in rows:
        r["pooled_matches_pipeline"] = ok
        r["pooled_mae_pipeline"] = mae_pipe
    print(f"[ft][{name}][{ticker}][fold {fold}] pooled manual {mae_manual:.10f} vs "
          f"pipeline {mae_pipe:.10f} -> {'ตรง' if ok else '!! ไม่ตรง'}", flush=True)
    if not ok:
        raise SystemExit(
            f"\n!! HARD STOP (FINETUNE_PLAN ข้อ 6): เส้นทาง manual scaling ของ {name} "
            f"ไม่เทียบเท่า Pipeline ({mae_manual} vs {mae_pipe}, rtol {PIPELINE_RTOL})\n"
            "   ผลทั้งตารางเชื่อไม่ได้ -- ต้องแก้ ScaledFrame ก่อน"
        )
    return rows


# ---------------------------------------------------------------
# สรุป
# ---------------------------------------------------------------
def summarise(ev, n_evals):
    rows = []
    for (name, label), g in ev.groupby(["model", "label"], sort=False):
        arm = g["arm"].iloc[0]
        passed = bool((g["StdRatio"] >= GUARD_MIN_STDRATIO).all())
        p_pool = g["dm_vs_pooled_p_one_sided"]
        p_naive = g["dm_vs_naive_p_one_sided"]
        rows.append({
            "plan_sha256": g["plan_sha256"].iloc[0],
            "model": name, "arm": arm, "label": label,
            "n_evals": len(g),
            "mean_MAE_return": g["MAE_return"].mean(),
            "mean_relMAE": g["relMAE"].mean(),
            "min_StdRatio": g["StdRatio"].min(),
            "passed_guard": passed,
            "mean_dm_vs_pooled_stat_hln": g["dm_vs_pooled_stat_hln"].mean(),
            "n_evals_better_than_pooled": int((p_pool < 0.5).sum()),
            "n_sig_better_than_pooled": int((p_pool < ALPHA).sum()),
            "mean_dm_vs_naive_stat_hln": g["dm_vs_naive_stat_hln"].mean(),
            "n_sig_better_than_naive": int((p_naive < ALPHA).sum()),
            "n_evals_relMAE_lt_1": int((g["relMAE"] < 1).sum()),
            "total_seconds": g["seconds"].sum(),
        })
    sm = pd.DataFrame(rows)
    assert (sm["n_evals"] == n_evals).all(), "จำนวน evaluation ต่อ config ไม่ครบ"
    # ส่วนต่างจาก pooled ของโมเดลเดียวกัน (รายงาน -- ไม่ใช่เกณฑ์ตัดสินเดี่ยว)
    base = sm[sm["arm"] == "pooled"].set_index("model")["mean_MAE_return"]
    sm["pooled_mean_MAE_return"] = sm["model"].map(base)
    sm["delta_vs_pooled"] = sm["mean_MAE_return"] - sm["pooled_mean_MAE_return"]
    sm["pct_vs_pooled"] = sm["delta_vs_pooled"] / sm["pooled_mean_MAE_return"]
    # กฎตัดสิน FINETUNE_PLAN ข้อ 4 -- ทั้งสามข้อต้องผ่าน
    sm["helps_by_plan_rule"] = (
        (sm["arm"] == "pretrain_finetune")
        & (sm["delta_vs_pooled"] < 0)
        & (sm["n_sig_better_than_pooled"] >= 4)
        & sm["passed_guard"]
    )
    return sm


def make_figure(sm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = ["pooled", "old_only", "recent_only", "pretrain_finetune",
            "recency_weighted", "recency_weighted_best_finite"]
    colors = dict(zip(arms, ["#000000", "#4C78A8", "#E45756", "#59A14F",
                             "#B07AA1", "#9C755F"]))
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(16.5, 7.0), sharey=False)
    for ax, name in zip(np.atleast_1d(axes), MODEL_NAMES):
        g = sm[sm["model"] == name].copy()
        # เรียงจาก MAE ต่ำ -> สูง เพื่อให้อ่าน "ใครดีกว่า pooled" ได้ทันที
        g = g.sort_values("mean_MAE_return")
        ypos = np.arange(len(g))[::-1]
        ax.barh(ypos, g["mean_MAE_return"],
                color=[colors[a] for a in g["arm"]],
                edgecolor="white", height=0.72)
        pooled = float(g.loc[g["arm"] == "pooled", "mean_MAE_return"].iloc[0])
        ax.axvline(pooled, color="black", ls="--", lw=1.4,
                   label=f"pooled = {pooled:.6f}")
        ax.set_yticks(ypos)
        ax.set_yticklabels(g["label"], fontsize=7.5)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("mean MAE_return (6 evaluations) — lower is better", fontsize=9)
        lo = min(g["mean_MAE_return"].min(), pooled)
        hi = g["mean_MAE_return"].max()
        pad = (hi - lo) * 0.12 or hi * 0.01
        ax.set_xlim(lo - pad, hi + pad)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", alpha=0.3, lw=0.6)
        ax.legend(fontsize=8, loc="lower right")
        for ypos_i, (v, ok) in zip(ypos, zip(g["mean_MAE_return"], g["passed_guard"])):
            if not ok:
                ax.text(v, ypos_i, "  guard FAIL", va="center", fontsize=7,
                        color="#E45756")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[a]) for a in arms]
    fig.legend(handles, arms, loc="upper center", ncol=len(arms), fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("task_b feedback item 7 — pretrain + fine-tune vs pooled "
                 "(walk-forward 3 folds x 2 stocks, dev data only, report only)",
                 fontsize=11.5, y=0.955)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"[ft] บันทึก {FIG_PATH.relative_to(BASE_DIR)}")


def main():
    p = argparse.ArgumentParser(
        description="pretrain + fine-tune experiment -- dev data เท่านั้น ไม่มีทางเปิด test")
    p.add_argument("--run", action="store_true", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="1 หุ้น 1 fold เพื่อจับเวลา (ไม่เขียนทับไฟล์ผลจริง)")
    args = p.parse_args()       # ห้ามมี flag ใดที่เกี่ยวกับ test

    sha = plan_sha256()
    half_lives = stage1_half_lives()
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  finetune_experiment.py (feedback ข้อ 7) -- ผลรายงานเท่านั้น")
    print(f"  plan_sha256 = {sha}")
    print(f"  half-life stage 1 (winner, best_finite) = {half_lives}")
    print("  *** ห้ามนำผู้ชนะไปใส่ config.py (FINETUNE_PLAN ข้อ 4) ***")
    print("=" * 78)

    data = load_dev_data()
    if args.smoke:
        first = list(data)[0]
        X, y, folds = data[first]
        data = {first: (X, y, folds[:1])}

    t_all = time.time()
    rows = []
    for name in MODEL_NAMES:
        cfgs = arm_configs(name, half_lives)
        print(f"\n[ft] {name}: {len(cfgs)} configs x "
              f"{sum(len(f) for _, _, f in data.values())} evaluation")
        for ticker, (X, y, folds) in data.items():
            for k, (tr, ev) in enumerate(folds, start=1):
                rows += run_eval(name, ticker, k, X, y, tr, ev, cfgs, sha)
    elapsed = time.time() - t_all

    ev = pd.DataFrame(rows)
    n_evals = len(data) * len(list(data.values())[0][2])
    if args.smoke:
        print(f"\n[ft] SMOKE เสร็จใน {elapsed:.0f}s "
              f"({len(ev)} แถว) -- ไม่เขียนไฟล์ผลจริง")
        print(f"[ft] ประมาณเวลาเต็ม = {elapsed * 2 * N_SPLITS / n_evals:.0f}s")
        return
    ev.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[ft] บันทึก {RESULTS_CSV.relative_to(BASE_DIR)} ({len(ev)} แถว)")

    sm = summarise(ev, n_evals)
    sm.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"[ft] บันทึก {SUMMARY_CSV.relative_to(BASE_DIR)}")
    make_figure(sm)
    pd.set_option("display.width", 250)
    print(sm[["model", "label", "mean_MAE_return", "mean_relMAE", "min_StdRatio",
              "passed_guard", "pct_vs_pooled", "n_sig_better_than_pooled",
              "n_sig_better_than_naive", "helps_by_plan_rule"]].to_string(index=False))
    print(f"\n[ft] เวลารวม {elapsed:.0f}s")
    print(f"[ft] สรุปตามกฎ FINETUNE_PLAN ข้อ 4: "
          f"{'มี config ที่ผ่าน' if sm['helps_by_plan_rule'].any() else 'ไม่มี config ใดผ่าน'}")


if __name__ == "__main__":
    main()
