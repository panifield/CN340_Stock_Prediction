"""
stats_tests.py — การทดสอบนัยสำคัญและตัวเลขฝั่งทิศทาง (feedback ข้อ 2)
=====================================================================

*** ไฟล์นี้ทำอะไร ***
1. `diebold_mariano()` — Diebold-Mariano test (R8) พร้อม Harvey-Leybourne-Newbold
   small-sample correction (R9) เพื่อตอบว่า "โมเดลต่างจาก Naive อย่างมีนัยสำคัญไหม"
2. `direction_metrics()` — สัดส่วนวันราคานิ่ง · hit rate ของทิศทางเฉพาะวันที่ราคาขยับ ·
   information coefficient (Pearson / Spearman)
3. `mae_decomposition()` — แยก MAE ของโมเดลและ Naive เป็น "วันที่ y = 0" กับ "วันที่ y != 0"
   เพื่อให้เห็นว่า Naive ได้เปรียบจากวันราคานิ่งเท่าไหร่
4. โหมด `--walkforward` — รัน config Phase 1C บน walk-forward 6 evaluation
   (2 หุ้น x 3 folds เดียวกับ `tune.py`) แล้วเซฟ `results/feedback/walkforward_stats.csv`

*** ทำไมต้องมี / ตอบ feedback ข้อไหน ***
feedback ข้อ 2 (14 ก.ย. 2026) ถามว่า "ทำไม Naive ชนะ" ก่อนหน้านี้คำอธิบายกลไกมีแค่ในงาน
16:00 (`INTRADAY_FINDINGS.md`) ไม่มีของโมเดล next_day เลย และ **ทั้ง task_b ไม่มี p-value
แม้แต่ค่าเดียว** -> พูดได้แค่ว่า "ต่างกัน 0.000016" แต่ไม่รู้ว่าต่างแบบมีนัยสำคัญหรือเป็น noise

*** ทำไมเป็นโมดูลใหม่ ไม่ใส่ใน evaluate.regression_metrics ***
`evaluate.prediction_shape` มีกฎเขียนกำกับไว้ว่า
    "ห้ามเพิ่ม p-value / confidence interval / significance test ใด ๆ ลงในฟังก์ชันนี้"
และ `evaluate.py` หัวไฟล์ล็อกไว้ว่า metric มี 5 ค่า เพื่อให้ตารางหลักคงที่
การทดสอบนัยสำคัญจึงต้องอยู่ "คนละตาราง คนละไฟล์" เหมือนที่ตารางต้นทุน (`trading_costs.py`)
แยกออกไป -> ตารางหลักที่ใช้เลือกโมเดลไม่เปลี่ยน และ p-value ไม่มีทางไหลเข้าไปเป็นเกณฑ์เลือก

*** p-value เหล่านี้ใช้ทำอะไรได้ / ไม่ได้ ***
- **ได้**: รายงานว่าส่วนต่าง MAE ระหว่างโมเดลกับ Naive แยกจาก 0 ได้ไหม
- **ไม่ได้**: ใช้เลือกโมเดล / เลือก config / เลือก feature (การเลือกใช้ mean MAE_return เท่านั้น
  ตาม `TUNING_PLAN.md`) · และผลบน fixed val เป็น in-sample ของการจูน (val ถูกใช้จูนมาแล้ว)

    python stats_tests.py --walkforward
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats

PRIMARY_LOSSES = ("abs", "sq")


# ---------------------------------------------------------------
# 1) Diebold-Mariano + Harvey-Leybourne-Newbold
# ---------------------------------------------------------------
def diebold_mariano(e_model, e_naive, h=1, loss="abs"):
    """
    Diebold-Mariano test (1995) + HLN small-sample correction (1997)

    e_model / e_naive = **prediction error** (y_true - y_pred) ของแต่ละวัน ยาวเท่ากัน
    h    = forecast horizon (งานนี้ทำนายล่วงหน้า 1 วัน -> h = 1)
    loss = "abs" (MAE, primary ของงาน B) หรือ "sq" (MSE)

    นิยาม:
        d_t  = L(e_model,t) - L(e_naive,t)       ค่าลบ = โมเดล loss ต่ำกว่า Naive = โมเดลดีกว่า
        dbar = mean(d)
        V    = (gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k) / n      (long-run variance ของ dbar)
        DM   = dbar / sqrt(V)

    HLN correction (R9): คูณ DM ด้วย
        sqrt( (n + 1 - 2h + h(h-1)/n) / n )
    แล้วเทียบกับ **t-distribution df = n - 1** (ไม่ใช่ normal) -- HLN แสดงว่า DM ดั้งเดิม
    ปฏิเสธ H0 บ่อยเกินจริงเมื่อ n เล็ก ซึ่งเป็นกรณีของเรา (n = 50-280 ต่อ evaluation)

    *** autocovariance คำนวณแบบ biased (หารด้วย n) *** ตามสูตรดั้งเดิมของ DM
    สำหรับ h = 1 จะได้ V = gamma_0 / n ตรง ๆ (ไม่มีพจน์ autocovariance)

    คืน dict:
        n, h, loss
        mean_loss_model / mean_loss_naive / mean_loss_diff  (= dbar)
        dm_stat        DM ดั้งเดิม (ยังไม่ correct)
        dm_stat_hln    DM หลัง HLN correction (ตัวที่ใช้รายงาน)
        df
        p_two_sided    H0: loss เท่ากัน · H1: ไม่เท่า
        p_one_sided_model_better   H1: โมเดล loss ต่ำกว่า Naive (dbar < 0)
        model_better   dbar < 0 ไหม (ยังไม่ดูนัยสำคัญ)

    กรณีขอบ: ถ้า d เป็นค่าคงที่ทุกวัน (เช่น โมเดล = Naive เป๊ะ -> d = 0 ทุกตัว)
    variance = 0 -> DM ไม่นิยาม · คืน dm = 0.0 และ p = 1.0 ซึ่งเป็นการตีความที่ตรงที่สุด
    ("ไม่มีหลักฐานว่าต่างกัน") ไม่ใช่ nan ที่จะทำให้แถวในรายงานหายไปเงียบ ๆ
    """
    if loss not in PRIMARY_LOSSES:
        raise ValueError(f"loss ต้องเป็น {PRIMARY_LOSSES} (ได้ {loss!r})")
    if h < 1:
        raise ValueError(f"h ต้อง >= 1 (ได้ {h})")

    e_model = np.asarray(e_model, dtype=float)
    e_naive = np.asarray(e_naive, dtype=float)
    if e_model.shape != e_naive.shape:
        raise ValueError(f"e_model {e_model.shape} กับ e_naive {e_naive.shape} ยาวไม่เท่ากัน")
    n = len(e_model)
    if n < 3:
        raise ValueError(f"ต้องมีอย่างน้อย 3 วัน (ได้ {n})")
    if h > n:
        raise ValueError(f"h ({h}) ต้องไม่เกินจำนวนวัน ({n})")

    f = np.abs if loss == "abs" else np.square
    l_model, l_naive = f(e_model), f(e_naive)
    d = l_model - l_naive
    dbar = float(d.mean())

    dc = d - dbar
    gamma0 = float((dc * dc).mean())                  # biased (หารด้วย n) ตามสูตร DM
    acov = 0.0
    for k in range(1, h):
        acov += float((dc[k:] * dc[:-k]).mean())
    v = (gamma0 + 2.0 * acov) / n

    # HLN factor (R9 สมการ (9)) · h = 1 -> sqrt((n-1)/n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)

    if v <= 0:
        dm = 0.0
        dm_hln = 0.0
        p_two = 1.0
        p_one = 1.0
    else:
        dm = dbar / np.sqrt(v)
        dm_hln = dm * hln
        df = n - 1
        p_two = float(2.0 * stats.t.sf(abs(dm_hln), df))
        # H1: โมเดลดีกว่า = dbar < 0 = DM ติดลบ -> ใช้ cdf ฝั่งซ้าย
        p_one = float(stats.t.cdf(dm_hln, df))

    return {
        "n": n, "h": h, "loss": loss,
        "mean_loss_model": float(l_model.mean()),
        "mean_loss_naive": float(l_naive.mean()),
        "mean_loss_diff": dbar,
        "dm_stat": float(dm),
        "dm_stat_hln": float(dm_hln),
        "hln_factor": float(hln),
        "df": n - 1,
        "p_two_sided": float(p_two),
        "p_one_sided_model_better": float(p_one),
        "model_better": bool(dbar < 0),
    }


# ---------------------------------------------------------------
# 2) ตัวเลขฝั่งทิศทาง
# ---------------------------------------------------------------
def direction_metrics(y_true, y_pred):
    """
    diagnostic ฝั่งทิศทาง -- ห้ามใช้เลือกโมเดล (เหตุผลเดียวกับ prediction_shape)

    frac_zero_move          สัดส่วนวันที่ y = 0 เป๊ะ (ราคาปิดไม่ขยับ)
                            ยิ่งสูง Naive (ทาย 0) ยิ่งได้เปรียบใน MAE
    frac_pred_zero          สัดส่วนวันที่โมเดลทำนาย 0 เป๊ะ (โมเดล ML แทบไม่เคยเป็น 0)
    dir_hit_rate_moved_days ทายทิศถูกกี่ % **เฉพาะวันที่ y != 0**
                            ตัดวันนิ่งออกเพราะวันที่ y = 0 ไม่มี "ทิศที่ถูก" ให้ทาย
                            (ถ้านับรวม นโยบายการให้คะแนนวันนิ่งจะกลายเป็นตัวกำหนดคำตอบ)
                            นับ y_pred = 0 เป๊ะเป็น "ไม่ถูก" (ไม่ได้บอกทิศ)
    n_moved_days            จำนวนวันที่ใช้คิด hit rate
    ic_pearson / ic_spearman  information coefficient = corr(y_pred, y_true) ทั้งชุด
                            (Spearman ทนค่าผิดปกติกว่า ซึ่งสำคัญกับ return ที่หางหนา)

    ไม่มี p-value ในฟังก์ชันนี้ -- นัยสำคัญใช้ diebold_mariano() เท่านั้น
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} กับ y_pred {y_pred.shape} ยาวไม่เท่ากัน")

    moved = y_true != 0.0
    n_moved = int(moved.sum())
    hit = (np.sign(y_pred[moved]) == np.sign(y_true[moved])).mean() if n_moved else np.nan

    def _corr(fn):
        # ถ้าฝั่งใดฝั่งหนึ่งเป็นค่าคงที่ correlation ไม่นิยาม -> nan (เช่น Naive ที่ทาย 0 ทุกวัน)
        if np.ptp(y_pred) == 0 or np.ptp(y_true) == 0:
            return np.nan
        return float(fn(y_pred, y_true)[0])

    return {
        "frac_zero_move": float(np.mean(y_true == 0.0)),
        "frac_pred_zero": float(np.mean(y_pred == 0.0)),
        "dir_hit_rate_moved_days": float(hit) if n_moved else np.nan,
        "n_moved_days": n_moved,
        "ic_pearson": _corr(stats.pearsonr),
        "ic_spearman": _corr(stats.spearmanr),
    }


def mae_decomposition(y_true, y_pred, y_naive=None):
    """
    แยก MAE เป็นวันที่ y = 0 กับวันที่ y != 0 -> เห็นว่า Naive ได้เปรียบจากวันนิ่งเท่าไหร่

    y_naive = None -> ใช้ Naive ตามนิยาม (return = 0 ทุกวัน) ซึ่งเป็นตัวเดียวกับ
    `baselines.baseline_naive_return` · ส่งเข้ามาเองได้เพื่อตรวจกับ baseline ตัวจริง

    ตัวเลขที่ตอบคำถาม "ทำไม Naive ชนะ" ตรงที่สุดคือ
        contrib_zero_days = (mae_model_zero_days - mae_naive_zero_days) * frac_zero_move
    = Naive ได้ MAE ต่ำกว่าโมเดลเท่าไหร่ **เพราะวันนิ่งเพียงอย่างเดียว**
    (บนวันนิ่ง Naive มี error = 0 เป๊ะ จึงเป็นเพดานล่างที่โมเดลเอาชนะไม่ได้ ได้แค่เสมอ)
    ส่วน contrib_moved_days คือฝั่งที่โมเดลมีโอกาสชนะจริง
    ผลรวมสองพจน์ = mae_model - mae_naive เป๊ะ (assert ไว้ด้านล่าง)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_naive = np.zeros_like(y_true) if y_naive is None else np.asarray(y_naive, dtype=float)

    zero = y_true == 0.0
    moved = ~zero
    err_m, err_n = np.abs(y_true - y_pred), np.abs(y_true - y_naive)
    nan = float("nan")

    out = {
        "n": len(y_true),
        "n_zero_days": int(zero.sum()),
        "n_moved_days": int(moved.sum()),
        "frac_zero_move": float(zero.mean()),
        "mae_model": float(err_m.mean()),
        "mae_naive": float(err_n.mean()),
        "mae_model_zero_days": float(err_m[zero].mean()) if zero.any() else nan,
        "mae_naive_zero_days": float(err_n[zero].mean()) if zero.any() else nan,
        "mae_model_moved_days": float(err_m[moved].mean()) if moved.any() else nan,
        "mae_naive_moved_days": float(err_n[moved].mean()) if moved.any() else nan,
    }
    # ส่วนต่าง MAE รวม แยกตามที่มา (ถ่วงด้วยสัดส่วนวัน) -> รวมกันได้ค่าเดิมเป๊ะ
    out["contrib_zero_days"] = float(
        (err_m[zero].sum() - err_n[zero].sum()) / len(y_true)) if zero.any() else 0.0
    out["contrib_moved_days"] = float(
        (err_m[moved].sum() - err_n[moved].sum()) / len(y_true)) if moved.any() else 0.0
    out["mae_diff"] = out["mae_model"] - out["mae_naive"]
    assert np.isclose(out["contrib_zero_days"] + out["contrib_moved_days"],
                      out["mae_diff"], atol=1e-15), "การแยกส่วน MAE ไม่สมดุล"
    out["relMAE"] = out["mae_model"] / out["mae_naive"] if out["mae_naive"] else nan
    return out


# ---------------------------------------------------------------
# 3) รวมทุกอย่างของ "หนึ่งโมเดล บนหนึ่ง evaluation"
# ---------------------------------------------------------------
def full_report(y_true, y_pred, y_naive=None, h=1):
    """
    รวม mae_decomposition + direction_metrics + DM (ทั้ง abs และ sq) เป็นแถวเดียว

    prefix ของ DM: dm_abs_* / dm_sq_* เพื่อให้แยกชัดว่า loss ไหน
    (primary ของงาน B คือ MAE -> ใช้ dm_abs_* เป็นตัวหลักในรายงาน)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_naive = np.zeros_like(y_true) if y_naive is None else np.asarray(y_naive, dtype=float)

    row = {**mae_decomposition(y_true, y_pred, y_naive),
           **direction_metrics(y_true, y_pred)}
    e_model, e_naive = y_true - y_pred, y_true - y_naive
    for loss in PRIMARY_LOSSES:
        dm = diebold_mariano(e_model, e_naive, h=h, loss=loss)
        for k in ("mean_loss_diff", "dm_stat", "dm_stat_hln", "p_two_sided",
                  "p_one_sided_model_better", "model_better", "df"):
            row[f"dm_{loss}_{k}"] = dm[k]
    return row


# ---------------------------------------------------------------
# 4) โหมด --walkforward: config Phase 1C บน 6 evaluation
# ---------------------------------------------------------------
ALPHA = 0.05            # ประกาศไว้ตรงนี้ ไม่เลือกหลังเห็นผล


def run_walkforward():
    """
    รัน config Phase 1C (ค่าที่อยู่ใน config.py ตอนนี้) บน walk-forward 6 evaluation
    = 2 หุ้น x 3 folds เดียวกับ `tune.py` (เรียก tune.load_dev_data / tune.make_model
    ตัวเดียวกัน -- ไม่เขียน logic เตรียมข้อมูลใหม่) · ANN ใช้ 10 seeds (ANN_SEEDS) ตามนิยามโมเดล

    ข้อมูลถูกตัดที่ SPLIT_BY_DATE["val_end"] ก่อนสร้าง feature ภายใน tune.load_dev_data()
    และมี assert ของมันเอง -> ไม่มีทางเห็น test

    เซฟ results/feedback/walkforward_stats.csv (หนึ่งแถวต่อ model x ticker x fold)
    + walkforward_summary.csv (หนึ่งแถวต่อ model: นับ evaluation ที่ชนะ Naive อย่างมีนัยสำคัญ
    ที่ alpha = 0.05 one-sided และหลัง Bonferroni ข้าม 6 evaluation)
    """
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler

    from config import OUTPUT_DIR, ANN_SEEDS, RECENCY_HALF_LIFE, CONFIG_TAG
    from weighting import recency_weights
    from tune import load_dev_data, make_model, MODEL_NAMES, N_SPLITS

    out_dir = OUTPUT_DIR / "feedback"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  stats_tests --walkforward · config = {CONFIG_TAG} · ANN {len(ANN_SEEDS)} seeds")
    print("  ข้อมูลถึง val_end เท่านั้น (tune.load_dev_data assert ให้)")
    print("=" * 78)
    data = load_dev_data()

    rows = []
    for name in MODEL_NAMES:
        half_life = RECENCY_HALF_LIFE[name]       # KeyError = ชื่อไม่ตรง -> ต้องพัง
        for ticker, (X, y, folds) in data.items():
            for k, (tr, ev) in enumerate(folds, start=1):
                t0 = time.time()
                X_tr, y_tr = X.iloc[tr], y.iloc[tr]
                X_ev, y_ev = X.iloc[ev], y.iloc[ev]
                assert X_tr.index[-1] < X_ev.index[0], "fold ข้ามเวลา"
                w = recency_weights(len(X_tr), half_life)
                fit_params = {} if w is None else {"model__sample_weight": w}
                wrapped = TransformedTargetRegressor(
                    # override=None -> ใช้ค่าใน config.py ตรง ๆ (Phase 1C)
                    regressor=make_model(name, None, ann_seeds=ANN_SEEDS),
                    transformer=StandardScaler(),
                )
                wrapped.fit(X_tr, y_tr, **fit_params)
                pred = wrapped.predict(X_ev)

                rep = full_report(y_ev, pred)
                rows.append({"model": name, "ticker": ticker, "fold": k,
                             "config_tag": CONFIG_TAG, "half_life": half_life,
                             "train_rows": len(tr), "eval_rows": len(ev),
                             "eval_start": X_ev.index[0].date().isoformat(),
                             "eval_end": X_ev.index[-1].date().isoformat(),
                             "fit_seconds": round(time.time() - t0, 2), **rep})
                print(f"[wf][{name}][{ticker}][fold {k}] "
                      f"relMAE={rep['relMAE']:.4f} "
                      f"DM(abs)={rep['dm_abs_dm_stat_hln']:+.3f} "
                      f"p1={rep['dm_abs_p_one_sided_model_better']:.3f} "
                      f"IC={rep['ic_pearson']:+.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    ev_df = pd.DataFrame(rows)
    n_evals = 2 * N_SPLITS
    assert len(ev_df) == len(MODEL_NAMES) * n_evals, "จำนวน evaluation ไม่ครบ"
    path = out_dir / "walkforward_stats.csv"
    ev_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[wf] บันทึก {path.relative_to(OUTPUT_DIR.parent)}")

    # --- สรุปต่อโมเดล ---
    bonf = ALPHA / n_evals
    srows = []
    for name in MODEL_NAMES:
        s = ev_df[ev_df["model"] == name]
        p1 = s["dm_abs_p_one_sided_model_better"]
        srows.append({
            "model": name,
            "n_evals": len(s),
            "mean_MAE_return": s["mae_model"].mean(),
            "mean_relMAE": s["relMAE"].mean(),
            "n_evals_lower_mae_than_naive": int((s["relMAE"] < 1).sum()),
            "alpha": ALPHA,
            "n_sig_better_uncorrected": int((p1 < ALPHA).sum()),
            "bonferroni_alpha": bonf,
            "n_sig_better_bonferroni": int((p1 < bonf).sum()),
            # นัยสำคัญฝั่งที่ Naive ดีกว่า (two-sided p เล็ก และ relMAE > 1)
            "n_sig_worse_uncorrected": int(((s["dm_abs_p_two_sided"] < ALPHA)
                                            & (s["relMAE"] > 1)).sum()),
            "min_p_one_sided": float(p1.min()),
            "mean_ic_pearson": s["ic_pearson"].mean(),
            "mean_ic_spearman": s["ic_spearman"].mean(),
            "mean_dir_hit_rate_moved_days": s["dir_hit_rate_moved_days"].mean(),
            "mean_frac_zero_move": s["frac_zero_move"].mean(),
            "mean_contrib_zero_days": s["contrib_zero_days"].mean(),
            "mean_contrib_moved_days": s["contrib_moved_days"].mean(),
        })
    sm = pd.DataFrame(srows)
    spath = out_dir / "walkforward_summary.csv"
    sm.to_csv(spath, index=False, encoding="utf-8-sig")
    print(f"[wf] บันทึก {spath.relative_to(OUTPUT_DIR.parent)}")
    print(sm.to_string(index=False))
    return ev_df, sm


def main():
    p = argparse.ArgumentParser(
        description="DM test / direction metrics ของงาน B -- ไม่มีทางเปิด test")
    p.add_argument("--walkforward", action="store_true",
                   help="รัน config Phase 1C บน walk-forward 6 evaluation แล้วเซฟ CSV")
    args = p.parse_args()     # ห้ามมี flag ใดที่เกี่ยวกับ test
    if not args.walkforward:
        p.error("ต้องระบุ --walkforward (โมดูลนี้ถูกเรียกใช้จาก main.py ในกรณีอื่น)")
    run_walkforward()


if __name__ == "__main__":
    main()
