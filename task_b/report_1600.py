"""
report_1600.py — สร้างรายงาน Phase 1D จากผลที่ tune_1600.py --run เขียนไว้แล้ว
===============================================================================

    python report_1600.py

*** อ่าน CSV อย่างเดียว -- ไม่ fit โมเดล ไม่เลือก config ใหม่ ไม่เปลี่ยนตัวเลขใด ๆ ***
input : results/phase1d/phase1d_{evals,scores,baselines,econ,folds}.csv  (จาก tune_1600.py --run)
output: results/phase1d/tuning_1600_evals.csv   (1 แถว/evaluation + วันที่ของ fold)
        results/phase1d/tuning_1600_scores.csv  (1 แถว/config + rank)
        results/phase1d/summary_1600.md         (development / walk-forward — not a test)

ทุกตัวเลขเป็น development / walk-forward บนข้อมูลที่เคย probe แล้ว — ไม่ใช่ test
ห้ามเทียบในตารางเดียวกับ daily model
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
D = BASE_DIR / "results" / "phase1d"
LABEL = "development / walk-forward — not a test"
FAMILIES = ["ANN (MLP)", "Random Forest", "XGBoost"]


def _load(name):
    return pd.read_csv(D / f"phase1d_{name}.csv")


def build_evals(ev, folds):
    out = ev.merge(folds, on=["ticker", "fold"], how="left", validate="many_to_one")
    assert out["train_start"].notna().all()
    out["params"] = out["config"]
    out["MAE_naive"] = out["naive_MAE_return"]
    # tune_1600.py บันทึกเป็น bool ว่าทุก prediction finite หรือไม่
    # True -> 0 แน่นอน · False -> นับจำนวนไม่ได้จากผลที่มี (>= 1) -> NaN
    out["n_nonfinite_pred"] = np.where(out["pred_finite"], 0, np.nan)
    cols = ["stage", "model", "config_id", "params", "n_seeds", "ticker", "fold",
            "train_start", "train_end", "eval_start", "eval_end", "train_rows", "eval_rows",
            "MAE_return", "MAE_naive", "relMAE", "StdRatio", "Bias", "Rho", "n_nonfinite_pred",
            "RMSE_return", "R2_return", "MAE_baht", "RMSE_baht", "R2_OOS", "frac_y_zero"]
    return out[cols]


def build_scores(sc):
    sc = sc.copy()
    sc["rank"] = np.nan
    for fam in FAMILIES:
        ok = (sc["model"] == fam) & sc["valid"] & sc["passed_guard"]
        sc.loc[ok, "rank"] = sc.loc[ok, "primary_score"].rank(method="min")
    sc = sc.rename(columns={"config": "params"})
    return sc[["model", "config_id", "params", "primary_score", "mean_MAE_return",
               "min_StdRatio", "n_evals", "valid", "passed_guard", "rank", "winner"]]


def _fmt(x, pct=False):
    if pd.isna(x):
        return "NaN"
    return f"{x:.2%}" if pct else f"{x:.6f}"


def md_table(df, cols, pct_cols=()):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(_fmt(v, c in pct_cols))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def build_summary(ev, sc, base, econ, folds):
    L = [f"# Phase 1D (Mode A 16:00) — {LABEL}", "",
         "> สร้างโดย `python report_1600.py` จากผลของ `python tune_1600.py --run` (รันครั้งเดียว)",
         "> ข้อมูลทั้งหมดเป็น development data ที่เคย probe แล้ว · **ไม่มี historical test** · "
         "prospective evaluation ยังไม่เปิด",
         "> **ห้ามเทียบในตารางเดียวกับ daily model** (คนละโจทย์ คนละแหล่งข้อมูล คนละ horizon)",
         "",
         "**ข้อความบังคับ:** ~43–49% ของวันราคาไม่ขยับเลย (close_bar16 = close_bar15) ⇒ MAE เอื้อ "
         "prediction ที่ใกล้ 0 (shrinkage) · StdRatio ต่ำของผู้ชนะเป็นสิ่งที่คาดได้ **ไม่ใช่หลักฐานของ skill**",
         "", "## 1. Fold boundaries และ omitted evaluation dates", ""]
    L += md_table(folds, list(folds.columns))
    L += ["", "Omitted evaluation dates: KBANK.BK 2026-09-25 · ADVANC.BK ไม่มี "
          "(ไม่ใช่ holdout/test · ใช้เทรน live ได้)", ""]

    L += [f"## 2. Selection ({LABEL})", "",
          "primary_score = mean relMAE (MAE_model / MAE_Naive) ของ 6 evaluation · ANN จัดอันดับด้วย 3 seeds · "
          "guard = StdRatio finite และ >= 0.05 ทุก evaluation · tie = ต่างจากค่าต่ำสุด < 1e-3", ""]
    sel = []
    for fam in FAMILIES:
        s = sc[sc["model"] == fam].sort_values("primary_score")
        w = s[s["winner"]]
        if w.empty:
            L.append(f"- **{fam}: ไม่มี valid config** — ไม่มี winner ไม่มี live model")
            continue
        w = w.iloc[0]
        rest = s[(s["config_id"] != w["config_id"]) & s["valid"] & s["passed_guard"]]
        c = rest.iloc[0]
        L.append(f"- **{fam}** ผู้ชนะ `{w['params']}` · primary_score {w['primary_score']:.6f} · "
                 f"mean MAE_return {w['mean_MAE_return']:.6f} · min StdRatio {w['min_StdRatio']:.4f}")
        L.append(f"  - ค่าที่แพ้ใกล้สุด `{c['params']}` = {c['primary_score']:.6f} "
                 f"(ต่าง {c['primary_score'] - w['primary_score']:+.6f} — เกิน tie 1e-3 จึงไม่ใช่เสมอ)"
                 if c["primary_score"] - w["primary_score"] >= 1e-3 else
                 f"  - ค่าที่แพ้ใกล้สุด `{c['params']}` = {c['primary_score']:.6f} "
                 f"(ต่าง {c['primary_score'] - w['primary_score']:+.6f} — เสมอ ตัดสินด้วย tie rule)")
        sel.append((fam, int(w["config_id"])))
    n_valid = int(sc["valid"].sum())
    n_guard = int(sc["passed_guard"].sum())
    n_below1 = int((sc["primary_score"] < 1).sum())
    L += ["", f"- searched configs: {len(sc)} · valid (prediction finite ทุกแถว): {n_valid} · "
          f"ผ่าน guard: {n_guard} · config ที่ primary_score < 1 (ดีกว่า Naive โดยเฉลี่ย): **{n_below1}**",
          "- NaN/Inf events: ไม่มี prediction non-finite · Naive MAE > 0 ทุก evaluation "
          f"(ต่ำสุด {ev['MAE_naive'].min():.6f}) · ทุก config มีครบ 6 evaluations", ""]

    L += [f"## 3. Development metrics ต่อหุ้น × fold ({LABEL})", ""]
    rows = []
    for fam, cid in sel:
        r = ev[(ev["stage"] == "search") & (ev["model"] == fam) & (ev["config_id"] == cid)]
        rows.append(r.assign(model=f"{fam} (winner)"))
    base2 = base.rename(columns={})
    allm = pd.concat(rows + [base2], ignore_index=True)
    cols = ["model", "ticker", "fold", "MAE_return", "RMSE_return", "R2_return", "MAE_baht",
            "RMSE_baht", "relMAE", "R2_OOS", "StdRatio", "Bias", "Rho", "frac_y_zero"]
    allm = allm.sort_values(["ticker", "fold", "model"])
    for t in allm["ticker"].unique():
        L += [f"### {t} ({LABEL})", ""]
        L += md_table(allm[allm["ticker"] == t], cols, pct_cols=("frac_y_zero",))
        L.append("")
    mean = allm.groupby("model")[["MAE_return", "relMAE", "R2_OOS", "StdRatio"]].mean()
    L += [f"### ค่าเฉลี่ย 6 evaluation ({LABEL})", ""]
    L += md_table(mean.reset_index(), ["model", "MAE_return", "relMAE", "R2_OOS", "StdRatio"])
    L += ["", "relMAE > 1 = แย่กว่า Naive (close_bar16 = close_bar15) · R2_OOS < 0 = MSE แย่กว่า Naive", ""]

    L += [f"## 4. ANN 3 seeds vs 10 seeds (sensitivity — ไม่เปลี่ยน winner / guard / live seeds) ({LABEL})", ""]
    ann_cid = dict(sel).get("ANN (MLP)")
    if ann_cid is not None:
        a3 = ev[(ev["stage"] == "search") & (ev["model"] == "ANN (MLP)") & (ev["config_id"] == ann_cid)]
        a10 = ev[ev["stage"] == "sensitivity10"]
        m = a3.merge(a10, on=["ticker", "fold"], suffixes=("_3", "_10"))
        L += md_table(m, ["ticker", "fold", "relMAE_3", "relMAE_10", "StdRatio_3", "StdRatio_10"])
        L += ["", f"- mean relMAE: 3 seeds {a3['relMAE'].mean():.6f} · 10 seeds {a10['relMAE'].mean():.6f} · "
              f"min StdRatio: 3 seeds {a3['StdRatio'].min():.4f} · 10 seeds {a10['StdRatio'].min():.4f}", ""]

    L += [f"## 5. Economic diagnostics — two-sided magnitude diagnostic, ไม่ใช่ backtest ({LABEL})", "",
          "cost = cost_round_trip() (สัดส่วน) · tick = tick_size(C15) (บาท) · สัดส่วนแสดงเป็น % ของวันใน eval window ·",
          "hit rate = sign(ŷ)=sign(y) บน signal days (y = 0 ไม่นับ hit) · ไม่มีการคำนวณ PnL ที่ C15", ""]
    eb = base[["model", "ticker", "fold"] + [c for c in econ.columns if c not in
                                              ("model", "config_id", "ticker", "fold")]]
    ew = econ.assign(model=econ["model"] + " (winner)").drop(columns="config_id")
    ea = pd.concat([ew, eb], ignore_index=True).sort_values(["ticker", "fold", "model"])
    ecols = ["model", "ticker", "fold", "frac_actual_ge_cost", "frac_actual_ge_1tick",
             "frac_pred_gt_cost", "signal_days", "sign_hit_rate_on_signal", "signal_days_y_zero",
             "mean_actual_move_baht", "mean_pred_move_baht"]
    L += md_table(ea, ecols, pct_cols=("frac_actual_ge_cost", "frac_actual_ge_1tick",
                                       "frac_pred_gt_cost", "sign_hit_rate_on_signal"))
    L += ["", "- forecast quality ≠ executability / profitability: ราคาที่ซื้อได้จริงเกิดหลัง prediction timestamp",
          "", "## 6. ข้อจำกัด", "",
          "- ไม่มี historical test — ตัวเลขทั้งหมด = development / walk-forward บนข้อมูลที่เคย probe แล้ว",
          "- prospective evaluation ยังไม่เปิด (ต้องมี human freeze + พิสูจน์ real-time availability ก่อน)",
          "- close_bar16 ≠ official SET close (ตรงกับ Investing close เป๊ะแค่ 45.6% / 40.5% ของวัน)",
          "- time semantics: bar semantics = working assumption · price alignment ยังไม่ตรวจ · "
          "real-time availability ยังไม่ได้พิสูจน์",
          "- ไม่มี executable backtest", ""]
    return "\n".join(L)


def main():
    ev_raw, sc_raw = _load("evals"), _load("scores")
    base, econ, folds = _load("baselines"), _load("econ"), _load("folds")
    assert (sc_raw["n_evals"] == 6).all()
    ev = build_evals(ev_raw, folds)
    sc = build_scores(sc_raw)
    ev.to_csv(D / "tuning_1600_evals.csv", index=False)
    sc.to_csv(D / "tuning_1600_scores.csv", index=False)
    (D / "summary_1600.md").write_text(build_summary(ev, sc, base, econ, folds), encoding="utf-8")
    print(f"[report_1600] เขียน tuning_1600_evals.csv ({len(ev)} แถว), "
          f"tuning_1600_scores.csv ({len(sc)} แถว), summary_1600.md")


if __name__ == "__main__":
    main()
