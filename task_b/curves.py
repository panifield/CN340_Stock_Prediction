"""
curves.py — งาน B: วาด curve เพิ่มเติม (เรียกผ่าน `python main.py --dev --curves` เท่านั้น)
================================================================================================
ไม่เทรนโมเดลใหม่ ไม่แก้ผลเดิม: ใช้ผลที่ main.py คำนวณไว้แล้ว (preds / fitted) และไฟล์จูนเดิมที่มีอยู่
เขียนลง results/curves/ เท่านั้น (ไม่แตะ results/*_val.csv)

  1) REC curve         {ticker}_rec_{stage}.csv/.png     สัดส่วนวันที่ |error_return| <= tolerance (โมเดล + baseline)
  2) ANN loss curve    {ticker}_ann_loss_{stage}.png     training loss ต่อ iteration ของทุก seed
  3) ANN alpha curve   ann_alpha_curve.csv/.png          อ่านจาก results/tuning_stage2_params.csv (ไม่เทรนใหม่)

matplotlib ไม่อยู่ใน requirements ของ task_b -> import ภายในฟังก์ชันเท่านั้น (ใช้ backend Agg)
"""

import numpy as np
import pandas as pd

from config import OUTPUT_DIR

CURVE_DIR = OUTPUT_DIR / "curves"

# เท่ากับ GUARD_MIN_STDRATIO ใน tune.py (hardcode ไว้ตามข้อกำหนด ห้าม import tune)
GUARD_MIN_STDRATIO = 0.05
NAIVE = "Baseline: Naive (RW)"
ANN = "ANN (MLP)"
N_GRID = 200


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:   # task_b/requirements.txt ไม่มี matplotlib
        raise RuntimeError("--curves ต้องใช้ matplotlib (pip install matplotlib) "
                           "-- ไม่อยู่ใน task_b/requirements.txt โดยตั้งใจ") from e
    return plt


def _stem(ticker):
    return ticker.replace(".", "_").replace("^", "")


def _report(path):
    print(f"[curves] เขียน {path.relative_to(OUTPUT_DIR.parent)}")


# ---------------------------------------------------------------
# 1) REC curve
# ---------------------------------------------------------------

def rec_curve(ticker, res):
    y = np.asarray(res["y_eval"], dtype=float)
    errs = {name: np.abs(y - np.asarray(p, dtype=float)) for name, p in res["preds"].items()}
    top = float(np.quantile(np.concatenate(list(errs.values())), 0.99))
    grid = np.linspace(0.0, top, N_GRID)

    rows = []
    for name, e in errs.items():
        s = np.sort(e)
        acc = np.searchsorted(s, grid, side="right") / len(s)
        rows += [{"predictor": name, "tolerance": g, "accuracy": a} for g, a in zip(grid, acc)]
    df = pd.DataFrame(rows, columns=["predictor", "tolerance", "accuracy"])

    stage, stem = res["stage"], _stem(ticker)
    path = CURVE_DIR / f"{stem}_rec_{stage}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    _report(path)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in errs:
        d = df[df["predictor"] == name]
        is_base = name.startswith("Baseline")
        ax.plot(d["tolerance"] * 100, d["accuracy"],
                linestyle="--" if is_base else "-",
                linewidth=2.6 if name == NAIVE else 1.6, label=name)
    ax.set_xlabel("Tolerance on |error| of return (%)")
    ax.set_ylabel("Share of days with |error| <= tolerance")
    ax.set_title(f"{ticker} - REC curve ({stage} set, error in return units)\n"
                 "higher / further left = better; dashed = baselines", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    png = path.with_suffix(".png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    _report(png)


# ---------------------------------------------------------------
# 2) ANN loss curve (training loss เท่านั้น)
# ---------------------------------------------------------------

def ann_loss_curve(ticker, res):
    # ต้องใช้ .regressor_ (ตัวที่ fit แล้ว) ไม่ใช่ .regressor
    ann = res["fitted"][ANN].regressor_.named_steps["model"]    # SeedAveragedRegressor
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for seed, est in zip(ann.seeds, ann.estimators_):
        lc = np.asarray(est.loss_curve_, dtype=float)
        it = np.arange(1, len(lc) + 1)
        ax.plot(it, lc, linewidth=1.1, label=f"seed {seed} ({len(lc)} it)")
        ax.plot(it[-1], lc[-1], "o", color="black", markersize=4, zorder=5)
    ax.plot([], [], "o", color="black", markersize=4, label="last iteration")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration (epoch)")
    ax.set_ylabel("Training loss (standardized target, incl. L2 term)")
    ax.set_title(f"{ticker} - ANN training loss only ({res['stage']} run)\n"
                 "early_stopping=False, so there is no validation loss", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    png = CURVE_DIR / f"{_stem(ticker)}_ann_loss_{res['stage']}.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    _report(png)


# ---------------------------------------------------------------
# 3) ANN alpha curve -- อ่านผลจูนเดิม (ไม่เทรนใหม่)
# ---------------------------------------------------------------

def ann_alpha_curve():
    src = OUTPUT_DIR / "tuning_stage2_params.csv"
    if not src.exists():
        print(f"[curves] ข้าม alpha curve: ไม่พบ {src.name}")
        return
    d = pd.read_csv(src)
    d = d[d["model"] == ANN]
    g = (d.groupby(["hidden_layer_sizes", "alpha"])
          .agg(mean_MAE_return=("MAE_return", "mean"),
               mean_StdRatio=("StdRatio", "mean"),
               min_StdRatio=("StdRatio", "min"))
          .reset_index().sort_values(["hidden_layer_sizes", "alpha"]))
    path = CURVE_DIR / "ann_alpha_curve.csv"
    g.to_csv(path, index=False, encoding="utf-8-sig")
    _report(path)

    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    plt = _plt()
    alphas = sorted(g["alpha"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for hs, grp in g.groupby("hidden_layer_sizes"):
        axes[0].plot(grp["alpha"], grp["mean_MAE_return"], marker="o", label=f"hidden {hs}")
        axes[1].plot(grp["alpha"], grp["min_StdRatio"], marker="o", label=f"hidden {hs}")
    axes[1].axhline(GUARD_MIN_STDRATIO, color="red", linestyle="--", linewidth=1.2,
                    label=f"guard = {GUARD_MIN_STDRATIO}")
    axes[0].set_ylabel("mean MAE_return (lower = better)")
    axes[1].set_ylabel("min StdRatio over folds (higher = less collapsed)")
    axes[0].set_title("Accuracy vs alpha", fontsize=10)
    axes[1].set_title("Prediction collapse vs alpha", fontsize=10)
    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(alphas))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("alpha (L2 strength, log scale)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("ANN alpha curve - from walk-forward folds on train+val "
                 "(tuning_stage2_params.csv, no retraining)", fontsize=11)
    fig.tight_layout()
    png = path.with_suffix(".png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    _report(png)


def make_all(all_results):
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    for r in all_results:
        rec_curve(r["ticker"], r["b"])
        ann_loss_curve(r["ticker"], r["b"])
    ann_alpha_curve()
