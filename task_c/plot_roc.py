"""
plot_roc.py — งาน C: วาด ROC curve จาก validation predictions ที่บันทึกไว้แล้ว (ไม่เทรนใหม่)
================================================================================================

    cd task_c
    python plot_roc.py

อ่านเฉพาะไฟล์ใน results/ ที่ระบุชื่อไว้ใน PREDICTION_FILES ด้านล่าง (ระบุตรง ๆ ไม่เลือก "ไฟล์ล่าสุด" อัตโนมัติ)
  คอลัมน์ที่ใช้: Ticker, Model, Actual_Label ("ขึ้น" = 1 / "ลง/นิ่ง" = 0), Probability_Up
  (ไฟล์ *_logistic_validation_predictions.csv รายหุ้น, knn_* และ task2_intraday_* / intraday_13h_* เป็นการทดลองอื่น
   ไม่ใช่ validation ของ 3 โมเดลหลัก จึงไม่ใช้)
AUC คำนวณด้วย sklearn.metrics.roc_auc_score ตัวเดียวกับ evaluate.classification_metrics แล้วตรวจให้ตรงกัน
เขียนผลที่ results/curves/ เท่านั้น -- ไม่แตะไฟล์ผลเดิม
matplotlib import ในฟังก์ชันเท่านั้น (backend Agg)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
CURVE_DIR = RESULTS_DIR / "curves"

# ไฟล์ validation predictions ของ 3 โมเดลหลัก (รันเดียวกัน 2026-09-13 22:13:43) -- ห้ามเปลี่ยนเป็น glob/latest
PREDICTION_FILES = {
    "ANN (MLP)": "ann_mlp_validation_predictions_20260913_221343.csv",
    "Logistic Regression": "logistic_regression_validation_predictions_20260913_221343.csv",
    "XGBoost": "xgboost_validation_predictions_20260913_221343.csv",
}
REQUIRED_COLUMNS = ["Ticker", "Model", "Actual_Label", "Probability_Up"]
POSITIVE_LABEL = "ขึ้น"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def load_predictions():
    frames = []
    for model, fname in PREDICTION_FILES.items():
        path = RESULTS_DIR / fname
        df = pd.read_csv(path, encoding="utf-8-sig")
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{fname}: ขาดคอลัมน์ {missing} (มี {list(df.columns)})")
        if set(df["Model"].unique()) != {model}:
            raise ValueError(f"{fname}: คอลัมน์ Model = {df['Model'].unique()} ไม่ตรงกับ {model!r}")
        unknown = set(df["Actual_Label"].unique()) - {POSITIVE_LABEL, "ลง/นิ่ง"}
        if unknown:
            raise ValueError(f"{fname}: Actual_Label มีค่าที่ไม่รู้จัก {unknown}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def check_against_evaluate(y, p, auc):
    """AUC ต้องตรงกับ evaluate.classification_metrics (ตัวที่ main.py ใช้รายงานผล)"""
    from evaluate import classification_metrics
    ref = classification_metrics(y, (p >= 0.5).astype(int), p)["ROC-AUC"]
    if not np.isclose(ref, auc, rtol=0, atol=1e-12):
        raise AssertionError(f"AUC ไม่ตรงกับ evaluate.py: {auc} vs {ref}")


def make_roc():
    plt = _plt()
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_predictions()
    summary = []
    for ticker, g in data.groupby("Ticker"):
        fig, ax = plt.subplots(figsize=(6.5, 6))
        for model in PREDICTION_FILES:
            m = g[g["Model"] == model]
            y = (m["Actual_Label"] == POSITIVE_LABEL).astype(int).to_numpy()
            p = m["Probability_Up"].to_numpy(dtype=float)
            auc = roc_auc_score(y, p)
            check_against_evaluate(y, p, auc)
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, linewidth=1.8, label=f"{model} (AUC = {auc:.4f})")
            summary.append({"ticker": ticker, "model": model, "n": len(m), "roc_auc": auc})
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess (AUC = 0.5000)")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.set_title(f"{ticker} - ROC curve (validation set)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        png = CURVE_DIR / f"{ticker.replace('.', '_').replace('^', '')}_roc_validation.png"
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"[roc] เขียน {png.relative_to(BASE_DIR)}")
    csv = CURVE_DIR / "roc_auc_validation.csv"
    pd.DataFrame(summary).to_csv(csv, index=False, encoding="utf-8-sig")
    print(f"[roc] เขียน {csv.relative_to(BASE_DIR)}")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    make_roc()
