"""Export ข้อมูลสำหรับ plot ของ Logistic Regression โดยไม่ใช้ test set.

รันจาก task_c:
    python export_logistic_plot_data.py

สร้างไฟล์ใน results/:
  - <ticker>_logistic_coefficients_val.csv : bar chart coefficient
  - <ticker>_logistic_validation_predictions.csv : probability/label ตามเวลา,
    ใช้ทำ line chart, ROC curve หรือ calibration plot
"""

import os

import pandas as pd

from config import OUTPUT_DIR, TICKERS
from data_loader import load_stock
from features import build_features
from main import _prepare
from models import get_classifiers
from splits import chronological_split
from targets import build_targets


def export_one_ticker(ticker):
    """Fit Logistic บน train และ export เฉพาะ validation artifacts."""
    df = load_stock(ticker, verbose=False)
    X = build_features(df, verbose=False)
    targets = build_targets(df, verbose=False)
    X, y, _ = _prepare(X, targets["y_updown"], targets[["is_flat", "prev_close"]])
    parts = chronological_split(X, y, verbose=False)
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]

    # ใช้โมเดล/feature ชุดเดียวกับ main.py ค่า default = feature selection none
    model = get_classifiers()["Logistic Regression"]
    model.fit(X_train, y_train)
    probability_up = model.predict_proba(X_val)[:, 1]
    prediction = model.predict(X_val)
    estimator = model.named_steps["model"]

    ticker_safe = ticker.replace(".", "_").replace("^", "")
    coefficient_data = pd.DataFrame({
        "feature": X_train.columns,
        # coefficient หลัง StandardScaler: เปรียบขนาดข้าม feature ได้
        "standardized_coefficient": estimator.coef_.ravel(),
        "absolute_coefficient": estimator.coef_.ravel().__abs__(),
    }).sort_values("absolute_coefficient", ascending=False)
    coefficient_data["direction"] = coefficient_data[
        "standardized_coefficient"
    ].map(lambda value: "up" if value > 0 else "down_or_flat")
    coefficient_data["logistic_intercept"] = estimator.intercept_[0]

    prediction_data = pd.DataFrame({
        "date": X_val.index,
        "actual_updown": y_val.astype(int).to_numpy(),
        "probability_up": probability_up,
        "predicted_updown": prediction.astype(int),
    })
    prediction_data["correct"] = (
        prediction_data["actual_updown"] == prediction_data["predicted_updown"]
    ).astype(int)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    coefficient_path = os.path.join(
        OUTPUT_DIR, f"{ticker_safe}_logistic_coefficients_val.csv"
    )
    prediction_path = os.path.join(
        OUTPUT_DIR, f"{ticker_safe}_logistic_validation_predictions.csv"
    )
    coefficient_data.to_csv(coefficient_path, index=False, encoding="utf-8-sig")
    prediction_data.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    return coefficient_path, prediction_path, estimator.intercept_[0]


def main():
    print("Export Logistic Regression plot data (train → validation only)")
    for ticker in TICKERS:
        coefficient_path, prediction_path, intercept = export_one_ticker(ticker)
        print(f"{ticker}: intercept={intercept:.6f}")
        print(f"  coefficients: {coefficient_path}")
        print(f"  validation predictions: {prediction_path}")


if __name__ == "__main__":
    main()
