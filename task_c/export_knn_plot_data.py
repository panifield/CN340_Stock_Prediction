"""สร้างผล K-Nearest Neighbors สำหรับ plot โดยไม่ใช้ test set.

รันจาก task_c:
    python export_knn_plot_data.py

ผลลัพธ์ใน results/:
  - knn_validation_metrics_by_k_<timestamp>.csv
    ใช้ plot K เทียบ Accuracy / Precision / Recall / F1 / ROC-AUC
  - knn_validation_predictions_<timestamp>.csv
    ใช้ plot probability/prediction ตามวัน (มีผลครบทุก K และทั้งสองหุ้น)
"""

import os
from datetime import datetime

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import OUTPUT_DIR, TICKERS
from data_loader import load_stock
from evaluate import classification_metrics
from features import build_features
from main import _prepare
from splits import chronological_split
from targets import build_targets


# ใช้จำนวนเพื่อนบ้านเป็นเลขคี่ เพื่อลดโอกาสคะแนนโหวตเสมอใน binary classification
K_VALUES = (1, 3, 5, 7, 9, 11, 15, 21)


def make_knn(k):
    """KNN ต้อง scale เพราะ Euclidean distance ไวต่อขนาดของ feature."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=k, weights="uniform", metric="minkowski", p=2,
        )),
    ])


def load_train_validation(ticker):
    """สร้าง X/y แล้วคืนเฉพาะ train และ validation; ไม่อ่าน test."""
    df = load_stock(ticker, verbose=False)
    X = build_features(df, verbose=False)
    targets = build_targets(df, verbose=False)
    X, y, _ = _prepare(X, targets["y_updown"], targets[["is_flat", "prev_close"]])
    parts = chronological_split(X, y, verbose=False)
    return parts["train"], parts["val"]


def main():
    metric_rows, prediction_rows = [], []
    for ticker in TICKERS:
        (X_train, y_train), (X_val, y_val) = load_train_validation(ticker)
        for k in K_VALUES:
            model = make_knn(k)
            model.fit(X_train, y_train)
            predicted = model.predict(X_val)
            probability_up = model.predict_proba(X_val)[:, 1]
            metrics = classification_metrics(y_val, predicted, probability_up)
            metric_rows.append({"Ticker": ticker, "K": k, **metrics})

            prediction_rows.append(pd.DataFrame({
                "Ticker": ticker,
                "K": k,
                "Date": X_val.index,
                "Actual": y_val.astype(int).to_numpy(),
                "Probability_Up": probability_up,
                "Predicted": predicted.astype(int),
                "Correct": (predicted == y_val.to_numpy()).astype(int),
            }))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    metric_path = os.path.join(
        OUTPUT_DIR, f"knn_validation_metrics_by_k_{run_at}.csv"
    )
    prediction_path = os.path.join(
        OUTPUT_DIR, f"knn_validation_predictions_{run_at}.csv"
    )
    pd.DataFrame(metric_rows).to_csv(metric_path, index=False, encoding="utf-8-sig")
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        prediction_path, index=False, encoding="utf-8-sig"
    )

    print("KNN validation only — test set was not used")
    print(pd.DataFrame(metric_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nmetrics: {metric_path}")
    print(f"predictions: {prediction_path}")


if __name__ == "__main__":
    main()
