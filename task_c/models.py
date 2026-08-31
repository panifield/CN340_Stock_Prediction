"""
models.py — งาน C (ขึ้น/ลง)
===========================
นิยามโมเดลของงาน C: ANN + Logistic Regression + XGBoost

ทำไมงาน C ใช้ Logistic Regression?
  เพื่อให้รายงานมีครบ 3 แนวทาง ไม่ใช่โมเดลคล้ายกัน 3 ตัว
      Logistic Regression -> เชิงเส้น
      XGBoost             -> tree-based ไม่เชิงเส้น
      ANN                 -> neural network ไม่เชิงเส้น

*** เรื่อง Scaling ***
ANN และ Logistic Regression ต้อง standardize ไม่งั้นไม่ converge
XGBoost ไม่ต้องก็ได้ แต่ทำไปด้วยไม่เสียหาย

ใช้ Pipeline ของ sklearn ครอบไว้ ทำให้ scaler ถูก fit
เฉพาะบน train set โดยอัตโนมัติ -> ไม่มีทาง leak
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from config import ANN_PARAMS, XGB_PARAMS, LOGREG_PARAMS


def _scaled(estimator):
    """ครอบโมเดลด้วย imputer + scaler"""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def _unscaled(estimator):
    """สำหรับ tree-based ที่ไม่ต้อง scale (แต่ยังต้อง impute)"""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])


def get_classifiers():
    """ANN + Logistic Regression + XGBoost สำหรับงาน C (ขึ้น/ลง)"""
    ann = _scaled(MLPClassifier(**ANN_PARAMS))
    logreg = _scaled(LogisticRegression(
        class_weight="balanced", **LOGREG_PARAMS
    ))
    xgb = _unscaled(XGBClassifier(
        eval_metric="logloss", **XGB_PARAMS
    ))
    return {"ANN (MLP)": ann,
            "Logistic Regression": logreg,
            "XGBoost": xgb}
