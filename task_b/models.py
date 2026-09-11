"""
models.py — งาน B (ราคาปิด / return)
====================================
นิยามโมเดลของงาน B: ANN + Random Forest + XGBoost (regression)

*** เรื่อง ANN ***
ที่นี่ใช้ MLPRegressor ของ sklearn ซึ่งเป็น Multi-Layer Perceptron = ANN
ตามนิยามจริง ข้อดี: ไม่ต้องลง TensorFlow ให้ยุ่งยาก

*** เรื่อง Scaling ***
ANN ต้อง standardize ไม่งั้นไม่ converge
RF ไม่ต้องก็ได้ แต่ทำไปด้วยไม่เสียหาย

ใช้ Pipeline ของ sklearn ครอบไว้ ทำให้ scaler ถูก fit
เฉพาะบน train set โดยอัตโนมัติ -> ไม่มีทาง leak
"""

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from config import ANN_PARAMS, ANN_SEEDS, RF_PARAMS, XGB_PARAMS


class SeedAveragedRegressor(RegressorMixin, BaseEstimator):
    """
    เทรนโมเดลเดิมซ้ำด้วย random_state หลายค่า แล้วเฉลี่ยการทำนาย

    ใช้กับ ANN เพราะผลของ MLP ขึ้นกับการสุ่ม weight เริ่มต้น -- ดูเหตุผลที่
    ANN_SEEDS ใน config.py ทุก seed น้ำหนักเท่ากัน ไม่มีการเลือก seed
    """

    def __init__(self, estimator=None, seeds=(0,)):
        self.estimator = estimator
        self.seeds = seeds

    def fit(self, X, y):
        self.estimators_ = [
            clone(self.estimator).set_params(random_state=s).fit(X, y)
            for s in self.seeds
        ]
        return self

    def predict(self, X):
        return np.mean([e.predict(X) for e in self.estimators_], axis=0)


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


def get_regressors():
    """
    ANN + Random Forest + XGBoost สำหรับทำนาย return

    หมายเหตุ: ค่า return มีขนาดเล็กมาก (~0.01)
    ANN อาจเทรนไม่ค่อยดี -> ใน main.py จะมีการ scale target ให้
    """
    ann = _scaled(SeedAveragedRegressor(MLPRegressor(**ANN_PARAMS),
                                        seeds=tuple(ANN_SEEDS)))
    rf = _unscaled(RandomForestRegressor(**RF_PARAMS))
    xgb = _unscaled(XGBRegressor(**XGB_PARAMS))
    return {"ANN (MLP)": ann, "Random Forest": rf, "XGBoost": xgb}
