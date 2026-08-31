"""
models.py — งาน A (คู่/คี่)
==========================
นิยามโมเดลของงาน A: ANN + Random Forest + XGBoost

*** เรื่อง ANN ***
ที่นี่ใช้ MLPClassifier ของ sklearn ซึ่งเป็น Multi-Layer Perceptron = ANN
ตามนิยามจริง ข้อดี: ไม่ต้องลง TensorFlow ให้ยุ่งยาก
ถ้าอาจารย์อยากได้ Keras ดูตัวอย่างการสลับที่ท้ายไฟล์

*** เรื่อง Scaling ***
ANN ต้อง standardize ไม่งั้นไม่ converge
RF ไม่ต้องก็ได้ แต่ทำไปด้วยไม่เสียหาย

ใช้ Pipeline ของ sklearn ครอบไว้ ทำให้ scaler ถูก fit
เฉพาะบน train set โดยอัตโนมัติ -> ไม่มีทาง leak
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from config import ANN_PARAMS, RF_PARAMS, XGB_PARAMS, RANDOM_STATE


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
    """ANN + Random Forest + XGBoost สำหรับงาน A (คู่/คี่)"""
    ann = _scaled(MLPClassifier(**ANN_PARAMS))
    rf = _unscaled(RandomForestClassifier(
        class_weight="balanced", **RF_PARAMS
    ))
    xgb = _unscaled(XGBClassifier(
        eval_metric="logloss", **XGB_PARAMS
    ))
    return {"ANN (MLP)": ann, "Random Forest": rf, "XGBoost": xgb}


# ---------------------------------------------------------------
# ถ้าอยากใช้ Keras แทน sklearn MLP
# ---------------------------------------------------------------
"""
ติดตั้ง:  pip install tensorflow

from tensorflow import keras
from scikeras.wrappers import KerasClassifier

def build_keras_ann(n_features):
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# แล้วเปลี่ยนใน get_classifiers เป็น
# ann = _scaled(KerasClassifier(model=build_keras_ann, epochs=100,
#                               batch_size=32, verbose=0))
"""
