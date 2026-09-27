"""
models.py — งาน A (คู่/คี่)
==========================
นิยามโมเดลของงาน A: ANN + LightGBM + Logistic Regression

*** เรื่อง ANN ***
ที่นี่ใช้ MLPClassifier ของ sklearn ซึ่งเป็น Multi-Layer Perceptron = ANN
ตามนิยามจริง ข้อดี: ไม่ต้องลง TensorFlow ให้ยุ่งยาก
ถ้าอาจารย์อยากได้ Keras ดูตัวอย่างการสลับที่ท้ายไฟล์

*** ทำไมเปลี่ยนจาก RF/XGBoost เป็น LightGBM + Logistic Regression ***
- LightGBM (ตัวหลัก) ตั้งค่าแบบระวัง overfit หนักๆ (leaf น้อย, depth ตื้น,
  learning rate ต่ำ, reg_lambda สูง) เพราะ signal ของ y_flip อ่อนมาก
  จับ interaction สำคัญ (เช่น vol × n_mod2) ได้ ซึ่ง linear model
  จับไม่ได้ถ้าไม่ใส่ interaction term เอง
- Logistic Regression + L2 ไม่ได้ใส่มาเพื่อชนะ accuracy แต่เพื่อดู
  coefficient ว่า feature ไหนมีผลจริง เอาไปเขียนรายงานได้
- รวมกับ ANN แล้วโมเดลทั้ง 3 ตัวครอบคลุม 3 ประเภท: เชิงเส้น (LogReg) /
  tree-based (LightGBM) / neural network (ANN) ไม่ใช่โมเดลคล้ายกัน 3 ตัว

*** เรื่อง Scaling ***
ANN และ Logistic Regression ต้อง standardize ไม่งั้นไม่ converge / ตี
ความหมาย coefficient ผิด
LightGBM ไม่ต้องก็ได้ แต่ยัง impute เหมือนกัน

ใช้ Pipeline ของ sklearn ครอบไว้ ทำให้ scaler ถูก fit
เฉพาะบน train set โดยอัตโนมัติ -> ไม่มีทาง leak
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier

from config import ANN_PARAMS, LGBM_PARAMS, LOGREG_PARAMS, RANDOM_STATE


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
    """ANN + LightGBM + Logistic Regression สำหรับงาน A (คู่/คี่)"""
    ann = _scaled(MLPClassifier(**ANN_PARAMS))
    lgbm = _unscaled(LGBMClassifier(**LGBM_PARAMS))
    logreg = _scaled(LogisticRegression(**LOGREG_PARAMS))
    return {
        "ANN (MLP)": ann,
        "LightGBM": lgbm,
        "Logistic Regression": logreg,
    }


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
