# Phase 1D search summary (development / walk-forward — ไม่ใช่ test)

- ANN (MLP): `{'alpha': 20, 'hidden_layer_sizes': (8, 4)}` primary_score (mean relMAE) = np.float64(1.057461706643797) · mean MAE_return = np.float64(0.002229071638024418) · min StdRatio = np.float64(0.13091165038702687)
- Random Forest: `{'max_depth': 3, 'min_samples_leaf': 15, 'max_features': 0.5}` primary_score (mean relMAE) = np.float64(1.0512815164186) · mean MAE_return = np.float64(0.0022179875098033244) · min StdRatio = np.float64(0.19971422737564168)
- XGBoost: `{'max_depth': 2, 'learning_rate': 0.01, 'subsample': 1.0}` primary_score (mean relMAE) = np.float64(1.0543072153182573) · mean MAE_return = np.float64(0.0022220364845228755) · min StdRatio = np.float64(0.18528026181805585)

~43–49% ของวันราคาไม่ขยับเลย ⇒ MAE เอื้อ prediction ที่ใกล้ 0 · StdRatio ต่ำของผู้ชนะเป็นสิ่งที่คาดได้ ไม่ใช่หลักฐานของ skill
