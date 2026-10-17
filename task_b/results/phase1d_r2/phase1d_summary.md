# Phase 1D รอบ 2 (target = ราคาปิดทางการ) search summary (development / walk-forward — ไม่ใช่ test)

- ANN (MLP): `{'alpha': 20, 'hidden_layer_sizes': (8, 4)}` primary_score (mean relMAE) = np.float64(1.036361695811949) · mean MAE_return = np.float64(0.0025677020145891886) · min StdRatio = np.float64(0.061636985849948395)
- Random Forest: `{'max_depth': 3, 'min_samples_leaf': 15, 'max_features': 0.5}` primary_score (mean relMAE) = np.float64(1.0450564633221122) · mean MAE_return = np.float64(0.002590390797811868) · min StdRatio = np.float64(0.16046630606141077)
- XGBoost: `{'max_depth': 2, 'learning_rate': 0.01, 'subsample': 0.8}` primary_score (mean relMAE) = np.float64(1.0531249349397873) · mean MAE_return = np.float64(0.0026095613498240966) · min StdRatio = np.float64(0.17530065210544343)

ราคาปิดทางการ = C15 ใน 39.2% ของวัน (ทุกแถวหลังตัด warm-up) ⇒ MAE เอื้อ prediction ที่ใกล้ 0 · StdRatio ต่ำของผู้ชนะเป็นสิ่งที่คาดได้ ไม่ใช่หลักฐานของ skill · ห้ามวางตัวเลขรอบ 2 ในตารางเดียวกับรอบ 1 หรือ daily model (คนละ target)
