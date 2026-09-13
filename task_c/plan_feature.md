# แผนการปรับปรุง Features สำหรับ Model

เอกสารนี้สรุปแผนการปรับปรุง features ที่ใช้ในการเทรนโมเดล (Logistic Regression, ANN) เพื่อแก้ไขปัญหา Accuracy ที่ลดลง โดยเน้นที่การจัดการกับข้อมูลที่ไม่เหมาะสม (Raw data), การจัดการ Outlier/Missing value, การเพิ่ม Features ที่มีประสิทธิภาพ, และการลดปัญหา Multicollinearity

## 1. Features ที่ควรตัดหรือแปลง (เพื่อแก้ปัญหา Accuracy)

ปัญหาหลักในตอนนี้คือการใช้ข้อมูลดิบ (Raw Data) ซึ่งเป็น Non-stationary และการไม่จัดการกับ Outliers อย่างเหมาะสม

### 1.1 ตัดหรือแปลง Raw price/volume (close, open, high, low, volume)
*   **ปัญหา:** ราคาหุ้นเป็นข้อมูล Non-stationary (Scale ของราคาเปลี่ยนไปตามเวลา) เมื่อแบ่งข้อมูลแบบ time-based ไม่ shuffle โมเดล (โดยเฉพาะ Logistic Regression และ ANN) จะเรียนรู้ scale ของช่วง train และจะไม่สามารถ generalize ไปยังช่วง test ที่อาจมีราคาใน scale ที่ต่างกันมากได้
*   **การแก้ไข (เลือกอย่างใดอย่างหนึ่ง):**
    *   **ตัดทิ้ง:** ลบ raw price และ volume ออกไปเลย
    *   **แปลง (Relative Scaling):** ถ้าต้องการเก็บข้อมูลราคาไว้ ให้แปลงเป็นสัดส่วนเทียบกับเส้นค่าเฉลี่ย เช่น `close / close.rolling(252).mean()`

### 1.2 จัดการ `volume_change` (ป้องกัน `inf`)
*   **ปัญหา:** การใช้ `pct_change()` กับ volume หาก volume วันก่อนหน้าเป็น 0 จะทำให้ค่าเป็น `inf` ซึ่งทำให้ loss function ของ Logistic Regression และ ANN เสียหายได้ง่าย
*   **การแก้ไข:** ทำการคลิป (Clip) หรือเปลี่ยนค่า `inf` ให้เป็น `nan` ก่อนนำไปใช้งานต่อไป

### 1.3 Winsorize (ตัด Outlier) สำหรับ Return-based features
*   **ปัญหา:** Features อย่าง `ret_Nd`, `volume_change` ในช่วงวันที่มีข่าวสำคัญหรือประกาศผลประกอบการ จะทำให้ค่าพุ่งสูง/ต่ำสุดขั้ว (Outlier) ทำให้ loss function ถูกดึงไปทาง Outlier มากเกินไป
*   **การแก้ไข:** ทำ Winsorize (Clip) ข้อมูลที่ Percentile 1-99 หลังจากคำนวณใน `build_raw_features` เสร็จแล้ว

---

## 2. Features ที่ควรเพิ่ม (Predictive Signals แบบไม่ Leak)

เพิ่ม Features ที่ช่วยบอกแนวโน้มและสภาพตลาดได้ดีขึ้น โดยต้องมั่นใจว่าใช้เฉพาะข้อมูลในอดีต (ไม่เกิด Look-ahead bias)

### 2.1 Streak / Momentum ของทิศทางราคา
บอกความต่อเนื่องของทิศทางราคา
```python
direction = np.sign(close.diff())
f["up_streak"] = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
f["prev_direction"] = direction  # จะโดน shift(1) ทีเดียวตอนท้ายเหมือนตัวอื่น
```

### 2.2 ระยะห่างจาก Rolling High/Low (Breakout signal)
ใช้ดูว่าราคาอยู่ห่างจากจุดสูงสุด/ต่ำสุดในรอบ N วันแค่ไหน
```python
f["dist_from_high20"] = close / close.rolling(20).max() - 1
f["dist_from_low20"]  = close / close.rolling(20).min() - 1
```

### 2.3 Bollinger Band Width (Volatility Regime)
ใช้วัดความผันผวนของตลาด นอกเหนือจาก `bb_position` และ `volatility_Nd`
```python
f["bb_width"] = width / ma  # width ที่คำนวณอยู่แล้วใน bollinger_position
```

### 2.4 Normalized ATR (Average True Range)
ใช้วัดความผันผวนรวมช่วง Gap ด้วย ซึ่งดีกว่าการใช้ `hl_range` ที่ดูแค่ความยาวของแท่งเทียนเดียว

---

## 3. จัดการปัญหา Multicollinearity (สำหรับ Logistic Regression)

Features หลายตัวที่มาจากฐานข้อมูลเดียวกัน (เช่น `macd`, `macd_signal`, `macd_hist` มาจาก `close` และกลุ่มเส้นค่าเฉลี่ย `close_over_sma{w}`) มักจะมี Correlation กันสูง ทำให้สัมประสิทธิ์ (Coefficient) ของ Logistic Regression ไม่เสถียร (High Variance) ส่งผลให้ loss function ทำงานยากขึ้น

*   **การแก้ไข:**
    1.  **ใช้ L2 Regularization:** ตรวจสอบและตั้งค่าให้ใช้ L2 penalty ในโมเดล Logistic Regression (ในไฟล์ `models.py`)
    2.  **คัดเลือก Features (Feature Selection):** ตรวจสอบ Correlation matrix หากพบเส้นค่าเฉลี่ยที่มี window ใกล้เคียงกันเกินไป (เช่น 5, 10, 20) ให้พิจารณาตัดออกบางส่วน (เช่น เหลือแค่ 5 กับ 20)

---

## สรุปแผนการปรับปรุง

| การดำเนินการ | ผลที่คาดว่าจะได้รับ |
| :--- | :--- |
| **ตัด** raw `close`, `open`, `high`, `low`, `volume` | ลด Overfitting กับ scale ของราคาในช่วง train ช่วยให้ Generalize ดีขึ้นตอน test |
| **Clip/Winsorize** return-based features | Loss function ไม่ถูกดึงด้วย Outlier ทำให้การเทรนเสถียรมากขึ้น |
| **แก้** `volume_change` ที่เป็น `inf` | ป้องกัน `NaN` หรือ `inf` หลุดเข้าโมเดล ทำให้คำนวณ Error ได้อย่างถูกต้อง |
| **เพิ่ม** Streak, Distance from High/Low, BB Width, ATR | เพิ่มสัญญาณ (Signals) ใหม่ๆ ที่ไม่ซ้ำซ้อน (Redundant) กับ Features เดิมที่มีอยู่ |
| **Regularize / ตัด** MA ที่ Correlate กันสูง | ทำให้โมเดล Logistic Regression ทำงานได้เสถียรขึ้น และปรับลด Variance ลง |

**หมายเหตุสำคัญ:**
การปรับปรุงทั้งหมดตามที่แนะนำมานี้ ยังคงทำภายในแพทเทิร์นเดิมของไฟล์ นั่นคือ **ทำการคำนวณใน `build_raw_features` ทั้งหมด และปล่อยให้การป้องกัน Data Leak จัดการด้วยคำสั่ง `shift(1)` ในขั้นตอนสุดท้าย** ไม่มีความจำเป็นต้องเข้าไปแก้ Logic การ shift ใดๆ เพิ่มเติม
