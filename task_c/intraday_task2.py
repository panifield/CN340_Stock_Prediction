"""Task 2 — ทำนายทิศทางราคาปิดจากข้อมูลระหว่างวันก่อนตลาดปิด.

โจทย์เวลา
-----------
ใช้ข้อมูลแท่ง 1 ชั่วโมงของ "วัน D" ถึง cutoff (ค่า default 15:00) เพื่อ
ทำนายว่าราคาปิดทางการ (ATC) ของวัน D จะสูงกว่าราคาที่ cutoff หรือไม่:

    y_intraday[D] = 1  if OfficialClose[D] > Close[D, cutoff]
                     0  otherwise (ลงหรือเท่าเดิม)

Task นี้แยกจาก main.py / Task 1 โดยสมบูรณ์:
  - Task 1: ก่อนตลาดเปิด ใช้ข้อมูลถึงวันก่อนหน้า ทำนายวันถัดไป
  - Task 2: ระหว่างวัน ใช้ข้อมูลวันเดียวกันถึง cutoff ทำนาย close ของวันนั้น

*** แก้ 2026-09-29: เปลี่ยนแหล่งข้อมูลของ target (OfficialClose) ***
เดิม target มาจากแท่ง 1h ที่ Hour == 16 ใน data_cache/{ticker}_1h_730d.csv
(yfinance รายชั่วโมง) -- ตรวจกับราคาปิดทางการจริงจาก Settrade แล้ว
(check_settrade_atc_match.py) พบว่า **ตรงกันแค่ 40-46%** (KBANK 46.0%,
ADVANC 40.2% จาก 721 วัน) เพราะแท่ง 1h คือราคาซื้อขายต่อเนื่องช่วง
15:00-16:00 ไม่ใช่ราคาจากรอบ ATC (call auction) ตอนปิดตลาดจริง

ตอนนี้เปลี่ยนไปใช้ yfinance **daily-interval** close แทน (ดึงแยกจาก 1h,
cache ไว้ที่ data_cache/{ticker}_daily_close.csv) ซึ่งยืนยันแล้วว่าตรงกับ
ราคาปิดทางการ 100% (ดู task_b/tools/fetch_yahoo_daily.py ที่เทียบกับ
investing.com) -- INTRADAY_CLOSE_HOUR ใน config.py ยังใช้เป็นคำอธิบาย
เจตนา ("ราคาปิดคือ ~16:00 ตามเวลาตลาด") แต่ไม่ได้ใช้เลือกแท่งอีกต่อไป

รันจากโฟลเดอร์ task_c:
    python intraday_task2.py --dev  # train/validation; ไม่แตะ test
    python intraday_task2.py        # train/validation/test
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    INTRADAY_CLOSE_HOUR, INTRADAY_CUTOFF_HOUR, OUTPUT_DIR, TICKERS,
)
from evaluate import classification_metrics, print_confusion, print_table, results_table
from models import get_classifiers
from splits import chronological_split


REQUIRED_HOURLY_COLUMNS = {"Datetime", "Open", "High", "Low", "Close", "Volume"}


def _hourly_path(ticker):
    safe = ticker.replace("^", "").replace(".", "_")
    return os.path.join("data_cache", f"{safe}_1h_730d.csv")


def _daily_close_path(ticker):
    safe = ticker.replace("^", "").replace(".", "_")
    return os.path.join("data_cache", f"{safe}_daily_close.csv")


def load_hourly(ticker):
    """โหลดแท่ง 1 ชั่วโมงและตรวจ schema ที่จำเป็น."""
    path = _hourly_path(ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบไฟล์ intraday: {path}")
    df = pd.read_csv(path, parse_dates=["Datetime"])
    missing = REQUIRED_HOURLY_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} ไม่มีคอลัมน์: {sorted(missing)}")
    df = df.sort_values("Datetime").drop_duplicates("Datetime").copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[df["Close"] > 0]
    # Datetime ในไฟล์มี +07:00; normalize โดยคงเขตเวลาท้องถิ่นไว้
    df["Date"] = df["Datetime"].dt.normalize()
    df["Hour"] = df["Datetime"].dt.hour
    return df


def load_daily_close(ticker, start, end, use_cache=True):
    """
    ราคาปิดทางการรายวันจริงจาก yfinance (interval=1d) -- ใช้เป็น target ของ
    Task 2 แทนแท่ง 1h ที่ Hour==close_hour (ดูเหตุผลใน docstring หัวไฟล์)

    cache ไว้แยกจาก data_cache/{ticker}_1h_730d.csv เพราะคนละ query กัน
    (interval ต่างกัน) ห้ามใช้ปนกับ cache ของ task_a/task_c เดิมที่แช่แข็ง
    ไว้ที่ END_DATE ของ config.py -- ไฟล์นี้ต้องครอบคลุมช่วงเดียวกับข้อมูล
    1h ซึ่งขยับตามวันที่ดึงแต่ละครั้ง ไม่ใช่ค่าคงที่
    """
    # yfinance daily index ไม่มี tz -- ตัด tz ของ start/end ทิ้งก่อนเทียบ/query
    # เสมอ (ผู้เรียกอาจส่ง tz-aware Timestamp มาจาก hourly["Date"] ที่เป็น
    # Asia/Bangkok)
    start = pd.Timestamp(start).tz_localize(None) if pd.Timestamp(start).tz else pd.Timestamp(start)
    end = pd.Timestamp(end).tz_localize(None) if pd.Timestamp(end).tz else pd.Timestamp(end)

    path = _daily_close_path(ticker)
    if use_cache and os.path.exists(path):
        s = pd.read_csv(path, index_col=0, parse_dates=True)["Close"]
        s.index = s.index.normalize()
        s_start, s_end = s.index.min(), s.index.max()
        if s_start <= pd.Timestamp(start) and s_end >= pd.Timestamp(end) - pd.Timedelta(days=3):
            return s
        # cache ไม่ครอบคลุมช่วงที่ต้องการ (เช่นข้อมูล 1h ใหม่กว่า) -> ดึงใหม่

    import yfinance as yf
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                     end=end_exclusive, interval="1d",
                     auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"ดึงราคาปิดรายวันของ {ticker} ไม่ได้")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna().copy()
    s.index = pd.to_datetime(s.index).normalize()
    os.makedirs("data_cache", exist_ok=True)
    s.to_frame("Close").to_csv(path)
    return s


def build_task2_dataset(hourly, cutoff_hour=INTRADAY_CUTOFF_HOUR,
                        close_hour=INTRADAY_CLOSE_HOUR, daily_close=None):
    """สร้าง X/y รายวัน โดยไม่ใช้แท่งหลัง cutoff เป็น input.

    input (feature) ใช้แท่ง timestamp <= cutoff_hour เท่านั้น ถ้าไม่มีแท่งตรง
    cutoff_hour พอดี ใช้แท่งล่าสุดก่อนหน้านั้น และบันทึก cutoff_hour_used เป็น
    feature เพื่อให้โมเดลรับรู้ความต่างนี้

    target (actual_close): ถ้าใส่ daily_close (pd.Series, index=วันที่,
    ค่า=ราคาปิดทางการรายวันจาก yfinance interval=1d) มาจะใช้ค่านี้เป็น
    ราคาปิดทางการแทนแท่ง 1h ที่ Hour==close_hour — **แนะนำให้ใส่เสมอ**
    เพราะแท่ง 1h ตรวจแล้วว่าตรง ATC จริงแค่ ~40-46% เท่านั้น (ดู
    check_settrade_atc_match.py) การไม่ใส่ daily_close (None) ยังรองรับไว้
    เพื่อ backward-compat กับ test เดิม แต่ไม่ควรใช้ใน pipeline จริงอีกแล้ว
    """
    rows = []
    for date, day in hourly.groupby("Date", sort=True):
        day = day.sort_values("Datetime")
        available = day.loc[day["Hour"] <= cutoff_hour]
        if available.empty:
            continue

        if daily_close is not None:
            date_key = pd.Timestamp(date).normalize().tz_localize(None)
            if date_key not in daily_close.index:
                continue
            actual_close_val = float(daily_close.loc[date_key])
        else:
            closing = day.loc[day["Hour"] == close_hour]
            if closing.empty:
                continue
            actual_close_val = float(closing.iloc[-1]["Close"])

        cutoff = available.iloc[-1]
        observed = available.loc[available["Datetime"] <= cutoff["Datetime"]]
        open_price = float(observed.iloc[0]["Open"])
        cutoff_close = float(cutoff["Close"])
        high_so_far = float(observed["High"].max())
        low_so_far = float(observed["Low"].min())
        cum_volume = float(observed["Volume"].fillna(0).sum())
        cutoff_open = float(cutoff["Open"])

        rows.append({
            "Date": date,
            # ใช้เก็บเพื่อสร้าง target / export เท่านั้น ไม่เป็น feature ดิบ
            "cutoff_price": cutoff_close,
            "actual_close": actual_close_val,
            "daily_volume": float(day["Volume"].fillna(0).sum()),
            "session_open": open_price,
            "cutoff_hour_used": int(cutoff["Hour"]),
            # ทุก feature ต่อไปนี้รู้ได้ ณ เวลา cutoff
            "return_open_to_cutoff": cutoff_close / open_price - 1,
            "high_from_open": high_so_far / open_price - 1,
            "low_from_open": low_so_far / open_price - 1,
            "range_to_cutoff": (high_so_far - low_so_far) / cutoff_close,
            "cutoff_bar_return": cutoff_close / cutoff_open - 1,
            "cutoff_bar_range": (float(cutoff["High"]) - float(cutoff["Low"]))
                                / cutoff_close,
            "cumulative_volume": cum_volume,
            "day_of_week": date.dayofweek,
        })

    data = pd.DataFrame(rows).set_index("Date").sort_index()
    if data.empty:
        raise ValueError("สร้าง Task 2 dataset ไม่ได้: ตรวจ cutoff/close hours")

    # ค่าเหล่านี้ใช้ได้ ณ cutoff เพราะอ้างอิง "วันก่อนหน้า" ด้วย shift(1)
    data["prev_close"] = data["actual_close"].shift(1)
    data["prev_day_return"] = data["prev_close"] / data["prev_close"].shift(1) - 1
    data["overnight_gap"] = data["session_open"] / data["prev_close"] - 1
    expected_volume = data["daily_volume"].shift(1).rolling(20).mean()
    data["volume_progress_vs_20d"] = data["cumulative_volume"] / expected_volume

    # Target: ทิศทาง "จาก cutoff ไป close" ไม่ใช่ทิศทางทั้งวันจาก open
    data["y_up_to_close"] = (data["actual_close"] > data["cutoff_price"]).astype(int)

    feature_columns = [
        "cutoff_hour_used", "return_open_to_cutoff", "high_from_open",
        "low_from_open", "range_to_cutoff", "cutoff_bar_return",
        "cutoff_bar_range", "prev_day_return", "overnight_gap",
        "volume_progress_vs_20d", "day_of_week",
    ]
    X = data[feature_columns]
    y = data["y_up_to_close"]
    details = data[["cutoff_price", "actual_close", "cutoff_hour_used"]]
    return X, y, details


def _prepare(X, y, details):
    """จัด index และตัดเฉพาะแถวที่ missing มากเกินกว่าจะใช้ได้."""
    valid = y.notna()
    X, y, details = X.loc[valid], y.loc[valid], details.loc[valid]
    # rolling volume ช่วงต้นจะเป็น NaN หนึ่ง column; ปล่อยให้ median imputer จัดการ
    keep = X.isna().mean(axis=1) < 0.5
    return X.loc[keep], y.loc[keep], details.loc[keep]


def run_task2_for_ticker(ticker, dev=False):
    print("\n" + "=" * 78)
    print(f"Task 2 Intraday — {ticker}")
    print(f"ใช้ข้อมูลถึงแท่ง <= {INTRADAY_CUTOFF_HOUR}:00 "
          f"เพื่อทำนาย Close {INTRADAY_CLOSE_HOUR}:00")
    if dev:
        print("โหมด dev: ใช้ train/validation เท่านั้น ไม่แตะ test")
    print("=" * 78)

    hourly = load_hourly(ticker)
    daily_close = load_daily_close(ticker, hourly["Date"].min(), hourly["Date"].max())
    X, y, details = build_task2_dataset(hourly, daily_close=daily_close)
    X, y, details = _prepare(X, y, details)
    print(f"[task2] ได้ {len(X)} วัน, {X.shape[1]} intraday features "
          f"({X.index[0].date()} -> {X.index[-1].date()})")
    print(f"[task2] สัดส่วนขึ้นจาก cutoff ถึง close: {y.mean():.2%}")

    parts = chronological_split(X, y, verbose=True, name=f"Task 2 {ticker}")
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]
    X_test, y_test = parts["test"]

    models = get_classifiers()
    val_results, val_predictions, val_probabilities = {}, {}, {}
    test_results, test_predictions, test_probabilities = {}, {}, {}
    for name, model in models.items():
        print(f"[task2] เทรน {name} ...", end=" ", flush=True)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_results[name] = classification_metrics(y_val, val_pred, val_proba)
        val_predictions[name], val_probabilities[name] = val_pred, val_proba

        if dev:
            print(f"val Accuracy={val_results[name]['Accuracy']:.4f}")
        else:
            test_pred = model.predict(X_test)
            test_proba = model.predict_proba(X_test)[:, 1]
            test_results[name] = classification_metrics(y_test, test_pred, test_proba)
            test_predictions[name], test_probabilities[name] = test_pred, test_proba
            print(f"val Accuracy={val_results[name]['Accuracy']:.4f}, "
                  f"test Accuracy={test_results[name]['Accuracy']:.4f}")

    best = max(val_results, key=lambda name: val_results[name]["Accuracy"])
    split_name = "validation" if dev else "test"
    X_eval, y_eval = (X_val, y_val) if dev else (X_test, y_test)
    details_eval = details.loc[X_eval.index]
    results = val_results if dev else test_results
    predictions = val_predictions if dev else test_predictions
    probabilities = val_probabilities if dev else test_probabilities

    table = results_table(results, sort_by="Accuracy", ascending=False)
    print_table(table, f"Task 2: {ticker} — ผลบน {split_name}")
    print(f"[task2] เลือกจาก validation: {best} "
          f"(Accuracy={val_results[best]['Accuracy']:.4f})")
    print_confusion(y_eval, predictions[best], labels=("ลง/นิ่ง", "ขึ้น"),
                    title=f"Task 2 {ticker}: {best} ({split_name})")

    prediction_frame = pd.DataFrame({
        "Ticker": ticker,
        "Model": best,
        "Date": X_eval.index,
        "Cutoff_Hour_Used": details_eval["cutoff_hour_used"].to_numpy(),
        "Cutoff_Price": details_eval["cutoff_price"].to_numpy(),
        "Actual_Close": details_eval["actual_close"].to_numpy(),
        "Actual_Up_To_Close": y_eval.astype(int).to_numpy(),
        "Probability_Up_To_Close": probabilities[best],
        "Predicted_Up_To_Close": predictions[best].astype(int),
    })
    prediction_frame["Correct"] = (
        prediction_frame["Actual_Up_To_Close"]
        == prediction_frame["Predicted_Up_To_Close"]
    ).astype(int)
    return {"ticker": ticker, "table": table, "predictions": prediction_frame,
            "best_model": best, "split": split_name}


def save_outputs(results, dev):
    """บันทึกผลของ Task 2 แยกจาก Task 1 ด้วย timestamp เดียวกัน."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split = "validation" if dev else "test"
    metrics = []
    for result in results:
        frame = result["table"].reset_index(names="Model")
        frame.insert(0, "Ticker", result["ticker"])
        metrics.append(frame)
    metric_path = os.path.join(OUTPUT_DIR, f"task2_intraday_{split}_metrics_{stamp}.csv")
    prediction_path = os.path.join(
        OUTPUT_DIR, f"task2_intraday_{split}_predictions_{stamp}.csv"
    )
    pd.concat(metrics, ignore_index=True).to_csv(metric_path, index=False,
                                                   encoding="utf-8-sig")
    pd.concat([r["predictions"] for r in results], ignore_index=True).to_csv(
        prediction_path, index=False, encoding="utf-8-sig"
    )
    print(f"\n[task2] metrics: {metric_path}")
    print(f"[task2] predictions: {prediction_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 2: ทำนายทิศทาง Close 16:00 จากข้อมูล intraday ก่อน cutoff"
    )
    parser.add_argument("--dev", action="store_true",
                        help="ใช้ train/validation เท่านั้น; ไม่แตะ test")
    return parser.parse_args()


def main():
    args = parse_args()
    output = []
    for ticker in TICKERS:
        try:
            output.append(run_task2_for_ticker(ticker, dev=args.dev))
        except Exception as error:
            print(f"[task2] {ticker} รันไม่ผ่าน: {type(error).__name__}: {error}")
    if output:
        save_outputs(output, dev=args.dev)
    return output


if __name__ == "__main__":
    main()
