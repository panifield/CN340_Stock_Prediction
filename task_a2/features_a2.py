"""
features_a2.py — Task A2
=========================
สร้าง feature + target สำหรับงาน A2 (ทำนายคู่/คี่ของราคาปิดวัน t
ณ เวลา 16:00 ของวัน t) จากข้อมูล 1h ของ yfinance (722 วัน)

*** กฎเหล็ก: ใช้ข้อมูลไม่เกิน 16:00 ของวัน t เท่านั้น ***
แท่ง 1h เป็น bar-start (ยืนยันแล้ว: แท่งแรกของวัน = 10:00 ตรงเวลาเปิด
ตลาด) แท่ง label "15:00" ครอบคลุม [15:00,16:00) เต็ม ไม่ชนพักเที่ยง
-> close ของแท่งนี้ = ราคา ณ 16:00 เป๊ะ ห้ามใช้แท่ง label "16:00"
(ครอบคลุม [16:00,~16:30) เป็นข้อมูลหลัง 16:00)

target: y_flip_1600 = parity(close รายวัน) XOR parity(ราคา ณ 16:00)
        reconstruct: parity_close_hat = (parity_1600 + flip_hat) mod 2
        (โครงสร้างเดียวกับ y_flip ของ Task A เป๊ะ แค่เปลี่ยน "ฐาน" จาก
        ราคาปิดเมื่อวาน เป็นราคา ณ 16:00 ของวันนี้เอง)
"""

import os
import sys

import numpy as np
import pandas as pd

TASK_A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "task_a")
sys.path.insert(0, TASK_A)
from rounding import to_int_baht, parity, tick_size  # noqa: E402
import features as task_a_features  # noqa: E402
from data_loader import load_stock  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BAR_TIMES_BEFORE_1600 = ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00"]


def load_1h(ticker_safe):
    path = os.path.join(DATA_DIR, f"{ticker_safe}_1h_730d.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def _dist_to_round_boundary_ticks(price, tick):
    """
    ระยะ (หน่วย tick) จากราคาไปยังจุดปัดที่ใกล้ที่สุด (x.50 บาท พอดี
    ตามกฎอาจารย์ที่ threshold=0.5) ยิ่งใกล้ 0 ยิ่งเสี่ยงพลิก parity ง่าย
    """
    shifted = price - 0.5
    nearest = np.round(shifted)
    dist_baht = np.abs(shifted - nearest)
    return dist_baht / tick


def build_intraday_features(ticker, ticker_safe):
    """
    คืน DataFrame index=date ของ feature ทั้งหมดที่มาจากข้อมูล intraday
    วันเดียวกัน (ไม่เกิน 16:00) — ไม่ใช่ daily feature จาก Task A
    """
    df_1h = load_1h(ticker_safe)
    df_1h = df_1h.sort_index()

    rows = {}
    for date, g in df_1h.groupby(df_1h.index.date):
        g = g.set_index(g.index.time)
        times_present = [t for t in BAR_TIMES_BEFORE_1600
                         if pd.Timestamp(t).time() in g.index]
        if len(times_present) < 4:
            continue  # วันข้อมูลไม่ครบ ข้าม (holiday บางส่วน/half-day)

        closes = {t: g.loc[pd.Timestamp(t).time(), "Close"] for t in times_present}
        p1600 = closes[times_present[-1]] if times_present[-1] == "15:00" \
            else None
        if "15:00" not in closes:
            continue  # ไม่มีแท่งที่ตรงกับ "ราคา ณ 16:00" ข้ามวันนี้
        p1600 = closes["15:00"]

        tick = tick_size(pd.Series([p1600])).iloc[0]
        n1600 = round(p1600 / tick)

        satang = round(p1600 * 100) % 100
        if satang == 0:
            dec_flag = 0
        elif satang == 25:
            dec_flag = 1
        elif satang == 50:
            dec_flag = 2
        elif satang == 75:
            dec_flag = 3
        else:
            dec_flag = 4

        dist_boundary_ticks = _dist_to_round_boundary_ticks(
            np.array([p1600]), np.array([tick]))[0]

        # momentum ของแท่งก่อน 16:00 (return ระหว่างแท่งติดกัน)
        ordered = [closes[t] for t in times_present]
        rets = np.diff(ordered) / np.array(ordered[:-1])
        ret_last1 = rets[-1] if len(rets) >= 1 else np.nan
        ret_last2 = rets[-2] if len(rets) >= 2 else np.nan
        ret_sum_all = (ordered[-1] - ordered[0]) / ordered[0] if len(ordered) > 1 else np.nan

        day_high = g.loc[[pd.Timestamp(t).time() for t in times_present], "High"].max()
        day_low = g.loc[[pd.Timestamp(t).time() for t in times_present], "Low"].min()
        range_ticks_so_far = (day_high - day_low) / tick

        vol_so_far = g.loc[[pd.Timestamp(t).time() for t in times_present], "Volume"].sum()

        rows[date] = {
            "price_1600": p1600,
            "n_mod2_1600": n1600 % 2,
            "dec_flag_1600": dec_flag,
            "tick_regime_1600": tick,
            "dist_to_boundary_1600": dist_boundary_ticks,
            "ret_last1h": ret_last1,
            "ret_last2h": ret_last2,
            "ret_since_open": ret_sum_all,
            "range_ticks_so_far": range_ticks_so_far,
            "vol_so_far": vol_so_far,
            "n_bars_available": len(times_present),
        }

    out = pd.DataFrame(rows).T
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out


def build_daily_context_features(ticker):
    """feature รายวันเดิมจาก Task A (ข้อมูลถึงวัน t-1 เท่านั้น อยู่แล้ว
    โดยธรรมชาติของ features.py — ไม่มี leak เพิ่มเข้ามาจากตรงนี้)"""
    df = load_stock(ticker, verbose=False)
    X = task_a_features.build_features(df, verbose=False)
    return X


def build_dataset(ticker, ticker_safe):
    """
    รวม intraday feature (วันเดียวกัน ไม่เกิน 16:00) + daily context
    feature (จาก Task A ถึงวัน t-1) + target (y_flip_1600, y_parity,
    prev_parity=parity ณ 16:00) เป็นตารางเดียว
    """
    intraday = build_intraday_features(ticker, ticker_safe)
    daily_ctx = build_daily_context_features(ticker)

    close_daily = load_stock(ticker, verbose=False)["Close"]
    close_daily.index = pd.to_datetime(close_daily.index)

    common = intraday.index.intersection(close_daily.index).intersection(daily_ctx.index)
    intraday = intraday.loc[common]
    daily_ctx = daily_ctx.loc[common]
    close_t = close_daily.loc[common]

    y_parity = parity(to_int_baht(close_t))
    parity_1600 = parity(to_int_baht(intraday["price_1600"]))
    y_flip = (y_parity.values + parity_1600.values) % 2

    X = pd.concat([intraday.drop(columns=["price_1600"]), daily_ctx], axis=1)
    # vol_ratio_prev (สืบทอดจาก Task A) เป็น log(volume/MA) ถ้า volume=0
    # บางวันจะได้ -inf ซึ่ง imputer จัดการไม่ได้ (ต่างจาก NaN) แปลงเป็น
    # NaN ก่อน ให้ imputer (median) จัดการเหมือน missing value ทั่วไป
    X = X.replace([np.inf, -np.inf], np.nan)

    targets = pd.DataFrame({
        "y_parity": y_parity.values,
        "parity_1600": parity_1600.values,
        "y_flip": y_flip,
        "price_1600": intraday["price_1600"].values,
        "daily_close": close_t.values,
    }, index=common)

    return X, targets
