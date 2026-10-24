"""
features_experimental.py — feature groups + กลุ่มใหม่ cross_stock (feedback ข้อ 3)
=================================================================================

*** ไฟล์นี้ทำอะไร ***
1. จัด 29 feature เดิมจาก `features.build_features()` เป็น **7 กลุ่ม** เพื่อให้
   `optuna_features.py` เปิด/ปิดเป็นกลุ่มได้
2. เพิ่ม **กลุ่มใหม่กลุ่มเดียว** `cross_stock` = lag return 1/2/5 วัน + volatility 5 วัน
   ของ **หุ้นอีกตัว** (KBANK <-> ADVANC ซึ่งมีอยู่ใน `raw_data/` แล้ว)
3. `verify_no_leak_experimental()` — ตรวจ leak แบบเดียวกับ `features.verify_no_leak`

*** ทำไมต้องมี / ตอบ feedback ข้อไหน ***
feedback ข้อ 3 (14 ก.ย. 2026) ขอให้ใช้ HyperOpt หา feature ตอนนี้ `tune.py` เป็น
grid search hyperparameter อย่างเดียว feature fix 29 ตัวตายตัว ไม่มีกลไกเปิด/ปิด feature เลย

*** ห้ามแก้ features.py ***
ไฟล์นี้ **เรียก** `features.build_features()` ตัวเดิมแล้วเลือกคอลัมน์ ไม่คัดลอกสูตรใด ๆ
มาเขียนใหม่ -> ถ้า `features.py` เปลี่ยน ที่นี่เปลี่ยนตาม และ `GROUPS` จะ assert ว่า
ยังครบ 29 คอลัมน์เดิมเป๊ะ (ไม่เกิน ไม่ขาด)

*** กฎเหล็กเดียวกับ features.py ***
feature ของแถววันที่ t ต้องคำนวณจากข้อมูลตลาด "ถึงวันที่ t-1 เท่านั้น"
สำหรับ cross_stock: คำนวณ indicator จาก df ของหุ้นอีกตัวแล้ว `shift(1)` **บนปฏิทินของหุ้นนั้นเอง**
-> แถว t ได้ค่าของวันซื้อขายล่าสุดของหุ้นอีกตัวที่ "ก่อน t" เสมอ แล้วจึง align ด้วยวันที่
(inner join) กับหุ้นหลัก · ไม่มีข้อมูลวันที่ t ของหุ้นใดหลุดเข้ามา
"""

import numpy as np
import pandas as pd

from config import TICKERS, VOL_WINDOWS, SPLIT_BY_DATE
from data_loader import load_stock
from features import build_features, build_market_features

# ---------------------------------------------------------------
# 1) กลุ่มของ 29 feature เดิม
# ---------------------------------------------------------------
# ชื่อคอลัมน์ต้องตรงกับที่ features.build_features() คืนมาทุกตัวอักษร
# (ตรวจด้วย assert_groups_cover_base_features() -- เรียกอัตโนมัติใน build_grouped_features)
BASE_GROUPS = {
    "returns":    ["ret_1d_prev", "ret_2d_prev", "ret_3d_prev", "ret_5d_prev",
                   "ret_10d_prev"],
    "candle":     ["hl_range_prev", "oc_change_prev", "close_pos_in_range_prev"],
    "trend":      ["close_over_sma5_prev", "close_over_ema5_prev",
                   "close_over_sma10_prev", "close_over_ema10_prev",
                   "close_over_sma20_prev", "close_over_ema20_prev",
                   "sma5_over_sma20_prev"],
    "volatility": ["volatility_5d_prev", "volatility_20d_prev"],
    "momentum":   ["rsi_prev", "macd_prev", "macd_signal_prev", "macd_hist_prev",
                   "bb_position_prev"],
    "volume":     ["volume_change_prev", "volume_over_ma20_prev"],
    "calendar":   ["dow_0", "dow_1", "dow_2", "dow_3", "dow_4"],
}

# กลุ่มใหม่กลุ่มเดียวของรอบนี้ (feedback ข้อ 3 ข้อ 2)
CROSS_LAGS = [1, 2, 5]
CROSS_VOL_WINDOW = 5
CROSS_GROUP = "cross_stock"
CROSS_COLUMNS = ([f"x_ret_{lag}d_prev" for lag in CROSS_LAGS]
                 + [f"x_volatility_{CROSS_VOL_WINDOW}d_prev"])

GROUPS = {**BASE_GROUPS, CROSS_GROUP: CROSS_COLUMNS}
GROUP_NAMES = list(GROUPS)                      # ลำดับคงที่ -- ใช้เป็นชื่อ param ของ Optuna

# หุ้นคู่ของแต่ละตัว -- มีสองตัวใน TICKERS จึงจับคู่กันตรง ๆ
assert len(TICKERS) == 2, f"cross_stock ออกแบบไว้สำหรับ 2 หุ้น (TICKERS = {TICKERS})"
CROSS_PARTNER = {TICKERS[0]: TICKERS[1], TICKERS[1]: TICKERS[0]}

assert CROSS_VOL_WINDOW in VOL_WINDOWS, (
    f"cross_stock ใช้ volatility {CROSS_VOL_WINDOW}d ให้ตรงกับ config.VOL_WINDOWS "
    f"({VOL_WINDOWS}) เพื่อให้เทียบกับ volatility ของหุ้นหลักได้ตรง ๆ")


def assert_groups_cover_base_features(X_base):
    """
    BASE_GROUPS ต้องครอบคลุมคอลัมน์ของ features.build_features() **เป๊ะ** ไม่เกินไม่ขาด

    ถ้า features.py เพิ่ม/ลบ feature แล้วลืมแก้ที่นี่ ต้องพังทันที ไม่ใช่เงียบ ๆ
    ปล่อยให้ Optuna ค้นบน feature set ที่ไม่ใช่ 29 ตัวเดิม
    """
    listed = [c for cols in BASE_GROUPS.values() for c in cols]
    assert len(listed) == len(set(listed)), "BASE_GROUPS มีคอลัมน์ซ้ำ"
    missing = sorted(set(X_base.columns) - set(listed))
    extra = sorted(set(listed) - set(X_base.columns))
    assert not missing and not extra, (
        f"BASE_GROUPS ไม่ตรงกับ features.build_features()\n"
        f"  ไม่ได้จัดกลุ่ม: {missing}\n  ไม่มีในของจริง: {extra}")
    assert len(listed) == 29, f"ต้องมี 29 feature เดิม (ได้ {len(listed)})"
    return True


# ---------------------------------------------------------------
# 2) กลุ่มใหม่: cross_stock
# ---------------------------------------------------------------
def build_cross_stock_raw(partner_df):
    """
    indicator ของหุ้นอีกตัว **ก่อน** shift -- อย่าเอาไปเทรนตรง ๆ (มีข้อมูลวัน t อยู่)

    ใช้สูตรเดียวกับ `features.build_market_features` ทุกตัว (pct_change แบบ
    `fill_method=None` และ rolling std ของ return 1 วัน) เพื่อให้ตีความได้ว่าเป็น
    "feature เดิมแต่ของหุ้นอีกตัว" ไม่ใช่ตัวแปรใหม่ที่นิยามต่างกัน
    """
    close = partner_df["Close"]
    f = pd.DataFrame(index=partner_df.index)
    for lag in CROSS_LAGS:
        f[f"x_ret_{lag}d"] = close.pct_change(lag, fill_method=None)
    ret1 = close.pct_change(fill_method=None)
    f[f"x_volatility_{CROSS_VOL_WINDOW}d"] = ret1.rolling(CROSS_VOL_WINDOW).std()
    return f


def build_cross_stock_features(partner_df):
    """
    cross_stock features พร้อมใช้ = build_cross_stock_raw() แล้ว shift(1)

    shift(1) ทำบน **ปฏิทินของหุ้นอีกตัวเอง** -> แถววันที่ t ได้ค่าของวันซื้อขายล่าสุด
    ของหุ้นนั้นที่อยู่ก่อน t · ปลอดภัยแม้ปฏิทินสองหุ้นไม่ตรงกันทุกวัน เพราะ
    "วันซื้อขายก่อนหน้าของหุ้น X" ย่อมอยู่ก่อนวันที่ t เสมอ
    """
    out = build_cross_stock_raw(partner_df).shift(1)
    out.columns = [f"{c}_prev" for c in out.columns]
    assert list(out.columns) == CROSS_COLUMNS, (
        f"ชื่อคอลัมน์ cross_stock ไม่ตรง CROSS_COLUMNS: {list(out.columns)}")
    return out


# ---------------------------------------------------------------
# 3) ตัวสร้างหลัก
# ---------------------------------------------------------------
def build_grouped_features(df, partner_df=None, verbose=False, dev_only=True):
    """
    คืน (X_all, groups) โดย X_all มีทุกคอลัมน์ของทุกกลุ่ม (33 ตัว = 29 + cross 4)

    - 29 ตัวเดิมมาจาก `features.build_features(df)` ตรง ๆ (ไม่แก้ features.py)
    - cross_stock align ด้วยวันที่แบบ **inner join** -> เหลือเฉพาะวันที่มีข้อมูลทั้งสองหุ้น
      (`partner_df = None` -> ไม่มีกลุ่ม cross_stock ใน X_all เลย)

    groups = dict {ชื่อกลุ่ม: [คอลัมน์]} เฉพาะกลุ่มที่มีอยู่จริงใน X_all
    การเลือกกลุ่มทำทีหลังด้วย `select_groups()` -> X_all ถูกสร้างครั้งเดียวต่อ fold
    แล้วใช้ซ้ำได้ทุก trial (เร็วกว่าสร้างใหม่ทุกครั้ง และได้ค่าเดียวกันเป๊ะ)

    dev_only=True (ค่าเริ่มต้น) -- ตาข่ายนิรภัย: ปฏิเสธข้อมูลที่เลย
    `SPLIT_BY_DATE["val_end"]` ผู้เรียกต้องตัดมาก่อนเสมอ (เหมือน `tune.load_dev_data`)
    ตั้งเป็น False ได้ แต่ต้องตั้งใจพิมพ์เอง เหมือนที่ `main.py` บังคับให้พิมพ์
    `--open-test` -- กันกรณีมีคนเขียนตัวเรียกใหม่แล้วลืมตัดข้อมูล
    """
    if dev_only:
        val_end = pd.Timestamp(SPLIT_BY_DATE["val_end"])
        for name, frame in (("df", df), ("partner_df", partner_df)):
            if frame is not None and frame.index[-1] > val_end:
                raise ValueError(
                    f"build_grouped_features: {name} มีแถวถึง "
                    f"{frame.index[-1].date()} ซึ่งเลย val_end ({val_end.date()})\n"
                    "ต้องตัดข้อมูลก่อนเรียก "
                    "(ดู optuna_features.load_dev_data_grouped)\n"
                    "ถ้าตั้งใจจะใช้ข้อมูลเลย val_end จริง ต้องส่ง dev_only=False เอง")
    X_base = build_features(df, verbose=verbose)
    assert_groups_cover_base_features(X_base)

    if partner_df is None:
        return X_base, dict(BASE_GROUPS)

    cross = build_cross_stock_features(partner_df)
    n_before = len(X_base)
    X_all = X_base.join(cross, how="inner")      # inner join ตามวันที่
    if verbose:
        print(f"[featx] cross_stock: inner join เหลือ {len(X_all)}/{n_before} แถว "
              f"(+{len(CROSS_COLUMNS)} คอลัมน์ รวม {X_all.shape[1]} ตัว)")
    assert X_all.index.is_monotonic_increasing
    assert list(X_all.columns) == list(X_base.columns) + CROSS_COLUMNS
    return X_all, dict(GROUPS)


def select_groups(X_all, active_groups, groups=None):
    """
    เลือกคอลัมน์ของกลุ่มที่เปิด -- เรียงตามลำดับ GROUP_NAMES เสมอ

    ลำดับคอลัมน์คงที่สำคัญกับ reproducibility: RandomForest/XGBoost ที่สุ่ม feature
    subset (max_features / colsample_bytree) จะได้ชุด feature ต่างกันถ้าลำดับคอลัมน์
    ต่างกัน แม้เซ็ตเดียวกัน -> trial เดิมจะให้ผลไม่เหมือนเดิม
    """
    groups = groups or GROUPS
    active = [g for g in GROUP_NAMES if g in active_groups and g in groups]
    if not active:
        raise ValueError("ต้องเปิดอย่างน้อย 1 กลุ่ม (active_groups ว่าง)")
    cols = [c for g in active for c in groups[g]]
    assert len(cols) == len(set(cols)), "คอลัมน์ซ้ำระหว่างกลุ่ม"
    return X_all[cols]


def load_pair(ticker, verbose=False):
    """
    โหลดหุ้นหลัก + หุ้นคู่ของมัน -- ผู้เรียกเป็นคนตัดวันที่เอง (ต้องตัดก่อนสร้าง feature)

    *** ไม่ตัดวันที่ให้ในฟังก์ชันนี้โดยตั้งใจ *** เพื่อไม่ให้มีสองที่ที่ตัดข้อมูล
    ซึ่งจะทำให้ลืมได้ว่าที่ไหนตัดแล้วที่ไหนยัง -- ดู `optuna_features.load_dev_data_grouped`
    """
    partner = CROSS_PARTNER[ticker]
    return load_stock(ticker, verbose=verbose), load_stock(partner, verbose=verbose)


# ---------------------------------------------------------------
# 4) ตรวจ leak
# ---------------------------------------------------------------
def verify_no_leak_experimental(df, partner_df, X_all, sample_idx=100):
    """
    ตรวจ leak ของ cross_stock แบบเดียวกับ `features.verify_no_leak`

    3 เรื่อง:
      1. คอลัมน์ cross แถววันที่ d ต้องเท่ากับ raw indicator ของหุ้นคู่
         ณ **วันซื้อขายล่าสุดของหุ้นคู่ที่อยู่ก่อน d** (ไม่ใช่วันที่ d เอง)
      2. ไม่มีคอลัมน์ cross ใดเท่ากับ raw ของวันที่ d เอง (ซึ่งจะเป็น leak)
      3. ตรวจทุกแถว ไม่ใช่แค่แถวตัวอย่าง (เหมือน test_market_feature_alignment)
    """
    raw = build_cross_stock_raw(partner_df)
    shifted = build_cross_stock_features(partner_df)

    # (1) ทุกแถว: cross ใน X_all ต้องเท่ากับ shifted ที่วันเดียวกัน
    common = X_all.index.intersection(shifted.index)
    assert len(common) == len(X_all), "X_all มีวันที่ที่ไม่มีใน cross features"
    for col in CROSS_COLUMNS:
        got = X_all.loc[common, col].to_numpy()
        want = shifted.loc[common, col].to_numpy()
        assert np.allclose(got, want, equal_nan=True), f"LEAK/ALIGN! {col} ไม่ตรง"

    # (2) แถวตัวอย่าง: ต้องเท่ากับ raw ของวันซื้อขายก่อนหน้าของหุ้นคู่ ไม่ใช่วันเดียวกัน
    d = X_all.index[sample_idx]
    pos = partner_df.index.get_indexer([d])[0]
    assert pos > 0, "sample_idx ชี้ไปวันแรกของหุ้นคู่ -- เลือก sample_idx ใหญ่กว่านี้"
    checked = 0
    for col, raw_col in zip(CROSS_COLUMNS, raw.columns):
        a, b = X_all.loc[d, col], raw[raw_col].iloc[pos - 1]
        same_day = raw[raw_col].iloc[pos]
        if pd.isna(a) and pd.isna(b):
            continue
        assert np.isclose(a, b), (
            f"LEAK! {col} ที่ {d.date()} = {a} แต่ raw ของวันก่อนหน้า = {b}")
        if not pd.isna(same_day) and not np.isclose(b, same_day):
            assert not np.isclose(a, same_day), (
                f"LEAK! {col} ที่ {d.date()} เท่ากับ raw ของวันเดียวกัน")
        checked += 1
    assert checked > 0, "ไม่ได้ตรวจคอลัมน์ cross เลยสักตัว"

    # (3) ของหุ้นหลักเองยังต้องผ่านกฎเดิม -- เรียกฟังก์ชันเดิมไม่คัดลอก logic
    #
    # *** ห้ามส่ง X_all เข้า verify_no_leak ตรง ๆ ***
    # verify_no_leak เทียบ features.iloc[i] กับ raw ของ df ที่ .iloc[i-1] แบบ
    # **ตามตำแหน่ง** แต่ X_all ผ่าน inner join มาแล้ว ตำแหน่งจะตรงกับ df เฉพาะตอนที่
    # ปฏิทินของสองหุ้นเหมือนกันทุกวัน (ตอนนี้เหมือนกัน แต่ไม่รับประกันในอนาคต)
    # ถ้าปฏิทินต่างกันแม้วันเดียว การตรวจจะ "ไม่ผ่าน" ด้วยเหตุผลที่ผิด (false alarm)
    # จึงส่ง X_base ซึ่ง index ตรงกับ df เป๊ะเข้าไปแทน แล้วตรวจแยกว่าคอลัมน์ฐานใน
    # X_all เท่ากับ X_base ทุกค่าบนวันที่ร่วมกัน -> ครอบคลุมเท่าเดิมโดยไม่มี false alarm
    from features import verify_no_leak
    X_base = build_features(df, verbose=False)
    base_cols = [c for c in X_all.columns if c not in CROSS_COLUMNS]
    assert list(X_base.columns) == base_cols, "คอลัมน์ฐานของ X_all ไม่ตรง features.py"
    for col in base_cols:
        a = X_all[col].to_numpy()
        b = X_base.loc[X_all.index, col].to_numpy()
        assert np.allclose(a, b, equal_nan=True), (
            f"คอลัมน์ฐาน {col} ใน X_all ไม่ตรงกับ features.build_features()")
    verify_no_leak(df, X_base, sample_idx=sample_idx)

    print(f"[featx] verify_no_leak_experimental ผ่าน "
          f"({checked} cross cols x {len(X_all)} แถว + กฎเดิมของ features.py)")
    return True


def main():
    """ตรวจเร็ว ๆ ว่าไฟล์นี้ทำงานได้และผ่าน leak test (ไม่เซฟอะไร)"""
    from config import SPLIT_BY_DATE
    val_end = pd.Timestamp(SPLIT_BY_DATE["val_end"])
    for ticker in TICKERS:
        df, partner = load_pair(ticker, verbose=False)
        df, partner = df.loc[:val_end], partner.loc[:val_end]
        X_all, groups = build_grouped_features(df, partner, verbose=True)
        assert X_all.index[-1] <= val_end
        print(f"[featx] {ticker} (คู่ {CROSS_PARTNER[ticker]}): {X_all.shape} · "
              f"{len(groups)} กลุ่ม = "
              + ", ".join(f"{g}({len(c)})" for g, c in groups.items()))
        verify_no_leak_experimental(df, partner, X_all)
        for g in GROUP_NAMES:
            sel = select_groups(X_all, [g])
            assert list(sel.columns) == groups[g]
        print(f"[featx] {ticker}: select_groups ครบทุกกลุ่ม")
        # ใช้ unused import ให้เป็นประโยชน์: ยืนยันว่า raw ของหุ้นหลักยังมี 24 คอลัมน์
        assert build_market_features(df).shape[1] == 24


if __name__ == "__main__":
    main()
