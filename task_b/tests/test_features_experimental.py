"""
tests/test_features_experimental.py — ตรวจกลุ่ม feature ใหม่ (feedback ข้อ 3)

รันได้ 2 แบบ:
    python tests/test_features_experimental.py
    python -m pytest tests/

ใช้ข้อมูลถึง VAL_END เท่านั้น -- ไม่มี test rows เข้ามาเลย (เหมือน test_pipeline.py)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SPLIT_BY_DATE, TICKERS
from features import build_features
from features_experimental import (
    BASE_GROUPS, GROUPS, GROUP_NAMES, CROSS_COLUMNS, CROSS_GROUP, CROSS_LAGS,
    CROSS_PARTNER, CROSS_VOL_WINDOW, assert_groups_cover_base_features,
    build_cross_stock_features, build_cross_stock_raw, build_grouped_features,
    load_pair, select_groups, verify_no_leak_experimental,
)

VAL_END = pd.Timestamp(SPLIT_BY_DATE["val_end"])


def _pair(ticker=TICKERS[0]):
    """หุ้นหลัก + หุ้นคู่ ตัดที่ val_end ทั้งคู่ (ตัดก่อนสร้าง feature)"""
    df, partner = load_pair(ticker, verbose=False)
    return df.loc[:VAL_END].copy(), partner.loc[:VAL_END].copy()


# ---------------------------------------------------------------
# โครงสร้างกลุ่ม
# ---------------------------------------------------------------
def test_groups_cover_base_features_exactly():
    """BASE_GROUPS ต้องตรงกับ features.build_features() เป๊ะ -- ไม่เกิน ไม่ขาด"""
    for ticker in TICKERS:
        df, _ = _pair(ticker)
        X_base = build_features(df, verbose=False)
        assert assert_groups_cover_base_features(X_base)
        assert X_base.shape[1] == 29
    listed = [c for cols in BASE_GROUPS.values() for c in cols]
    assert len(listed) == 29 and len(set(listed)) == 29
    assert len(BASE_GROUPS) == 7, "29 feature เดิมต้องแบ่งเป็น 7 กลุ่ม"
    assert len(GROUPS) == 8 and GROUP_NAMES[-1] == CROSS_GROUP


def test_select_groups_order_is_stable():
    """ลำดับคอลัมน์ต้องคงที่ ไม่ขึ้นกับลำดับที่ส่ง active_groups เข้ามา"""
    df, partner = _pair()
    X_all, groups = build_grouped_features(df, partner)
    a = select_groups(X_all, ["momentum", "returns", CROSS_GROUP], groups)
    b = select_groups(X_all, [CROSS_GROUP, "momentum", "returns"], groups)
    assert list(a.columns) == list(b.columns)
    # เรียงตาม GROUP_NAMES: returns -> momentum -> cross_stock
    assert list(a.columns) == (groups["returns"] + groups["momentum"]
                               + groups[CROSS_GROUP])


def test_select_groups_requires_at_least_one():
    df, partner = _pair()
    X_all, groups = build_grouped_features(df, partner)
    try:
        select_groups(X_all, [], groups)
    except ValueError:
        return
    raise AssertionError("active_groups ว่างต้อง raise ValueError")


def test_all_groups_on_reproduces_base_plus_cross():
    """เปิดทุกกลุ่ม = 29 เดิม + 4 cross และค่าของ 29 ตัวเดิมต้องเท่าเดิมทุกบิต"""
    df, partner = _pair()
    X_base = build_features(df, verbose=False)
    X_all, groups = build_grouped_features(df, partner)
    full = select_groups(X_all, GROUP_NAMES, groups)
    assert full.shape[1] == 33
    common = X_base.index.intersection(full.index)
    for col in X_base.columns:
        assert np.allclose(full.loc[common, col].to_numpy(),
                           X_base.loc[common, col].to_numpy(),
                           equal_nan=True), f"{col} เปลี่ยนค่าไปจาก features.py"


def test_no_partner_means_no_cross_group():
    df, _ = _pair()
    X_all, groups = build_grouped_features(df, None)
    assert CROSS_GROUP not in groups
    assert not any(c in X_all.columns for c in CROSS_COLUMNS)
    assert X_all.shape[1] == 29


# ---------------------------------------------------------------
# cross_stock: leak / alignment
# ---------------------------------------------------------------
def test_cross_features_are_shifted_by_one_partner_day():
    """ทุกแถว: cross แถว t ต้องเท่ากับ raw ของหุ้นคู่แถว t-1 (ปฏิทินของหุ้นคู่เอง)"""
    for ticker in TICKERS:
        _, partner = _pair(ticker)
        raw = build_cross_stock_raw(partner)
        shifted = build_cross_stock_features(partner)
        assert list(shifted.columns) == CROSS_COLUMNS
        for raw_col, col in zip(raw.columns, CROSS_COLUMNS):
            got = shifted[col].iloc[1:].to_numpy()
            want = raw[raw_col].iloc[:-1].to_numpy()
            assert np.allclose(got, want, equal_nan=True), f"{ticker} {col} ไม่ตรง t-1"
        assert shifted.iloc[0].isna().all(), "แถวแรกของ cross ต้องเป็น NaN ทั้งแถว"


def test_cross_features_never_use_same_day_partner_data():
    """
    cross แถววันที่ d ต้องไม่เท่ากับ raw ของหุ้นคู่วันที่ d เอง

    ตรวจทุกแถวที่ค่าวันก่อนหน้ากับวันเดียวกันต่างกันจริง (ถ้าสองค่าเท่ากันเองโดย
    ธรรมชาติ การทดสอบจะไม่มีความหมาย จึงข้ามแถวนั้น)
    """
    for ticker in TICKERS:
        _, partner = _pair(ticker)
        raw = build_cross_stock_raw(partner)
        shifted = build_cross_stock_features(partner)
        n_checked = 0
        for raw_col, col in zip(raw.columns, CROSS_COLUMNS):
            same = raw[raw_col].to_numpy()
            prev = shifted[col].to_numpy()
            ok = ~np.isnan(same) & ~np.isnan(prev) & ~np.isclose(same, prev)
            assert ok.sum() > 100, f"{ticker} {col}: แถวที่เทียบได้น้อยเกินไป"
            n_checked += int(ok.sum())
        assert n_checked > 1000


def test_verify_no_leak_experimental_passes():
    for ticker in TICKERS:
        df, partner = _pair(ticker)
        X_all, _ = build_grouped_features(df, partner)
        assert verify_no_leak_experimental(df, partner, X_all, sample_idx=100)


def test_cross_leak_is_detected_when_shift_removed():
    """
    negative control: ถ้าลืม shift(1) ตัวตรวจต้องจับได้

    ถ้า test นี้ผ่านโดยที่ตัวตรวจไม่ raise แปลว่าตัวตรวจไร้ประโยชน์
    """
    df, partner = _pair()
    X_base = build_features(df, verbose=False)
    leaky = build_cross_stock_raw(partner)          # ไม่ shift = leak
    leaky.columns = CROSS_COLUMNS
    X_leaky = X_base.join(leaky, how="inner")
    try:
        verify_no_leak_experimental(df, partner, X_leaky, sample_idx=100)
    except AssertionError:
        return
    raise AssertionError("ตัวตรวจ leak ต้องจับกรณีไม่ shift ได้ แต่ไม่จับ")


def test_inner_join_keeps_only_shared_dates():
    """ถ้าหุ้นคู่ขาดวันไป แถวนั้นต้องหายจาก X_all (inner join)"""
    df, partner = _pair()
    drop = partner.index[500]
    partner_gap = partner.drop(index=drop)
    X_all, _ = build_grouped_features(df, partner_gap)
    assert drop not in X_all.index
    X_full, _ = build_grouped_features(df, partner)
    assert len(X_all) == len(X_full) - 1
    # วันถัดไปยังอยู่ และใช้ค่าของวันซื้อขายล่าสุดที่เหลือ -> ยังไม่ใช่ข้อมูลอนาคต
    nxt = partner.index[501]
    assert nxt in X_all.index


def test_cross_columns_match_config_windows():
    """ชื่อคอลัมน์ต้องสอดคล้องกับ CROSS_LAGS / CROSS_VOL_WINDOW (กันชื่อหลุด)"""
    want = [f"x_ret_{lag}d_prev" for lag in CROSS_LAGS] + \
           [f"x_volatility_{CROSS_VOL_WINDOW}d_prev"]
    assert CROSS_COLUMNS == want
    assert GROUPS[CROSS_GROUP] == want


def test_partner_mapping_is_symmetric():
    assert set(CROSS_PARTNER) == set(TICKERS)
    for a, b in CROSS_PARTNER.items():
        assert a != b and CROSS_PARTNER[b] == a


def test_no_rows_after_val_end():
    """ทุกเส้นทางในไฟล์นี้ต้องไม่เห็นแถวหลัง val_end (ผู้เรียกตัดมาแล้ว)"""
    for ticker in TICKERS:
        df, partner = _pair(ticker)
        X_all, _ = build_grouped_features(df, partner)
        assert X_all.index[-1] <= VAL_END
        assert partner.index[-1] <= VAL_END


def test_nan_only_in_warmup_for_cross():
    """NaN ของ cross ต้องอยู่ต้นชุดต่อเนื่องเท่านั้น ไม่มีกลางชุด"""
    for ticker in TICKERS:
        df, partner = _pair(ticker)
        X_all, groups = build_grouped_features(df, partner)
        sub = X_all[groups[CROSS_GROUP]]
        pos = np.where((~sub.notna().all(axis=1)).to_numpy())[0]
        assert len(pos) > 0, "ต้องมี warm-up NaN อย่างน้อยช่วงต้น"
        assert (pos == np.arange(len(pos))).all(), f"{ticker}: cross มี NaN กลางชุด"
        assert np.isfinite(sub.iloc[len(pos):].to_numpy(dtype=float)).all()


def test_leak_check_survives_calendar_divergence():
    """
    ตัวตรวจ leak ต้องไม่ false-alarm ถ้าปฏิทินสองหุ้นไม่ตรงกัน

    `features.verify_no_leak` เทียบ feature แถว i กับ raw แถว i-1 **ตามตำแหน่ง**
    ถ้าส่ง X_all (ที่ผ่าน inner join แล้ว) เข้าไปตรง ๆ ตำแหน่งจะเลื่อนทันทีที่
    หุ้นคู่ขาดวันไป -> การตรวจจะล้มด้วยเหตุผลที่ผิด ทั้งที่ข้อมูลถูกต้อง
    (เคสนี้ยังไม่เกิดจริง เพราะ KBANK กับ ADVANC มีปฏิทินตรงกันทุกวัน
     แต่ต้องไม่พังเงียบ ๆ วันที่มันเกิด)
    """
    df, partner = _pair()
    partner_gap = partner.drop(index=partner.index[300:303])   # หุ้นคู่ขาด 3 วัน
    X_all, _ = build_grouped_features(df, partner_gap)
    assert len(X_all) == len(build_features(df, verbose=False)) - 3
    # ต้องผ่าน ไม่ใช่ raise
    assert verify_no_leak_experimental(df, partner_gap, X_all, sample_idx=400)


def test_dev_only_guard_rejects_rows_past_val_end():
    """ตาข่ายนิรภัย: ถ้าผู้เรียกลืมตัดข้อมูล ต้อง raise ไม่ใช่สร้าง feature เงียบ ๆ"""
    df_full, partner_full = load_pair(TICKERS[0], verbose=False)
    assert df_full.index[-1] > VAL_END, "ข้อมูลดิบต้องมีแถวเลย val_end จริง"

    for args in ((df_full, partner_full.loc[:VAL_END]),
                 (df_full.loc[:VAL_END], partner_full),
                 (df_full, partner_full)):
        try:
            build_grouped_features(*args)
        except ValueError as e:
            assert "val_end" in str(e)
            continue
        raise AssertionError(f"ควร raise ValueError แต่ไม่ raise ({args[0].index[-1]})")

    # ตัดมาแล้วต้องผ่าน และ dev_only=False ต้องยอมให้ผ่านแม้ไม่ตัด
    build_grouped_features(df_full.loc[:VAL_END], partner_full.loc[:VAL_END])
    X_all, _ = build_grouped_features(df_full, partner_full, dev_only=False)
    assert X_all.index[-1] > VAL_END


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
