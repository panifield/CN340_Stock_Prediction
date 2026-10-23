"""
tests/test_settrade_features.py — feature จากแท่ง 1 นาทีของ Settrade + leak test (2026-10-24)

leak test: "วางยา" แท่งตั้งแต่ 16:00 (ราคา/volume เพี้ยนมาก) -> feature ต้องเหมือนเดิมเป๊ะ
control  : วางยาแท่ง 15:59 (ควรเห็น) -> feature ต้องเปลี่ยน (ยืนยันว่าการวางยาไวพอ)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import settrade_features as SF  # noqa: E402

DAY = "2026-10-22"


def _bars(closes_before=None):
    """แท่ง 1 นาที 15:00-16:35 ของวันเดียว ราคาเด้ง 242/243"""
    times = list(pd.date_range(f"{DAY} 15:00", f"{DAY} 16:29", freq="1min", tz="Asia/Bangkok"))
    times.append(pd.Timestamp(f"{DAY} 16:35", tz="Asia/Bangkok"))
    close = np.where(np.arange(len(times)) % 2 == 0, 242.0, 243.0)
    if closes_before is not None:
        close[:len(closes_before)] = closes_before
    return pd.DataFrame({"open": close, "high": 243.0, "low": 242.0, "close": close,
                         "volume": 1000.0}, index=pd.DatetimeIndex(times, name="dt"))


def _poison(bars, mask):
    p = bars.copy()
    p.loc[mask, ["open", "high", "low", "close"]] = 300.0
    p.loc[mask, "volume"] = 9e9
    return p


def test_features_use_only_bars_before_1600():
    bars = _bars()
    after = bars.index >= pd.Timestamp(f"{DAY} 16:00", tz="Asia/Bangkok")
    assert SF.day_features(bars) == SF.day_features(_poison(bars, after))


def test_control_poisoning_1559_changes_features():
    bars = _bars()
    last = bars.index == pd.Timestamp(f"{DAY} 15:59", tz="Asia/Bangkok")
    assert SF.day_features(bars) != SF.day_features(_poison(bars, last))


def test_feature_values():
    bars = _bars()
    f = SF.day_features(bars)
    assert f["last_bar_time"] == "15:59"
    assert f["price_1559"] == 243.0                 # แท่งที่ 59 (คี่) = 243
    assert f["last_tick_dir"] == 1 and f["last_change_dir"] == 1
    assert f["dticks_5m"] == 1                      # 15:54 = 242 -> 15:59 = 243
    assert f["close_at_bar_high"] == 1 and f["close_at_bar_low"] == 0


def test_flat_last_bar_gives_zero_tick_dir_but_keeps_last_change():
    closes = np.array([242.0] * 58 + [243.0, 243.0])     # 15:58 ขึ้น, 15:59 เท่าเดิม
    f = SF.day_features(_bars(closes))
    assert f["last_tick_dir"] == 0 and f["last_change_dir"] == 1


def test_real_files_leak_test():
    """ไฟล์จริงทุกวัน: วางยาแท่งตั้งแต่ 16:00 แล้ว feature ไม่เปลี่ยน"""
    for symbol in ("KBANK", "ADVANC"):
        bars = SF.load_1m(symbol)
        if bars.empty:
            continue
        for day, g in bars.groupby(bars.index.date):
            cut = pd.Timestamp(day).tz_localize("Asia/Bangkok") + pd.Timedelta("16:00:00")
            a = SF.day_features(g)
            b = SF.day_features(_poison(g, g.index >= cut))
            assert a == b, f"{symbol} {day} feature เปลี่ยนเมื่อวางยาแท่งหลัง 16:00"


def test_load_1m_from_folder():
    with tempfile.TemporaryDirectory() as d:
        b = _bars().reset_index()
        b["time"] = 0
        b.to_csv(Path(d) / f"KBANK_1m_{DAY}.csv", index=False)
        out = SF.build_features("KBANK", log_dir=d)
        assert list(out.index) == [pd.Timestamp(DAY)]
