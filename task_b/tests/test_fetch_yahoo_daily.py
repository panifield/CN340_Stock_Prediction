"""
tests/test_fetch_yahoo_daily.py — tools/fetch_yahoo_daily.py (ตัวเชื่อม Yahoo -> รูปแบบ Investing)

    python tests/test_fetch_yahoo_daily.py

ไม่ใช้อินเทอร์เน็ต: downloader ปลอมสร้างจาก raw_data/ ของจริง · เขียนลง temp dir เท่านั้น
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import append_investing as A        # noqa: E402
import fetch_yahoo_daily as F       # noqa: E402

N_CUT = 5


def _raw(t):
    return (ROOT / "raw_data" / f"{t}_10Y_Cleaned.csv").read_bytes()


def _raw_dates():
    """วันที่ทั้งหมดที่มีอยู่ใน raw_data/ ตอนนี้ (ไม่ hardcode) -- raw_data/ โตขึ้นทุกวันจาก
    daily_next_day.py จริง ดังนั้นวันที่ตายตัวในไฟล์นี้จะพังทุกครั้งที่มีการ append ข้อมูลใหม่
    (เจอบั๊กนี้จริง 2026-09-29 หลัง raw_data ขยับจาก 25 -> 28 ก.ย. และอีกครั้ง 2026-10-13
    หลังขยับถึง 5 ต.ค. -- รอบนั้นเหลือวันตายตัวค้างอยู่ในตัว test เอง ไม่ใช่แค่ NOW)
    ทุก test ด้านล่างจึงต้องอ้างวันที่ผ่าน DATES[...] เท่านั้น ห้ามพิมพ์วันที่ลงไปตรง ๆ"""
    _, old = A.read_canonical(ROOT / "raw_data" / "KBANK_10Y_Cleaned.csv")
    return old.index


DATES = _raw_dates()            # DATES[-1] = วันสุดท้ายจริง · DATES[-N_CUT-1] = วันสุดท้ายหลัง _cut_raw()
NEXT_DAY = DATES[-1] + pd.Timedelta(days=1)   # วันที่ยังไม่มีใน raw_data -> ใช้เป็น "แถวใหม่" ที่แต่งขึ้น
OVERLAP_PROBE = DATES[-10]      # วันที่อยู่กลางช่วงทับแน่นอน (เก่ากว่าขอบ _cut_raw ไม่กี่วันทำการ)

# NOW = วันสุดท้ายจริงใน raw_data + 3 วัน 10:00 -- ห่างพอที่แถวสุดท้ายจริงจะ "จบ" แน่นอน
# ไม่ว่าจะตรงกับวันหยุดสุดสัปดาห์พอดีหรือไม่ (กันปัญหา edge case ของ drop_unfinished)
NOW = (DATES[-1] + pd.Timedelta(days=3)).replace(
    hour=10, minute=0, tzinfo=F.BANGKOK)


def _cut_raw(d, n=N_CUT):
    """raw_data ปัจจุบันตัด n แถวท้าย -> จำลองว่ายังไม่ได้อัปเดต n วันล่าสุด"""
    for t in ("KBANK", "ADVANC"):
        lines = _raw(t).split(b"\r\n")[:-1]
        (Path(d) / f"{t}_10Y_Cleaned.csv").write_bytes(b"\r\n".join(lines[:-n]) + b"\r\n")


def _fake_downloader(extra=None, mutate=None):
    """คืน OHLCV จาก raw_data ปัจจุบัน (Volume = พันหุ้น × 1000) + แถวเสริม"""
    def dl(ticker, start, end_exclusive):
        _, df = A.read_canonical(ROOT / "raw_data" / f"{ticker.replace('.BK', '')}_10Y_Cleaned.csv")
        out = pd.DataFrame({"Open": df["Open"].astype(float), "High": df["High"].astype(float),
                            "Low": df["Low"].astype(float), "Close": df["Price"].astype(float),
                            "Volume": (df["Vol. ('000)"].astype(float) * 1000).round().astype("int64")})
        out = out[(out.index >= start) & (out.index < end_exclusive)]
        for d, row in (extra or {}).items():
            out.loc[pd.Timestamp(d)] = row
        out = out.sort_index()
        if mutate:
            mutate(ticker, out)
        return out
    return dl


def _raises(fn, *a, match="", **kw):
    try:
        fn(*a, **kw)
    except F.FetchError as e:
        assert match in str(e), f"ข้อความ error ไม่ตรง: {e}"
        return
    raise AssertionError("ต้อง raise FetchError")


def _write_files(d, plans):
    files = []
    for p in plans:
        f = Path(d) / f"{p['ticker']} Yahoo daily.csv"
        F.write_investing_csv(f, F.investing_rows(p["new"], p["prev_close"]))
        files.append(f)
    return files


def test_roundtrip_reproduces_raw_prices():
    """Yahoo(ปลอม) -> ไฟล์ Investing -> append_investing ได้ Date/OHLC/Change% ตรง raw_data จริงทุกตัวอักษร"""
    with tempfile.TemporaryDirectory() as d:
        _cut_raw(d)
        plans = F.plan_fetch(NOW, raw_dir=d, downloader=_fake_downloader())
        assert all(len(p["new"]) == N_CUT for p in plans)
        for ap in A.plan_append(_write_files(d, plans), raw_dir=d):
            assert not [w for w in ap["warnings"] if "Change %" in w], ap["warnings"]
            got = ap["candidate"].split(b"\r\n")[-N_CUT - 1:-1]
            exp = _raw(ap["ticker"]).split(b"\r\n")[-N_CUT - 1:-1]
            for g, e in zip(got, exp):
                gs, es = g.decode().split(","), e.decode().split(",")
                assert gs[:5] == es[:5] and gs[6] == es[6], (g, e)


def test_unfinished_today_dropped():
    """now = วันสุดท้ายจริง เวลา 16:45 (< FINAL_AFTER) -> แถวของ "วันนี้" ต้องถูกตัด เหลือถึงวันทำการก่อนหน้า"""
    with tempfile.TemporaryDirectory() as d:
        _cut_raw(d)
        early = DATES[-1].replace(hour=16, minute=45, tzinfo=F.BANGKOK)
        plans = F.plan_fetch(early, raw_dir=d, downloader=_fake_downloader())
        assert all(p["new"].index[-1] == DATES[-2] for p in plans)


def test_phantom_holiday_row_dropped():
    """แถวผีวันหยุด (Volume = 0 และ O=H=L=C) ที่ Yahoo แถมมา ต้องถูกตัดพร้อมคำเตือน"""
    extra = {NEXT_DAY: [252.0, 252.0, 252.0, 252.0, 0]}
    with tempfile.TemporaryDirectory() as d:
        _cut_raw(d)
        now = NEXT_DAY.replace(hour=20, minute=0, tzinfo=F.BANGKOK)
        plans = F.plan_fetch(now, raw_dir=d, downloader=_fake_downloader(extra))
        for p in plans:
            assert len(p["new"]) == N_CUT      # ได้แถวจริงที่ตัดไว้ครบ ไม่ใช่ว่างเพราะพลาดไปหมด
            assert NEXT_DAY not in p["new"].index
            assert any("แถวผี" in w for w in p["warnings"])


def test_price_mismatch_in_overlap_stops():
    def mut(ticker, df):
        if ticker == "ADVANC.BK":
            df.loc[OVERLAP_PROBE, "Close"] += 1
    with tempfile.TemporaryDirectory() as d:
        _cut_raw(d)
        _raises(F.plan_fetch, NOW, raw_dir=d, downloader=_fake_downloader(mutate=mut), match="Price ไม่ตรง")


def test_missing_overlap_date_stops():
    def mut(ticker, df):
        df.drop(OVERLAP_PROBE, inplace=True)
    with tempfile.TemporaryDirectory() as d:
        _cut_raw(d)
        _raises(F.plan_fetch, NOW, raw_dir=d, downloader=_fake_downloader(mutate=mut), match="วันที่ในช่วงทับ")


def test_zero_volume_real_move_stops():
    """Volume = 0 แต่ราคาไม่แบน = ไม่ใช่แถวผีปกติ -> ต้องหยุด ไม่ใช่ตัดทิ้งเงียบ ๆ"""
    extra = {NEXT_DAY: [252.0, 255.0, 250.0, 253.0, 0]}
    with tempfile.TemporaryDirectory() as d:
        _cut_raw(d)
        now = NEXT_DAY.replace(hour=20, minute=0, tzinfo=F.BANGKOK)
        _raises(F.plan_fetch, now, raw_dir=d, downloader=_fake_downloader(extra), match="Volume <= 0")


def test_tickers_must_end_same_day():
    def mut(ticker, df):
        if ticker == "KBANK.BK":
            df.drop(df.index[-1], inplace=True)
    with tempfile.TemporaryDirectory() as d:
        _cut_raw(d)
        _raises(F.plan_fetch, NOW, raw_dir=d, downloader=_fake_downloader(mutate=mut), match="ไม่เท่ากัน")


def test_volume_format_exact():
    # วันที่ตรงนี้เป็นแค่ index ของตารางสังเคราะห์ 1 แถว ไม่ได้อ่าน raw_data เลย -> ไม่ต้องผูกกับ DATES
    new = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [4564800]},
                       index=[pd.Timestamp("2026-09-28")])
    row = F.investing_rows(new, 1.0)[0]
    assert row[5] == "4564.8K" and A.vol_thousands(row[5]) == 4564.8
    assert row[6] == "0.00%"


def test_append_source_folder_named_by_source():
    with tempfile.TemporaryDirectory() as d:
        _cut_raw(d)
        plans = A.plan_append(_write_files(d, F.plan_fetch(NOW, raw_dir=d, downloader=_fake_downloader())),
                              raw_dir=d)
        out = A.archive_sources(plans, "20990101", sources_root=Path(d) / "src", source="yahoo")
        assert out.name == "yahoo_20990101" and "Yahoo" in (out / "SOURCES.md").read_text(encoding="utf-8")
        try:
            A.archive_sources(plans, "20990102", sources_root=Path(d) / "src", source="settrade")
        except A.AppendError:
            return
        raise AssertionError("source ที่ไม่รู้จักต้อง error")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
