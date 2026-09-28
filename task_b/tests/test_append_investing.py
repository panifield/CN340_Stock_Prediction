"""
tests/test_append_investing.py — tools/append_investing.py

    python tests/test_append_investing.py

ทุก test เขียนลง temp dir เท่านั้น -- ไม่แตะ raw_data/ ของจริง
"""

import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import append_investing as A        # noqa: E402

SRC = ROOT / "raw_data_sources" / "investing_20260927"
KB_SRC = SRC / "Kasikornbank Stock Price History 0831-0927.csv"
AD_SRC = SRC / "Advanced Info Stock Price History 0831-0927.csv"
PRE_APPEND_SHA = {"KBANK": "c2068d4ce1c71db799a361b9103fe563bffc8adde418629fac01e5106fbe61f6",
                  "ADVANC": "cbd5db1d6dbc3dd0bce59266825ee11e8cb49e12d2bcb425a7d33e4300933c68"}


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _pre_append_raw(d):
    """raw_data ก่อน append 2026-09-27 (session1, จาก KB_SRC/AD_SRC จริง) -- ตัดตามวันที่ที่
    archive นี้ครอบคลุม ไม่ใช่ "20 แถวท้ายของไฟล์ปัจจุบัน" ตายตัว เพราะ raw_data/ โตขึ้นทุกวันจาก
    daily_next_day.py จริง ถ้าตัดแบบนับแถวจากท้าย จำนวนที่ต้องตัดจะเปลี่ยนทุกครั้งที่มีข้อมูลใหม่
    กว่าช่วงของ archive นี้ถูก append เข้ามา (เจอบั๊กนี้จริง 2026-09-29 หลัง raw_data ขยับไปถึง
    28 ก.ย. ทั้งที่ archive นี้ครอบคลุมแค่ถึง 27 ก.ย. เท่านั้น) -- ตัดตามขอบเขตวันที่ของ archive
    แทน ทำให้ผลลัพธ์คงที่เสมอไม่ว่า raw_data จะโตต่อไปอีกแค่ไหนหลังจากนี้"""
    for t, src in (("KBANK", KB_SRC), ("ADVANC", AD_SRC)):
        b, df = A.read_canonical(ROOT / "raw_data" / f"{t}_10Y_Cleaned.csv")
        boundary = A.parse_investing(src).index.min()
        keep_n = int((df.index < boundary).sum())
        lines = b.split(b"\r\n")[:-1]
        old = b"\r\n".join(lines[:1 + keep_n]) + b"\r\n"     # header + keep_n แถวข้อมูล
        assert _sha(old) == PRE_APPEND_SHA[t]
        (Path(d) / f"{t}_10Y_Cleaned.csv").write_bytes(old)


def _investing_file(path, rows):
    """rows = [(MM/DD/YYYY, price, open, high, low, vol, chg)] -> รูปแบบดาวน์โหลดจริง (BOM, quote, ใหม่->เก่า)"""
    head = '"Date","Price","Open","High","Low","Vol.","Change %"\n'
    body = "".join(",".join(f'"{v}"' for v in r) + "\n" for r in reversed(rows))
    Path(path).write_bytes(b"\xef\xbb\xbf" + (head + body).encode("utf-8"))


def _raises(fn, *a, match=""):
    try:
        fn(*a)
    except A.AppendError as e:
        assert match in str(e), f"ข้อความ error ไม่ตรง: {e}"
        return
    raise AssertionError("ต้อง raise AppendError")


def test_reproduces_session1_append_exactly():
    with tempfile.TemporaryDirectory() as d:
        _pre_append_raw(d)
        plans = A.plan_append([KB_SRC, AD_SRC], raw_dir=d)
        for p in plans:
            real = (ROOT / "raw_data" / f"{p['ticker']}_10Y_Cleaned.csv").read_bytes()
            # เทียบแค่ prefix ที่ยาวเท่า candidate (= raw_data ตอนนั้น ทันทีหลัง session1) ไม่ใช่
            # ทั้งไฟล์ปัจจุบัน เพราะ raw_data/ โตขึ้นทุกวันจริงหลังจากนั้น -- append-only แปลว่า
            # bytes ช่วงต้นของไฟล์ปัจจุบันต้องตรงกับตอนนั้นเป๊ะเสมอไม่ว่าจะมีอะไรถูกเติมต่อท้าย
            # มาอีกแค่ไหน (เจอบั๊กนี้จริง 2026-09-29 หลัง raw_data ขยับไปถึง 28 ก.ย.)
            assert p["candidate"] == real[:len(p["candidate"])], p["ticker"]
            assert len(p["new_only"]) == 20
            assert p["candidate"].startswith(p["old_bytes"])
            assert not any("Change %" in w for w in p["warnings"])


def test_full_overlap_identical_appends_nothing():
    plans = A.plan_append([KB_SRC, AD_SRC])            # raw_data จริงมี 20 วันนี้แล้ว
    for p in plans:
        assert len(p["new_only"]) == 0 and p["candidate"] == p["old_bytes"]


def test_overlap_conflict_stops():
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "Kasikornbank x.csv"
        _investing_file(bad, [("09/25/2026", "253.00", "253.00", "254.00", "251.00", "8.65M", "0.80%")])
        _raises(A.plan_append, [bad, AD_SRC], match="ค่าไม่ตรง")


def test_ticker_mismatch_stops():
    with tempfile.TemporaryDirectory() as d:
        wrong = Path(d) / "Advanced Info swapped.csv"     # ชื่อ ADVANC แต่ราคาเป็นของ KBANK
        wrong.write_bytes(KB_SRC.read_bytes())
        _pre_append_raw(d)
        _raises(A.plan_append, [wrong], d, match="ไม่สอดคล้อง")


def test_volume_rules():
    assert A.vol_thousands("8.65M") == 8650.0
    assert A.vol_thousands("995.80K") == 995.8
    assert A.vol_thousands("1.2B") == 1200000.0
    assert A.vol_thousands("1,234.5K") == 1234.5
    for bad in ("-", "", "12x"):
        _raises(A.vol_thousands, bad)


def _synthetic_update(d, rows_kb, rows_ad):
    _pre_append_raw(d)
    kb, ad = Path(d) / "Kasikornbank new.csv", Path(d) / "Advanced Info new.csv"
    _investing_file(kb, rows_kb)
    _investing_file(ad, rows_ad)
    return kb, ad


AD_OK = [("08/31/2026", "355.00", "352.00", "357.00", "350.00", "6.03M", "0.85%")]


def test_missing_volume_and_ohlc_stop():
    with tempfile.TemporaryDirectory() as d:
        kb, ad = _synthetic_update(
            d, [("08/31/2026", "248.00", "248.00", "251.00", "245.00", "-", "0.40%")], AD_OK)
        _raises(A.plan_append, [kb, ad], d, match="Volume")
    with tempfile.TemporaryDirectory() as d:
        kb, ad = _synthetic_update(      # High < Price
            d, [("08/31/2026", "252.00", "248.00", "251.00", "245.00", "1.00M", "2.02%")], AD_OK)
        _raises(A.plan_append, [kb, ad], d, match="OHLC")


def test_dates_rules():
    with tempfile.TemporaryDirectory() as d:
        kb, ad = _synthetic_update(      # วันเสาร์
            d, [("08/29/2026", "248.00", "248.00", "251.00", "245.00", "1.00M", "0.40%")], AD_OK)
        _raises(A.plan_append, [kb, ad], d, match="เสาร์")
    with tempfile.TemporaryDirectory() as d:
        row = ("08/31/2026", "248.00", "248.00", "251.00", "245.00", "1.00M", "0.40%")
        kb, ad = _synthetic_update(d, [row, row], AD_OK)
        _raises(A.plan_append, [kb, ad], d, match="ซ้ำ")
    with tempfile.TemporaryDirectory() as d:
        kb, ad = _synthetic_update(      # ข้าม 09-01 -> เตือน · Change % ผิด -> เตือน
            d, [("08/31/2026", "248.00", "248.00", "251.00", "245.00", "1.00M", "0.40%"),
                ("09/02/2026", "250.00", "248.00", "251.00", "247.00", "1.00M", "9.99%")], AD_OK)
        p = [x for x in A.plan_append([kb, ad], raw_dir=d) if x["ticker"] == "KBANK"][0]
        assert any("2026-09-01" in w for w in p["warnings"])
        assert any("Change %" in w for w in p["warnings"])


def test_requires_all_tickers():
    _raises(A.plan_append, [KB_SRC], match="ขาดไฟล์")


def test_check_only_writes_nothing():
    before = {t: _sha((ROOT / "raw_data" / f"{t}_10Y_Cleaned.csv").read_bytes()) for t in ("KBANK", "ADVANC")}
    assert A.main([str(KB_SRC), str(AD_SRC), "--check-only"]) == 0
    after = {t: _sha((ROOT / "raw_data" / f"{t}_10Y_Cleaned.csv").read_bytes()) for t in ("KBANK", "ADVANC")}
    assert before == after


def test_archive_sources_byte_for_byte():
    with tempfile.TemporaryDirectory() as d:
        _pre_append_raw(d)
        plans = A.plan_append([KB_SRC, AD_SRC], raw_dir=d)
        out = A.archive_sources(plans, "20990101", sources_root=Path(d) / "src")
        for p in plans:
            assert (out / p["source"].name).read_bytes() == p["source"].read_bytes()
        assert (out / "SOURCES.md").exists()
        (out / KB_SRC.name).write_bytes(b"changed")
        _raises(A.archive_sources, plans, "20990101", Path(d) / "src", match="ห้ามเขียนทับ")


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
