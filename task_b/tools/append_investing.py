"""
tools/append_investing.py — append ไฟล์ Investing.com ใหม่ต่อท้าย raw_data/ อย่างปลอดภัย
=========================================================================================

    python tools/append_investing.py "<ไฟล์ Kasikornbank ...csv>" "<ไฟล์ Advanced Info ...csv>"
    python tools/append_investing.py <ไฟล์...> --check-only        # ตรวจอย่างเดียว ไม่เขียนอะไร

ขั้นตอน (หยุดทันทีถ้ามี error ใด -- ไม่แตะ raw_data/ เลย):
  1. parse ไฟล์ Investing (รองรับ BOM, quote, comma หลักพัน, เรียงใหม่->เก่า)
  2. จับคู่ ticker 2 ทาง: ชื่อไฟล์ + ราคาต่อเนื่อง |Open วันแรก / Close ล่าสุด − 1| < 10%
  3. แปลงเป็น canonical: MM/DD/YYYY · float repr (252.0) · Vol. ('000) (M×1000, K×1, B×1e6) ·
     Change % 2 ตำแหน่ง · CRLF · ASCII · ไม่มี quote · เรียงเก่า->ใหม่
  4. ตรวจ: วันซ้ำ/เสาร์-อาทิตย์ · overlap ต้องตรงทุกคอลัมน์ (ตรง = ข้าม, ขัด = error) ·
     OHLC · Volume ว่าง/'-' = error, 0 = เตือน · วันทำการที่ขาด = เตือน ·
     Change % เทียบราคาที่คำนวณได้ (ต่างเกิน 0.006 จุด% = เตือน)
  5. เขียน: ไฟล์ใหม่ต้องขึ้นต้นด้วย bytes เดิมทุกตัว (แถวเก่าไม่เปลี่ยน) · เขียนแบบ atomic
  6. เก็บต้นฉบับ byte-for-byte ใน raw_data_sources/investing_<YYYYMMDD>/ + SOURCES.md
  7. รัน check_raw_update.check_ticker (+ cross-check กับ raw_data_intraday แบบเตือน) · พิมพ์ SHA

ห้ามใช้ข้อมูล Yahoo / แหล่งอื่นเติม raw_data/ · หลัง append ต้องรัน `python main.py --dev`
แล้วเทียบ val CSV ว่าไม่เปลี่ยน (regression check)
"""

import argparse
import csv
import hashlib
import io
import re
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BASE_DIR, RAW_DATA_DIR, RAW_DATA_SUFFIX, TICKERS   # noqa: E402

BANGKOK = timezone(timedelta(hours=7))
HEADER = "Date,Price,Open,High,Low,Vol. ('000),Change %"
INPUT_HEADER = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]
VOL_MULT = {"K": Decimal(1), "M": Decimal(1000), "B": Decimal(1000000)}   # -> หน่วยพันหุ้น
NAME_PATTERNS = {"KBANK": ["kasikorn", "kbank"], "ADVANC": ["advanced info", "advanc"]}
CONTINUITY_MAX = 0.10
CHANGE_TOL = 0.006
SOURCES_ROOT = BASE_DIR / "raw_data_sources"


class AppendError(RuntimeError):
    """เงื่อนไขที่ต้องหยุด -- ไม่เขียนไฟล์ใด ๆ"""


# ---------------------------------------------------------------
# parse / normalize
# ---------------------------------------------------------------
def _num(s):
    s = s.replace(",", "").strip()
    if s in ("", "-"):
        raise AppendError(f"ค่าว่างหรือ '-': {s!r}")
    return float(s)


def vol_thousands(s):
    """'8.65M' -> 8650.0 (พันหุ้น) · คำนวณด้วย Decimal กัน 8650.000000000001"""
    s = s.replace(",", "").strip()
    if s in ("", "-"):
        raise AppendError(f"Volume ว่างหรือ '-': {s!r} -- ห้าม impute")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KMB])", s)
    if not m:
        raise AppendError(f"รูปแบบ Volume ไม่รู้จัก: {s!r}")
    return float(Decimal(m.group(1)) * VOL_MULT[m.group(2)])


def fmt_float(x):
    r = repr(float(x))
    assert float(r) == x
    return r


def parse_investing(path):
    """อ่านไฟล์ที่ดาวน์โหลดจาก Investing.com -> DataFrame เรียงเก่า->ใหม่ พร้อมค่าตัวเลข"""
    text = Path(path).read_bytes().decode("utf-8-sig")
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    if not rows or rows[0] != INPUT_HEADER:
        raise AppendError(f"{Path(path).name}: header ไม่ตรง {rows[0] if rows else None}")
    df = pd.DataFrame(rows[1:], columns=INPUT_HEADER)
    try:
        df.index = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    except ValueError as e:
        raise AppendError(f"{Path(path).name}: วันที่ parse ไม่ได้ ({e})") from e
    df = df.sort_index()
    for c in ["Price", "Open", "High", "Low"]:
        df[c + "_f"] = df[c].map(_num)
    df["Vol_k"] = df["Vol."].map(vol_thousands)
    df["Chg_f"] = df["Change %"].str.rstrip("%").map(_num)
    return df


def canonical_lines(new):
    out = []
    for dt, r in new.iterrows():
        out.append(",".join([dt.strftime("%m/%d/%Y"), fmt_float(r["Price_f"]), fmt_float(r["Open_f"]),
                             fmt_float(r["High_f"]), fmt_float(r["Low_f"]), fmt_float(r["Vol_k"]),
                             f"{r['Chg_f']:.2f}%"]))
    return out


def read_canonical(path):
    b = Path(path).read_bytes()
    if b.startswith(b"\xef\xbb\xbf") or not b.endswith(b"\r\n"):
        raise AppendError(f"{Path(path).name}: ไม่ใช่ canonical (มี BOM หรือไม่จบด้วย CRLF)")
    lines = b.decode("ascii").split("\r\n")[:-1]
    if lines[0] != HEADER:
        raise AppendError(f"{Path(path).name}: header ไม่ตรง canonical")
    df = pd.DataFrame([l.split(",") for l in lines[1:]], columns=HEADER.split(","))
    df.index = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    return b, df


# ---------------------------------------------------------------
# ticker mapping + validate
# ---------------------------------------------------------------
def ticker_from_name(fname):
    low = fname.lower()
    hits = [t for t, pats in NAME_PATTERNS.items() if any(p in low for p in pats)]
    if len(hits) != 1:
        raise AppendError(f"{fname}: ระบุ ticker จากชื่อไฟล์ไม่ได้ (เจอ {hits})")
    return hits[0]


def check_continuity(new, name_ticker, last_closes, last_date):
    """
    ราคาต้องต่อเนื่องกับ ticker จากชื่อไฟล์ และต้องไม่เข้าเกณฑ์ของ ticker อื่น
    ใช้ Open ของแถวแรกที่ "หลัง" วันสุดท้ายเดิม (ถ้าไม่มีแถวใหม่ ใช้แถวสุดท้ายของไฟล์)
    -> ไฟล์ที่ดาวน์โหลดช่วงยาวและทับของเดิมไม่ถูกตัดสินจากราคาเก่า
    (แถวที่ทับยังถูกเทียบค่าตรงทุกคอลัมน์ใน validate อีกชั้น)
    """
    after = new[new.index > last_date]
    first_open = (after if len(after) else new.iloc[[-1]])["Open_f"].iloc[0]
    dev = {t: abs(first_open / c - 1) for t, c in last_closes.items()}
    ok = dev[name_ticker] < CONTINUITY_MAX and all(
        v >= CONTINUITY_MAX for t, v in dev.items() if t != name_ticker)
    if not ok:
        raise AppendError(f"ชื่อไฟล์ ({name_ticker}) กับราคาต่อเนื่องไม่สอดคล้อง: "
                          + ", ".join(f"{t} {v:.1%}" for t, v in dev.items()))
    return dev


def validate(new, old, lines):
    """คืน (new_only, warnings) · raise AppendError ถ้าต้องหยุด"""
    warnings = []
    idx = new.index
    if not idx.is_unique:
        raise AppendError(f"วันที่ซ้ำในไฟล์ใหม่: {sorted(set(idx[idx.duplicated()].date))}")
    if (idx.dayofweek > 4).any():
        raise AppendError(f"มีวันเสาร์-อาทิตย์: {[str(d.date()) for d in idx[idx.dayofweek > 4]]}")

    last_date = old.index[-1]
    canon = dict(zip(idx, lines))
    for d in idx[idx <= last_date]:                             # Case B: overlap
        if d not in old.index:
            raise AppendError(f"{d.date()} อยู่ก่อนวันสุดท้ายเดิมแต่ไม่มีในไฟล์เดิม -- ห้ามแทรกย้อนหลัง")
        old_line = ",".join(old.loc[d].astype(str))
        if canon[d].split(",")[:6] != old_line.split(",")[:6]:
            raise AppendError(f"{d.date()}: ข้อมูลทับกับของเดิมแต่ค่าไม่ตรง\n  เดิม {old_line}\n  ใหม่ {canon[d]}")
    new_only = new[idx > last_date]
    if new_only.empty:
        return new_only, warnings + ["ไม่มีวันใหม่ให้ append (ทุกแถวทับกับของเดิมและตรงกัน)"]

    span = pd.bdate_range(last_date + pd.Timedelta(days=1), new_only.index[-1])
    missing = [str(d.date()) for d in span if d not in set(new_only.index)]
    if missing:
        warnings.append(f"วันทำการ (จ.-ศ.) ที่ไม่มีในไฟล์ใหม่ {missing} -- ตรวจว่าเป็นวันหยุด SET จริง ห้ามเติมข้อมูล")

    bad = new_only[(new_only["Low_f"] > new_only[["Open_f", "Price_f"]].min(axis=1))
                   | (new_only[["Open_f", "Price_f"]].max(axis=1) > new_only["High_f"])
                   | (new_only["High_f"] < new_only["Low_f"])]
    if len(bad):
        raise AppendError(f"OHLC ไม่สอดคล้อง: {[str(d.date()) for d in bad.index]}")
    zero = new_only[new_only["Vol_k"] <= 0]
    if len(zero):
        warnings.append(f"Volume = 0: {[str(d.date()) for d in zero.index]}")

    prev = [float(old["Price"].iloc[-1])] + new_only["Price_f"].iloc[:-1].tolist()
    calc = (new_only["Price_f"].to_numpy() / pd.Series(prev).to_numpy() - 1) * 100
    diff = abs(calc - new_only["Chg_f"].to_numpy())
    for d, c, r, dd in zip(new_only.index, calc, new_only["Chg_f"], diff):
        if dd > CHANGE_TOL:
            warnings.append(f"Change % {d.date()}: คำนวณได้ {c:.4f} แต่ไฟล์บอก {r:.2f} (ต่าง {dd:.4f})")
    return new_only, warnings


def build_candidate(old_bytes, new_only):
    add = "".join(l + "\r\n" for l in canonical_lines(new_only)).encode("ascii")
    cand = old_bytes + add
    if not cand.startswith(old_bytes):
        raise AppendError("candidate ไม่ขึ้นต้นด้วย bytes เดิม")
    return cand


# ---------------------------------------------------------------
# main flow
# ---------------------------------------------------------------
def sha256(b):
    return hashlib.sha256(b).hexdigest()


def plan_append(files, raw_dir=RAW_DATA_DIR, tickers=TICKERS):
    """ตรวจทุกไฟล์ก่อน -- คืน list ของแผนต่อ ticker (ยังไม่เขียนอะไร)"""
    raw_dir = Path(raw_dir)
    short = [t.replace(".BK", "") for t in tickers]
    olds = {t: read_canonical(raw_dir / f"{t}{RAW_DATA_SUFFIX}") for t in short}
    last_closes = {t: float(df["Price"].iloc[-1]) for t, (_, df) in olds.items()}
    plans, seen = [], set()
    for f in files:
        f = Path(f)
        t = ticker_from_name(f.name)
        if t in seen:
            raise AppendError(f"ได้ไฟล์ของ {t} มากกว่าหนึ่งไฟล์")
        seen.add(t)
        new = parse_investing(f)
        old_bytes, old = olds[t]
        dev = check_continuity(new, t, last_closes, old.index[-1])
        new_only, warns = validate(new, old, canonical_lines(new))
        cand = build_candidate(old_bytes, new_only) if len(new_only) else old_bytes
        plans.append({"ticker": t, "source": f, "new": new, "new_only": new_only,
                      "warnings": warns, "continuity": dev, "old_bytes": old_bytes,
                      "candidate": cand, "target": raw_dir / f"{t}{RAW_DATA_SUFFIX}"})
    missing = set(short) - seen
    if missing:
        raise AppendError(f"ขาดไฟล์ของ {sorted(missing)} -- ต้องอัปเดตทุกหุ้นพร้อมกัน "
                          "(ไม่งั้น data_cutoff ของสองหุ้นไม่เท่ากัน)")
    return plans


def archive_sources(plans, date_tag, sources_root=SOURCES_ROOT):
    """เก็บต้นฉบับ byte-for-byte + SOURCES.md (ห้ามเขียนทับไฟล์ต้นฉบับที่ต่างกัน)"""
    d = Path(sources_root) / f"investing_{date_tag}"
    d.mkdir(parents=True, exist_ok=True)
    for p in plans:
        dst = d / p["source"].name
        src_b = p["source"].read_bytes()
        if dst.exists() and dst.read_bytes() != src_b:
            raise AppendError(f"{dst} มีอยู่แล้วแต่เนื้อไฟล์ต่างกัน -- ห้ามเขียนทับต้นฉบับ")
        if not dst.exists():
            shutil.copyfile(p["source"], dst)
    md = d / "SOURCES.md"
    lines = [] if md.exists() else [
        f"# Investing.com daily update — ไฟล์ต้นฉบับ ({date_tag})", "",
        "สร้างโดย `tools/append_investing.py` · ไฟล์ต้นฉบับเก็บแบบ byte-for-byte ห้ามแก้ · "
        "pipeline อ่านจาก `raw_data/` เท่านั้น", "",
        "| ไฟล์เดิม | ticker | SHA-256 ต้นฉบับ | แถวในไฟล์ | append ใหม่ | ช่วงที่ append | "
        "raw_data SHA-256 ก่อน | raw_data SHA-256 หลัง | บันทึกเมื่อ |",
        "|---|---|---|---|---|---|---|---|---|"]
    now = datetime.now(BANGKOK).isoformat(timespec="seconds")
    for p in plans:
        n = p["new_only"]
        rng = f"{n.index[0].date()} → {n.index[-1].date()}" if len(n) else "-"
        lines.append(f"| `{p['source'].name}` | {p['ticker']}.BK | `{sha256(p['source'].read_bytes())}` | "
                     f"{len(p['new'])} | {len(n)} | {rng} | `{sha256(p['old_bytes'])}` | "
                     f"`{sha256(p['candidate'])}` | {now} |")
    with md.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return d


def write_atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def run_check_raw_update(plans, intraday_dir):
    """ใช้ check_raw_update.check_ticker ตัวเดิม (old = bytes เดิมใน temp dir)"""
    from check_raw_update import check_ticker
    ok = True
    with tempfile.TemporaryDirectory() as d:
        for p in plans:
            old = Path(d) / p["target"].name
            old.write_bytes(p["old_bytes"])
            ok &= check_ticker(old, p["target"], intraday_dir)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description="append ไฟล์ Investing.com ต่อท้าย raw_data/ อย่างปลอดภัย")
    ap.add_argument("files", nargs="+", type=Path, help="ไฟล์ที่ดาวน์โหลดจาก Investing.com (ทุกหุ้น)")
    ap.add_argument("--check-only", action="store_true", help="ตรวจอย่างเดียว ไม่เขียนไฟล์ใด ๆ")
    ap.add_argument("--date-tag", default=datetime.now(BANGKOK).strftime("%Y%m%d"),
                    help="ชื่อโฟลเดอร์ใน raw_data_sources/ (default = วันนี้)")
    ap.add_argument("--no-cross-check", action="store_true",
                    help="ไม่เทียบกับ raw_data_intraday (เตือนเท่านั้นอยู่แล้ว)")
    args = ap.parse_args(argv)

    try:
        plans = plan_append(args.files)
    except AppendError as e:
        print(f"!! หยุด -- ไม่ได้เขียนไฟล์ใด ๆ\n   {e}")
        return 1

    for p in plans:
        n = p["new_only"]
        print(f"\n=== {p['ticker']} <- {p['source'].name}")
        print("  continuity: " + ", ".join(f"{t} {v:.2%}" for t, v in p["continuity"].items()))
        print(f"  แถวในไฟล์ {len(p['new'])} · append ใหม่ {len(n)}"
              + (f" ({n.index[0].date()} -> {n.index[-1].date()})" if len(n) else ""))
        if len(n):
            s = canonical_lines(n)
            print(f"  แถวแรก: {s[0]}\n  แถวท้าย: {s[-1]}")
        for w in p["warnings"]:
            print(f"  ?? เตือน: {w}")

    if args.check_only:
        print("\n[check-only] ผ่านการตรวจ -- ไม่ได้เขียนไฟล์ใด ๆ")
        return 0
    if not any(len(p["new_only"]) for p in plans):
        print("\nไม่มีแถวใหม่ -- ไม่ได้เขียนไฟล์ใด ๆ")
        return 0

    src_dir = archive_sources(plans, args.date_tag)
    for p in plans:
        if len(p["new_only"]):
            write_atomic(p["target"], p["candidate"])
    intraday = None if args.no_cross_check else BASE_DIR / "raw_data_intraday"
    ok = run_check_raw_update(plans, intraday)
    print("\nSHA-256 raw_data:")
    for p in plans:
        print(f"  {p['ticker']}: {sha256(p['old_bytes'])}\n  {' ' * len(p['ticker'])}  -> {sha256(p['candidate'])}")
    print(f"ต้นฉบับเก็บที่ {src_dir.relative_to(BASE_DIR)}")
    if not ok:
        print("!! check_raw_update ไม่ผ่าน -- คืนไฟล์เดิมด้วย git แล้วหาสาเหตุ")
        return 1
    print("\nต่อไป: python main.py --dev แล้วเทียบ val CSV ว่าไม่เปลี่ยน (git diff results/*_val.csv ต้องว่าง)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
