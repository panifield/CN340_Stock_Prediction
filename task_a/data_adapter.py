"""
data_adapter.py  — แก้ข้อ 1
===========================
แปลง CSV ฟอร์แมต Investing.com ให้เป็นฟอร์แมตที่ pipeline เดิมต้องการ

ปัญหาที่แก้
-----------
ไฟล์ที่ดาวน์โหลดมามีหัวคอลัมน์แบบนี้
    Date, Price, Open, High, Low, Vol. ('000), Change %
แต่ data_loader.py ประกาศไว้ว่า
    REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

จุดที่ต่างและต้องระวัง
  1. ไม่มีคอลัมน์ Close  -> Investing.com เรียกราคาปิดว่า "Price"
  2. ไม่มีคอลัมน์ Volume -> มี "Vol. ('000)" ซึ่ง *หน่วยเป็นพันหุ้น*
     ต้องคูณ 1000 ก่อน ไม่งั้น vol_ratio จะเทียบข้ามแหล่งข้อมูลไม่ได้
  3. วันที่เป็น MM/DD/YYYY ถ้าไม่ระบุ format ให้ชัด pandas อาจเดา
     เป็น DD/MM/YYYY สำหรับวันที่ <= 12 แล้วเรียงลำดับผิดแบบเงียบ ๆ
     ซึ่งเป็นบั๊กที่หายากที่สุดชนิดหนึ่ง
  4. บางไฟล์ Volume มาเป็น string แบบ "6,790.0" หรือ "1.2M"

วิธีใช้
------
    from data_adapter import load_investing_csv

    df = load_investing_csv("KBANK_10Y_Cleaned.csv")
    # ได้ DataFrame คอลัมน์ Open/High/Low/Close/Volume index เป็น DatetimeIndex

หรือแปลงเป็นไฟล์ cache ให้ pipeline เดิมอ่านได้เลย
    python data_adapter.py --in KBANK_10Y_Cleaned.csv --ticker KBANK.BK
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

# ชื่อคอลัมน์ที่เจอได้ -> ชื่อมาตรฐาน
_COLUMN_ALIASES = {
    "price": "Close",
    "close": "Close",
    "close*": "Close",
    "adj close": "Close",      # เตือนแยกด้านล่าง
    "open": "Open",
    "high": "High",
    "low": "Low",
    "vol.": "Volume",
    "vol": "Volume",
    "volume": "Volume",
    "date": "Date",
}

# หน่วยตัวคูณที่ Investing.com ใช้
_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9}


def _normalise_name(raw: str) -> tuple[str, float]:
    """
    คืน (ชื่อมาตรฐาน, ตัวคูณ)

    "Vol. ('000)" -> ("Volume", 1000.0)   เพราะหน่วยเป็นพันหุ้น
    "Price"       -> ("Close", 1.0)
    """
    name = raw.strip()

    # ดึงตัวคูณจากวงเล็บ เช่น ('000) หรือ (000s) หรือ (M)
    multiplier = 1.0
    paren = re.search(r"\(([^)]*)\)", name)
    if paren:
        inner = paren.group(1).replace("'", "").replace(",", "").strip().upper()
        if inner in {"000", "000S", "K", "THOUSANDS"}:
            multiplier = 1e3
        elif inner in {"M", "MM", "MILLIONS"}:
            multiplier = 1e6
        name = name[: paren.start()].strip()

    key = name.lower().rstrip(".").strip()
    return _COLUMN_ALIASES.get(key, name), multiplier


def _to_number(series: pd.Series, multiplier: float = 1.0) -> pd.Series:
    """
    แปลงคอลัมน์เป็นตัวเลข รองรับ "6,790.0" / "1.2M" / "12.5K" / "-"
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float) * multiplier

    s = series.astype(str).str.strip()
    s = s.replace({"-": np.nan, "": np.nan, "nan": np.nan, "None": np.nan})

    suffix_mult = pd.Series(1.0, index=s.index)
    for suf, mult in _SUFFIX.items():
        hit = s.str.upper().str.endswith(suf, na=False)
        suffix_mult[hit] = mult
        s = s.mask(hit, s.str.slice(0, -1))

    s = s.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return pd.to_numeric(s, errors="coerce") * suffix_mult * multiplier


def load_investing_csv(
    path: str,
    date_format: str = "%m/%d/%Y",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    อ่าน CSV ฟอร์แมต Investing.com คืน DataFrame ที่ pipeline ใช้ได้ทันที

    date_format : ค่าเริ่มต้น MM/DD/YYYY ตามที่ไฟล์ของเราเป็น
                  ถ้าดาวน์โหลดมาเป็น DD/MM/YYYY ให้ส่ง "%d/%m/%Y"
    """
    raw = pd.read_csv(path)

    rename, multipliers = {}, {}
    for col in raw.columns:
        std, mult = _normalise_name(col)
        rename[col] = std
        multipliers[std] = mult

    if verbose:
        changed = {k: v for k, v in rename.items() if k != v}
        if changed:
            print(f"[adapter] เปลี่ยนชื่อคอลัมน์: {changed}")
        scaled = {k: v for k, v in multipliers.items() if v != 1.0}
        if scaled:
            print(f"[adapter] ปรับหน่วย (คูณ): {scaled}")

    df = raw.rename(columns=rename)

    if "Adj Close" in raw.columns and "Close" in df.columns:
        print("[adapter] !! เตือน: ไฟล์นี้มี Adj Close — ห้ามใช้กับงาน parity")
        print("           ราคาที่ปรับ dividend จะหลุด tick grid ทำให้ label พัง")

    if "Date" not in df.columns:
        raise ValueError(f"ไม่พบคอลัมน์วันที่ใน {path} (มี {list(df.columns)})")

    # --- วันที่: ระบุ format ชัดเจน ห้ามให้ pandas เดา ---
    try:
        df["Date"] = pd.to_datetime(df["Date"], format=date_format)
    except ValueError as exc:
        raise ValueError(
            f"parse วันที่ด้วย format={date_format!r} ไม่ได้: {exc}\n"
            f"ตัวอย่างค่าที่เจอ: {df['Date'].head(3).tolist()}\n"
            f"ถ้าไฟล์เป็น DD/MM/YYYY ให้ส่ง date_format='%d/%m/%Y'"
        ) from exc

    df = df.set_index("Date").sort_index()
    df.index.name = "Date"

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"ยังขาดคอลัมน์ {missing} หลังแปลงแล้ว\n"
            f"คอลัมน์ที่มี: {list(df.columns)}"
        )

    for col in REQUIRED_COLS:
        df[col] = _to_number(df[col], multipliers.get(col, 1.0))

    df = df[REQUIRED_COLS]
    validate(df, verbose=verbose)
    return df


def validate(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    ตรวจความถูกต้องเชิงโครงสร้าง คืน dict สรุปผล
    ตัวที่สำคัญที่สุดคือ on_tick_grid — ถ้าไม่ใช่ 100% แปลว่าราคาถูกปรับมาแล้ว
    และงาน parity จะใช้ไม่ได้
    """
    from tick_utils import tick_size          # ใช้ร่วมกับไฟล์อื่น

    close = df["Close"]
    tick = close.map(tick_size)
    ratio = close / tick
    on_grid = np.isclose(ratio, np.rint(ratio), atol=1e-6)

    report = {
        "rows": len(df),
        "start": df.index.min().date(),
        "end": df.index.max().date(),
        "duplicate_dates": int(df.index.duplicated().sum()),
        "monotonic": bool(df.index.is_monotonic_increasing),
        "nan_close": int(close.isna().sum()),
        "ohlc_violations": int(
            ((df.High < df.Low) | (close > df.High) | (close < df.Low)).sum()
        ),
        "zero_volume_days": int((df["Volume"] == 0).sum()),
        "on_tick_grid_pct": float(on_grid.mean() * 100),
    }

    if verbose:
        print(f"[adapter] {report['rows']} แถว "
              f"({report['start']} -> {report['end']})")
        print(f"[adapter] วันซ้ำ {report['duplicate_dates']} | "
              f"NaN Close {report['nan_close']} | "
              f"OHLC ผิด {report['ohlc_violations']} | "
              f"Volume=0 {report['zero_volume_days']} วัน")
        print(f"[adapter] อยู่บน tick grid: {report['on_tick_grid_pct']:.2f}%")
        if report["on_tick_grid_pct"] < 99.5:
            print("[adapter] !! ราคาหลุด tick grid -> น่าจะเป็นข้อมูล adjusted")
            print("           ห้ามใช้กับงาน parity เด็ดขาด")

    for key, msg in [
        ("duplicate_dates", "มีวันที่ซ้ำ"),
        ("ohlc_violations", "ราคาปิดอยู่นอกช่วง High-Low"),
    ]:
        if report[key] > 0:
            raise ValueError(f"ข้อมูลผิดพลาด: {msg} ({report[key]} แถว)")
    if not report["monotonic"]:
        raise ValueError("index ไม่ได้เรียงตามเวลา")

    return report


def write_cache(df: pd.DataFrame, ticker: str, cache_dir: str,
                start: str, end: str, verbose: bool = True) -> str:
    """
    เขียนไฟล์ cache ในชื่อที่ data_loader.py / run_permutation.py มองหา
    รูปแบบ:  {TICKER_with_underscore}_{START}_{END}.csv
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe = ticker.replace("^", "").replace(".", "_")
    path = os.path.join(cache_dir, f"{safe}_{start}_{end}.csv")
    df.to_csv(path)
    if verbose:
        print(f"[adapter] เขียน cache: {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="แปลง CSV Investing.com เป็น cache ของ pipeline")
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--ticker", required=True,
                    help="เช่น KBANK.BK (ใช้ตั้งชื่อไฟล์ cache)")
    ap.add_argument("--cache-dir", default="data_cache")
    ap.add_argument("--start", default="2016-08-26")
    ap.add_argument("--end", default="2026-08-28")
    ap.add_argument("--date-format", default="%m/%d/%Y")
    args = ap.parse_args()

    print("=" * 70)
    print(f"  แปลงไฟล์: {args.infile}")
    print("=" * 70)
    from tick_utils import resolve_data_path
    infile = resolve_data_path(args.infile)
    frame = load_investing_csv(infile, date_format=args.date_format)
    write_cache(frame, args.ticker, args.cache_dir, args.start, args.end)
    print("\nตัวอย่าง 3 แถวแรก:")
    print(frame.head(3).to_string())
