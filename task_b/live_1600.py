"""
live_1600.py — live snapshot ของข้อมูลรายชั่วโมงสำหรับ Phase 1D (Mode A 16:00)
=============================================================================

snapshot = ไฟล์ CSV รายชั่วโมงที่ดาวน์โหลดตอนใช้งานจริง + meta (.meta.json)
    raw_data_intraday_live/<YYYYMMDD>/<KBANK_BK>_1h_<HHMMSS>.csv
    raw_data_intraday_live/<YYYYMMDD>/<KBANK_BK>_1h_<HHMMSS>.csv.meta.json
        {"ticker", "source", "downloaded_at" (+07:00), "sha256", "original_name"}

กติกา (PHASE1D_PLAN.md ข้อ 16):
  - ห้ามแก้ raw_data_intraday/ ที่ล็อก SHA · snapshot ใหม่เก็บแยกใน raw_data_intraday_live/
  - แท่งที่ทับกับไฟล์ล็อก: ใช้ค่าจากไฟล์ล็อก และรายงานแท่งที่ค่าไม่ตรง (ไม่ใช่ error)
  - แหล่งต้องเป็นแหล่งเดียวกับที่เทรน (Yahoo) -- แหล่งอื่น (เช่น SETTRADE) ต้องทำแผนรอบสองก่อน
  - snapshot ที่ใช้ทำนายต้องดาวน์โหลด ณ หรือหลัง 16:00 ของวันนั้น (แท่ง 15:00 จบที่ 16:00)
  - snapshot ที่ใช้เป็นผลจริง (close_bar16) ต้องดาวน์โหลดหลัง OUTCOME_READY (17:00) ของวันนั้น

ไม่ import data_loader / ไม่อ่าน raw_data/ (Investing) -- เหมือนโมดูล Phase 1D อื่น
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from intraday_1600 import (BANGKOK, BASE_DIR, EXPECTED_SHA256, TARGET_HOUR,
                           load_ticker_bars, prepare_bars, time_fields)

LIVE_DIR = BASE_DIR / "raw_data_intraday_live"
ALLOWED_SOURCES = {"yahoo"}          # แหล่งเดียวกับข้อมูลที่ใช้เทรน
OUTCOME_READY_HOUR = 17              # ปิดตลาด ~16:35-16:40 -> ใช้เป็นผลจริงได้หลัง 17:00
COMPARE = ["Open", "High", "Low", "Close", "Volume"]


class SnapshotError(RuntimeError):
    pass


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _ts(s):
    t = pd.Timestamp(s)
    if t.tzinfo is None:
        raise SnapshotError(f"เวลา {s!r} ต้องมี timezone")
    return t.tz_convert(BANGKOK)


# ---------------------------------------------------------------
# save / load
# ---------------------------------------------------------------
def save_snapshot(src, ticker, source, downloaded_at=None, root=LIVE_DIR):
    """คัดลอกไฟล์ byte-for-byte เข้า raw_data_intraday_live/ + เขียน meta · คืน path"""
    src = Path(src)
    if ticker not in EXPECTED_SHA256:
        raise SnapshotError(f"ไม่รู้จัก ticker {ticker}")
    source = source.strip().lower()
    downloaded_at = _ts(downloaded_at) if downloaded_at else pd.Timestamp(datetime.now(BANGKOK))
    raw = src.read_bytes()
    prepare_bars(pd.read_csv(src, encoding="utf-8-sig"))        # parse ได้จริงก่อนเก็บ
    d = Path(root) / downloaded_at.strftime("%Y%m%d")
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"{ticker.replace('.', '_')}_1h_{downloaded_at.strftime('%H%M%S')}.csv"
    if dst.exists():
        raise SnapshotError(f"{dst} มีอยู่แล้ว -- ห้ามเขียนทับ snapshot")
    shutil.copyfile(src, dst)
    meta = {"ticker": ticker, "source": source,
            "downloaded_at": downloaded_at.isoformat(timespec="seconds"),
            "sha256": _sha(raw), "original_name": src.name}
    Path(str(dst) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    return dst


def load_snapshot(path):
    """คืน (bars, meta) · ตรวจว่า SHA ของไฟล์ตรง meta"""
    path = Path(path)
    mp = Path(str(path) + ".meta.json")
    if not mp.exists():
        raise SnapshotError(f"ไม่พบ meta ของ {path.name}")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if _sha(path.read_bytes()) != meta["sha256"]:
        raise SnapshotError(f"{path.name}: SHA-256 ไม่ตรง meta -- ไฟล์ถูกแก้หลังบันทึก")
    meta["downloaded_at"] = _ts(meta["downloaded_at"])
    meta["path"] = path
    return prepare_bars(pd.read_csv(path, encoding="utf-8-sig")), meta


def find_snapshots(ticker, root=LIVE_DIR):
    """snapshot ทั้งหมดของหุ้นนี้ เรียงตามเวลาดาวน์โหลด"""
    stem = ticker.replace(".", "_")
    out = []
    for p in sorted(Path(root).glob(f"*/{stem}_1h_*.csv")):
        out.append(load_snapshot(p))
    return sorted(out, key=lambda x: x[1]["downloaded_at"])


def latest_snapshot_for(ticker, target, root=LIVE_DIR):
    """snapshot ล่าสุดที่ดาวน์โหลดในวัน target (สำหรับทำนาย)"""
    target = pd.Timestamp(target).normalize()
    same_day = [s for s in find_snapshots(ticker, root)
                if s[1]["downloaded_at"].tz_localize(None).normalize() == target]
    if not same_day:
        raise SnapshotError(f"ไม่พบ snapshot ของ {ticker} ที่ดาวน์โหลดในวัน {target.date()}")
    return same_day[-1]


# ---------------------------------------------------------------
# validate / merge
# ---------------------------------------------------------------
def validate_for_prediction(meta, target):
    """snapshot ที่ใช้ทำนาย: แหล่งถูก + ดาวน์โหลดในวันเดียวกัน ณ/หลัง 16:00"""
    if meta["source"] not in ALLOWED_SOURCES:
        raise SnapshotError(
            f"แหล่ง '{meta['source']}' ไม่ใช่แหล่งที่ใช้เทรน {sorted(ALLOWED_SOURCES)} -- "
            "การเปลี่ยนแหล่งข้อมูลต้องทำแผน Phase 1D รอบสองก่อน")
    tf = time_fields(target)
    dl = meta["downloaded_at"]
    if dl < tf["feature_window_end"]:
        raise SnapshotError(f"snapshot ดาวน์โหลด {dl.isoformat()} ก่อน 16:00 -- แท่ง 15:00 ยังไม่จบ")
    if dl.tz_localize(None).normalize() != pd.Timestamp(target).normalize():
        raise SnapshotError(f"snapshot ดาวน์โหลดคนละวันกับ target ({dl.date()})")


def merge_with_locked(locked, snap):
    """
    รวมไฟล์ล็อก + snapshot · แท่งที่ทับกันใช้ค่าจากไฟล์ล็อก
    คืน (bars, mismatches) -- mismatches = แท่งที่ทับแต่ค่าไม่ตรง (รายงานเท่านั้น)
    """
    key = ["date", "hour"]
    both = locked.merge(snap, on=key, suffixes=("_lock", "_snap"))
    diff = pd.Series(False, index=both.index)
    for c in COMPARE:
        diff |= both[f"{c}_lock"] != both[f"{c}_snap"]
    mism = both.loc[diff, key + [f"{c}_{s}" for c in COMPARE for s in ("lock", "snap")]]
    seen = set(map(tuple, locked[key].astype(str).to_numpy()))
    extra = snap[[tuple(k) not in seen for k in snap[key].astype(str).to_numpy()]]
    cols = ["Datetime", "Adj Close", "Close", "High", "Low", "Open", "Volume"]
    merged = pd.concat([locked[cols], extra[cols]], ignore_index=True)
    return prepare_bars(merged), mism.reset_index(drop=True)


def bars_for_prediction(ticker, target, snapshot=None, root=LIVE_DIR, locked=None):
    """
    แท่งที่ใช้ทำนาย target: ไฟล์ล็อก (+ snapshot ถ้ามี)
    คืน (bars, info) · info = sha, downloaded_at, source, mismatches, path
    """
    locked = load_ticker_bars(ticker) if locked is None else locked
    if snapshot is None:
        return locked, {"sha256": EXPECTED_SHA256[ticker], "downloaded_at": "", "source": "locked",
                        "path": "", "mismatches": 0}
    snap_bars, meta = snapshot
    validate_for_prediction(meta, target)
    bars, mism = merge_with_locked(locked, snap_bars)
    return bars, {"sha256": meta["sha256"], "downloaded_at": meta["downloaded_at"].isoformat(),
                  "source": meta["source"], "path": str(meta["path"]), "mismatches": len(mism)}


def outcome_close_bar16(ticker, target, snapshots, locked=None):
    """
    close_bar16 จริงของ target จาก snapshot ล่าสุดที่ดาวน์โหลดหลัง 17:00 ของวันนั้น
    (หรือจากไฟล์ล็อกสำหรับวันที่อยู่ในไฟล์ล็อก)
    คืน (C16, C15, source, sha) หรือ None ถ้ายังไม่มีผลจริง
    """
    target = pd.Timestamp(target).normalize()
    ready = time_fields(target)["feature_window_end"] + pd.Timedelta(hours=OUTCOME_READY_HOUR - 16)

    def pick(bars):
        day = bars[bars["date"] == target].set_index("hour")
        if TARGET_HOUR in day.index and 15 in day.index:
            return float(day.loc[TARGET_HOUR, "Close"]), float(day.loc[15, "Close"])
        return None

    for bars, meta in reversed(snapshots):
        if meta["downloaded_at"] >= ready and meta["source"] in ALLOWED_SOURCES:
            got = pick(bars)
            if got:
                return got[0], got[1], str(meta["path"]), meta["sha256"]
    if locked is not None:
        got = pick(locked)
        if got:
            return got[0], got[1], f"raw_data_intraday/{ticker.replace('.', '_')}_1h_730d.csv", \
                EXPECTED_SHA256[ticker]
    return None
