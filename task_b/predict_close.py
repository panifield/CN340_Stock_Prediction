"""
predict_close.py — Phase 1D รอบ 2 live pipeline (PHASE1D_PLAN_R2.md)
=====================================================================

ณ 16:00 ของ target_date ทำนาย "ราคาปิดทางการ" ของวันเดียวกัน ด้วยผู้ชนะจาก tune_1600.py --run --round r2
(results/phase1d_r2/phase1d_scores.csv) · target  y = ราคาปิดทางการ / C15 − 1 · predicted_close = C15 × (1 + ŷ)
รอบ 1 (predict_1600.py · target close_bar16) ยังอยู่ครบเป็นประวัติ แต่ไม่ถูกเรียกโดย daily_1600.py เริ่มต้นแล้ว

    # dry-run ย้อนหลังจากไฟล์ที่ล็อก (วันที่อยู่ในไฟล์ล็อก ≤ 2026-09-25)
    python predict_close.py --target-date 2026-09-25 --dry-run
    # dry-run กับ live snapshot (ทั้งรายชั่วโมงและรายวัน)
    python predict_close.py --target-date 2026-10-19 --dry-run --latest-snapshot
    # official (วันทำการ 16:00–16:30 · โค้ดและ log รอบก่อน commit แล้ว)
    python predict_close.py --target-date 2026-10-19 --official --latest-snapshot

dry-run  -> results/dryrun/prediction_close_dryrun.csv เท่านั้น
official -> results/prediction_log.csv (schema เดิม · prediction_type = same_day_close · config_tag = phase1d-r2-close)
            + results/prediction_log_close_meta.csv (sidecar: เวลา/แหล่ง snapshot รายชั่วโมงและรายวัน)

Guard ของ official (ผิดข้อเดียว = ไม่เขียนอะไรเลย) -- เหมือน predict_1600.py + ข้อ 7–8:
  1. เวอร์ชัน library ตรง requirements.txt
  2. เวลาตอนรัน: วันเดียวกับ target และ 16:00 <= now < 16:30 (+07:00)
  3. *.py / แผน / results/phase1d* / raw_data_daily commit แล้วและไม่มีการแก้ค้าง
  4. prediction log + sidecar รอบก่อน commit แล้ว
  5. ใช้ live snapshot รายชั่วโมง (Yahoo) ที่ดาวน์โหลด ณ/หลัง 16:00 ของวันนั้น · มีแท่ง 10, 11, 12, 14, 15
  6. key (prediction_type, ticker, target_date, model) ต้องยังไม่มีใน log
  7. ใช้ snapshot ราคาปิดรายวัน (Yahoo) ที่ดาวน์โหลดในวันเดียวกับ target
  8. ต้องมีราคาปิดทางการของ "วันทำการก่อนหน้า" (ตามปฏิทิน set_holidays.txt) ไม่งั้นไม่ทำนาย
ราคาปิดของวัน target เองและวันหลังจากนั้น ถูกตัดทิ้งก่อนสร้าง feature / label เสมอ (intraday_1600.build_1600_live_feature)
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

import daily_close as DC
from intraday_1600 import BANGKOK, BASE_DIR, build_1600_live_feature, build_dataset
from live_1600 import LIVE_DIR, SnapshotError, bars_for_prediction, latest_snapshot_for, load_snapshot
from predict_1600 import (GuardError, LOG_COLUMNS, _git, check_no_duplicate, check_time_window,
                          load_winners)
from tune_1600 import GRIDS, LIVE_ANN_SEEDS, TICKERS, fit_predict, make_live_model
from tools.check_env import require_pinned_env

SCORES_PATH = BASE_DIR / "results" / "phase1d_r2" / "phase1d_scores.csv"
DRYRUN_PATH = BASE_DIR / "results" / "dryrun" / "prediction_close_dryrun.csv"
LOG_PATH = BASE_DIR / "results" / "prediction_log.csv"
SIDECAR_PATH = BASE_DIR / "results" / "prediction_log_close_meta.csv"
PREDICTION_TYPE = "same_day_close"
CONFIG_TAG_CLOSE = "phase1d-r2-close"
KEY = ["prediction_type", "ticker", "target_date", "model"]

COLUMNS = [                         # dry-run schema
    "generated_at", "prediction_type", "ticker", "target_date", "model", "config_id", "params",
    "predicted_return", "close_bar15", "predicted_close",
    "last_feature_bar_start", "feature_window_end", "target_time", "snapshot_downloaded_at",
    "daily_close_source", "daily_close_sha256", "daily_close_last_date",
    "train_days", "n_features", "dataset_sha256", "code_commit", "worktree_dirty",
    "historical_x_match", "is_dry_run",
]
SIDECAR_COLUMNS = KEY + [
    "generated_at", "snapshot_path", "snapshot_source", "snapshot_downloaded_at", "snapshot_sha256",
    "locked_mismatch_bars", "last_feature_bar_start", "feature_window_end", "target_time",
    "config_id", "params",
    "daily_close_path", "daily_close_source", "daily_close_sha256", "daily_close_last_date",
    "daily_close_mismatch_rows", "daily_close_dropped_rows",
]


def target_time(target):
    """เวลาประกาศราคาปิดทางการ (รอบประมูลปิดตลาด ~16:35) -- ใช้บันทึกเท่านั้น"""
    d = pd.Timestamp(target).date()
    return pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=16, minute=35, tz=BANGKOK)


def check_clean(git=_git):
    dirty = git("status", "--porcelain", "--", "*.py", "PHASE1D_PLAN.md", "PHASE1D_PLAN_R2.md",
                "results/phase1d", "results/phase1d_r2", "raw_data_daily")
    if dirty:
        raise GuardError("โค้ด/แผน/ผลค้นหา/ไฟล์ล็อกราคาปิด ยังไม่ commit:\n" + dirty)
    pending = git("status", "--porcelain", "--", "results/prediction_log.csv",
                  "results/prediction_log_close_meta.csv")
    if pending:
        raise GuardError("prediction log รอบก่อนยังไม่ commit:\n" + pending)


def check_against_historical(bars, target, live, daily):
    """X_live == X(t) ของ historical builder และ train_days ถูกต้อง (เฉพาะวัน labeled -- daily ต้องมีราคาปิดของ target)"""
    X, _, _, _ = build_dataset(bars, daily)
    if target not in X.index:
        return None                      # วัน live ที่ยังไม่มี label -> ไม่มีอะไรให้เทียบ
    import numpy as np
    same_x = np.array_equal(live["X_live"].iloc[0].to_numpy(), X.loc[target].to_numpy())
    same_n = live["train_days"] == int((X.index < target).sum())
    if not (same_x and same_n):
        raise AssertionError(f"{target.date()}: live ไม่ตรง historical (X {same_x}, train_days {same_n})")
    return True


def predict_ticker(ticker, target, winners, bars, info, daily, dinfo):
    live = build_1600_live_feature(bars, target, daily)
    match = check_against_historical(bars, target, live, daily)
    rows = []
    for fam, cid in winners.items():
        pred, _ = fit_predict(make_live_model(fam, GRIDS[fam][cid]),
                              live["X_train"], live["y_train"], live["X_live"])
        ret = float(pred[0])
        rows.append({
            "ticker": ticker, "model": fam, "config_id": cid, "params": str(GRIDS[fam][cid]),
            "predicted_return": ret, "close_bar15": live["close_bar15"],
            "predicted_close": live["close_bar15"] * (1 + ret),
            "last_feature_bar_start": live["last_feature_bar_start"].isoformat(),
            "feature_window_end": live["feature_window_end"].isoformat(),
            "target_time": target_time(target).isoformat(),
            "snapshot_downloaded_at": info["downloaded_at"],
            "snapshot_path": info["path"], "snapshot_source": info["source"],
            "locked_mismatch_bars": info["mismatches"],
            "daily_close_path": dinfo["path"], "daily_close_source": dinfo["source"],
            "daily_close_sha256": dinfo["sha256"], "daily_close_last_date": dinfo["last_date"],
            "daily_close_mismatch_rows": dinfo["mismatches"], "daily_close_dropped_rows": dinfo["dropped_rows"],
            "train_days": live["train_days"], "n_features": live["X_live"].shape[1],
            "dataset_sha256": info["sha256"], "historical_x_match": match,
        })
        print(f"    {ticker} {fam:15s} ŷ = {ret:+.6f} -> ราคาปิด {rows[-1]['predicted_close']:.4f}"
              f" (C15 {live['close_bar15']}, ปิดวันก่อน {live['daily_prev_close']}, train_days {live['train_days']})")
    if info["mismatches"]:
        print(f"    ?? {ticker}: แท่งใน snapshot ที่ทับไฟล์ล็อกแต่ค่าไม่ตรง {info['mismatches']} แท่ง "
              "(ใช้ค่าจากไฟล์ล็อก)")
    if dinfo["mismatches"]:
        print(f"    ?? {ticker}: ราคาปิดรายวันใน snapshot ที่ทับไฟล์ล็อกแต่ไม่ตรง {dinfo['mismatches']} วัน "
              "(ใช้ค่าจากไฟล์ล็อก)")
    return rows


def write_official(rows, log_path=None, sidecar_path=None):
    log_path = LOG_PATH if log_path is None else log_path          # อ่านตอนเรียก (tests ปรับ path ได้)
    sidecar_path = SIDECAR_PATH if sidecar_path is None else sidecar_path
    log = pd.DataFrame([{
        "generated_at": r["generated_at"], "prediction_type": PREDICTION_TYPE, "ticker": r["ticker"],
        "target_date": r["target_date"], "data_cutoff": r["feature_window_end"],
        "train_rows": r["train_days"], "half_life": "", "model": r["model"],
        "predicted_return": r["predicted_return"], "prev_close": r["close_bar15"],
        "predicted_close": r["predicted_close"], "n_features": r["n_features"],
        "dataset_sha256": r["dataset_sha256"], "code_commit": r["code_commit"],
        "config_tag": CONFIG_TAG_CLOSE, "is_dry_run": False} for r in rows])[LOG_COLUMNS]
    if log_path.exists() and pd.read_csv(log_path, nrows=0).columns.tolist() != LOG_COLUMNS:
        raise GuardError("schema ของ prediction_log.csv ไม่ตรง LOG_COLUMNS")
    side = pd.DataFrame([{**{k: r[k] for k in KEY if k != "prediction_type"},
                          "prediction_type": PREDICTION_TYPE,
                          "generated_at": r["generated_at"], "snapshot_path": r["snapshot_path"],
                          "snapshot_source": r["snapshot_source"],
                          "snapshot_downloaded_at": r["snapshot_downloaded_at"],
                          "snapshot_sha256": r["dataset_sha256"],
                          "locked_mismatch_bars": r["locked_mismatch_bars"],
                          "last_feature_bar_start": r["last_feature_bar_start"],
                          "feature_window_end": r["feature_window_end"],
                          "target_time": r["target_time"],
                          "config_id": r["config_id"], "params": r["params"],
                          "daily_close_path": r["daily_close_path"],
                          "daily_close_source": r["daily_close_source"],
                          "daily_close_sha256": r["daily_close_sha256"],
                          "daily_close_last_date": r["daily_close_last_date"],
                          "daily_close_mismatch_rows": r["daily_close_mismatch_rows"],
                          "daily_close_dropped_rows": r["daily_close_dropped_rows"]} for r in rows])[SIDECAR_COLUMNS]
    if sidecar_path.exists() and pd.read_csv(sidecar_path, nrows=0).columns.tolist() != SIDECAR_COLUMNS:
        raise GuardError("schema ของ sidecar ไม่ตรง")
    log.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    side.to_csv(sidecar_path, mode="a", header=not sidecar_path.exists(), index=False)
    print(f"[close] official: เขียน {len(log)} แถว -> {log_path.name} + {sidecar_path.name}")


def append_dryrun(rows, path=None):
    path = DRYRUN_PATH if path is None else path                  # อ่านตอนเรียก (tests ปรับ path ได้)
    df = pd.DataFrame(rows)[COLUMNS]
    if path.exists() and pd.read_csv(path, nrows=0).columns.tolist() != COLUMNS:
        raise RuntimeError(f"schema ของ {path.name} ไม่ตรง")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"[close] dry-run: เขียน {len(df)} แถว -> {path}")


def _parse_snapshot_args(pairs, loader):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--snapshot ต้องเป็น TICKER=PATH (ได้ {p!r})")
        t, path = p.split("=", 1)
        out[t.strip()] = loader(BASE_DIR / path)
    return out


def main(argv=None, now=None, live_root=LIVE_DIR, daily_root=DC.LIVE_DAILY_DIR, holidays=None):
    p = argparse.ArgumentParser(description="Phase 1D รอบ 2: ทำนายราคาปิดทางการ ณ 16:00 (dry-run / official)")
    p.add_argument("--target-date", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--official", action="store_true")
    p.add_argument("--snapshot", action="append", metavar="TICKER=PATH",
                   help="live snapshot รายชั่วโมงใน raw_data_intraday_live/ (ระบุทีละหุ้น)")
    p.add_argument("--daily-snapshot", action="append", metavar="TICKER=PATH",
                   help="live snapshot ราคาปิดรายวันใน raw_data_daily_live/ (ระบุทีละหุ้น)")
    p.add_argument("--latest-snapshot", action="store_true",
                   help="ใช้ snapshot ล่าสุด (รายชั่วโมง + รายวัน) ที่ดาวน์โหลดในวัน target ของแต่ละหุ้น")
    args = p.parse_args(argv)

    injected = now is not None           # tests ส่งเวลาเข้ามา · ใช้งานจริงใช้นาฬิกาเครื่อง

    def clock():
        return pd.Timestamp(now) if injected else pd.Timestamp(datetime.now(BANGKOK))

    now = clock()
    target = pd.Timestamp(args.target_date).normalize()
    if target > now.tz_localize(None).normalize():
        raise ValueError(f"target_date {target.date()} อยู่ในอนาคต")

    use_live = args.latest_snapshot or args.snapshot
    if args.official:
        require_pinned_env()             # เวอร์ชัน library ต้องตรง requirements.txt (INTEGRATION.md ข้อ 5)
        check_time_window(target, now)
        check_clean()
        if not use_live:
            raise GuardError("official ต้องใช้ live snapshot (--latest-snapshot หรือ --snapshot)")
        if not (args.latest_snapshot or args.daily_snapshot):
            raise GuardError("official ต้องใช้ snapshot ราคาปิดรายวัน (--latest-snapshot หรือ --daily-snapshot)")

    snaps = _parse_snapshot_args(args.snapshot, load_snapshot)
    dsnaps = _parse_snapshot_args(args.daily_snapshot, DC.load_snapshot)
    if args.latest_snapshot:
        for t in TICKERS:
            snaps.setdefault(t, latest_snapshot_for(t, target, live_root))
            dsnaps.setdefault(t, DC.latest_snapshot_for(t, target, daily_root))
    if use_live and set(snaps) != set(TICKERS):
        raise SnapshotError(f"ต้องมี snapshot รายชั่วโมงครบทุกหุ้น {TICKERS} (ได้ {sorted(snaps)})")
    if args.official and set(dsnaps) != set(TICKERS):
        raise DC.DailyCloseError(f"ต้องมี snapshot ราคาปิดรายวันครบทุกหุ้น {TICKERS} (ได้ {sorted(dsnaps)})")

    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--", "."))
    winners = load_winners(SCORES_PATH)
    print(f"[close] {'OFFICIAL' if args.official else 'DRY RUN'} target {target.date()} · "
          f"winners {winners} · ANN seeds {LIVE_ANN_SEEDS}")
    rows = []
    for t in TICKERS:
        bars, info = bars_for_prediction(t, target, snaps.get(t))
        daily, dinfo = DC.daily_close_for_prediction(t, target, dsnaps.get(t), check_previous=True,
                                                     holidays=holidays)
        rows += predict_ticker(t, target, winners, bars, info, daily, dinfo)
    generated_at = clock().isoformat(timespec="seconds")      # เวลาที่ออกคำทำนายจริง (หลัง fit)
    for r in rows:
        r.update(generated_at=generated_at,
                 prediction_type=PREDICTION_TYPE, target_date=str(target.date()),
                 code_commit=commit, worktree_dirty=dirty, is_dry_run=not args.official)
    if args.official:
        check_time_window(target, clock())                    # ต้องยังไม่เลย 16:30 หลัง fit เสร็จ
        check_no_duplicate(rows)
        write_official(rows)
        print("\nอย่าลืม commit + push ก่อน 16:30:\n"
              "  git add results/prediction_log.csv results/prediction_log_close_meta.csv "
              "raw_data_intraday_live/ raw_data_daily_live/\n"
              f"  git commit -m 'prediction log close: {target.date()}' && git push")
    else:
        append_dryrun(rows)
    return rows


if __name__ == "__main__":
    main()
