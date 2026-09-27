"""
predict_1600.py — Phase 1D (Mode A) live pipeline
=================================================

ณ 16:00 ของ target_date ทำนาย close_bar16 ของวันเดียวกัน ด้วยผู้ชนะจาก tune_1600.py --run
(results/phase1d/phase1d_scores.csv) · นิยามทั้งหมดตาม PHASE1D_PLAN.md

    # dry-run ย้อนหลังจากไฟล์ที่ล็อก (วันที่อยู่ในไฟล์ล็อก)
    python predict_1600.py --target-date 2026-09-25 --dry-run
    # dry-run กับ live snapshot (ทดสอบ real-time availability)
    python predict_1600.py --target-date 2026-09-29 --dry-run --latest-snapshot
    # official (วันทำการ 16:00–16:30 · โค้ดและ log รอบก่อน commit แล้ว)
    python predict_1600.py --target-date 2026-09-29 --official --latest-snapshot

dry-run  -> results/dryrun/prediction_1600_dryrun.csv เท่านั้น
official -> results/prediction_log.csv (schema เดิม · prediction_type = same_day_1600)
            + results/prediction_log_1600_meta.csv (sidecar: เวลา/แหล่ง snapshot ที่ schema เดิมไม่มีช่อง)

Guard ของ official (ผิดข้อเดียว = ไม่เขียนอะไรเลย):
  1. เวอร์ชัน library ตรง requirements.txt
  2. เวลาตอนรัน: วันเดียวกับ target และ 16:00 <= now < 16:30 (+07:00)
  3. *.py / PHASE1D_PLAN.md / results/phase1d/ commit แล้วและไม่มีการแก้ค้าง
  4. prediction log + sidecar รอบก่อน commit แล้ว
  5. ใช้ live snapshot แหล่ง Yahoo ที่ดาวน์โหลด ณ/หลัง 16:00 ของวันนั้น · มีแท่ง 10, 11, 12, 14, 15
  6. key (prediction_type, ticker, target_date, model) ต้องยังไม่มีใน log

ไม่มีขั้นอนุมัติล่วงหน้า (PHASE1D_LIVE_APPROVAL.md ถูกยกเลิก 2026-09-28 -- ดู PHASE1D_PLAN.md ข้อ 18)
real-time availability ตรวจ "หลังเกิด" ทุกวันแทน: record_outcomes.py บันทึก prev_close_match
(close_bar15 ที่ใช้ทำนาย เทียบ snapshot หลัง 17:00) และ tools/check_snapshot_1600.py บันทึก
bar15_changed_vs_later ลง results/dryrun/availability_1600.csv

ถ้า target_date เป็น eligible labeled day ในข้อมูล จะตรวจเพิ่มว่า X_live เท่ากับ X ของ
historical builder และ train_days ถูกต้อง
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_1600 import BANGKOK, BASE_DIR, build_1600_live_feature, build_dataset
from live_1600 import (LIVE_DIR, SnapshotError, bars_for_prediction, latest_snapshot_for,
                       load_snapshot)
from tune_1600 import GRIDS, LIVE_ANN_SEEDS, TICKERS, fit_predict, make_live_model
from tools.check_env import require_pinned_env

SCORES_PATH = BASE_DIR / "results" / "phase1d" / "phase1d_scores.csv"
DRYRUN_PATH = BASE_DIR / "results" / "dryrun" / "prediction_1600_dryrun.csv"
LOG_PATH = BASE_DIR / "results" / "prediction_log.csv"
SIDECAR_PATH = BASE_DIR / "results" / "prediction_log_1600_meta.csv"
PREDICTION_TYPE = "same_day_1600"
CONFIG_TAG_1600 = "phase1d-modeA"
DEADLINE = (16, 30)
KEY = ["prediction_type", "ticker", "target_date", "model"]

COLUMNS = [                         # dry-run schema (ไม่เปลี่ยน)
    "generated_at", "prediction_type", "ticker", "target_date", "model", "config_id", "params",
    "predicted_return", "close_bar15", "predicted_close_bar16",
    "last_feature_bar_start", "feature_window_end", "target_bar_start", "snapshot_downloaded_at",
    "train_days", "n_features", "dataset_sha256", "code_commit", "worktree_dirty",
    "historical_x_match", "is_dry_run",
]
# ต้องเท่ากับ predict_live.LOG_COLUMNS (ตรวจใน tests) -- ไม่ import predict_live เพื่อไม่ดึง data_loader
LOG_COLUMNS = [
    "generated_at", "prediction_type", "ticker", "target_date", "data_cutoff",
    "train_rows", "half_life", "model", "predicted_return", "prev_close",
    "predicted_close", "n_features", "dataset_sha256", "code_commit",
    "config_tag", "is_dry_run",
]
SIDECAR_COLUMNS = KEY + ["generated_at", "snapshot_path", "snapshot_source", "snapshot_downloaded_at",
                         "snapshot_sha256", "locked_mismatch_bars", "last_feature_bar_start",
                         "feature_window_end", "target_bar_start", "config_id", "params"]


class GuardError(RuntimeError):
    pass


def _git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=BASE_DIR)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} ล้มเหลว: {r.stderr.strip()}")
    return r.stdout.strip()


def load_winners(path=SCORES_PATH):
    """ผู้ชนะต่อ family จากผลค้นหา · family ที่ไม่มี valid config -> ไม่มี live model"""
    s = pd.read_csv(path)
    out = {}
    for fam in GRIDS:
        w = s[(s["model"] == fam) & (s["winner"].astype(str) == "True")]
        if w.empty:
            print(f"[1600] {fam}: ไม่มี valid config -> ไม่มี live model")
            continue
        cid = int(w["config_id"].iloc[0])
        assert w["config"].iloc[0] == str(GRIDS[fam][cid]), "config ในผลค้นหาไม่ตรง GRIDS"
        out[fam] = cid
    return out


def check_against_historical(bars, target, live):
    """X_live == X(t) ของ historical builder และ train_days ถูกต้อง (เฉพาะวัน labeled)"""
    X, _, _, _ = build_dataset(bars)
    if target not in X.index:
        return None                      # วัน live ที่ยังไม่มี label -> ไม่มีอะไรให้เทียบ
    same_x = np.array_equal(live["X_live"].iloc[0].to_numpy(), X.loc[target].to_numpy())
    same_n = live["train_days"] == int((X.index < target).sum())
    if not (same_x and same_n):
        raise AssertionError(f"{target.date()}: live ไม่ตรง historical (X {same_x}, train_days {same_n})")
    return True


def predict_ticker(ticker, target, winners, bars, info):
    live = build_1600_live_feature(bars, target)
    match = check_against_historical(bars, target, live)
    rows = []
    for fam, cid in winners.items():
        pred, _ = fit_predict(make_live_model(fam, GRIDS[fam][cid]),
                              live["X_train"], live["y_train"], live["X_live"])
        ret = float(pred[0])
        rows.append({
            "ticker": ticker, "model": fam, "config_id": cid, "params": str(GRIDS[fam][cid]),
            "predicted_return": ret, "close_bar15": live["close_bar15"],
            "predicted_close_bar16": live["close_bar15"] * (1 + ret),
            "last_feature_bar_start": live["last_feature_bar_start"].isoformat(),
            "feature_window_end": live["feature_window_end"].isoformat(),
            "target_bar_start": live["target_bar_start"].isoformat(),
            "snapshot_downloaded_at": info["downloaded_at"],
            "snapshot_path": info["path"], "snapshot_source": info["source"],
            "locked_mismatch_bars": info["mismatches"],
            "train_days": live["train_days"], "n_features": live["X_live"].shape[1],
            "dataset_sha256": info["sha256"], "historical_x_match": match,
        })
        print(f"    {ticker} {fam:15s} ŷ = {ret:+.6f} -> close_bar16 {rows[-1]['predicted_close_bar16']:.4f}"
              f" (C15 {live['close_bar15']}, train_days {live['train_days']})")
    if info["mismatches"]:
        print(f"    ?? {ticker}: แท่งใน snapshot ที่ทับไฟล์ล็อกแต่ค่าไม่ตรง {info['mismatches']} แท่ง "
              "(ใช้ค่าจากไฟล์ล็อก)")
    return rows


# ---------------------------------------------------------------
# official guards
# ---------------------------------------------------------------
def check_clean(git=_git):
    dirty = git("status", "--porcelain", "--", "*.py", "PHASE1D_PLAN.md", "results/phase1d")
    if dirty:
        raise GuardError("โค้ด/แผน/ผลค้นหา ยังไม่ commit:\n" + dirty)
    pending = git("status", "--porcelain", "--", "results/prediction_log.csv",
                  "results/prediction_log_1600_meta.csv")
    if pending:
        raise GuardError("prediction log รอบก่อนยังไม่ commit:\n" + pending)


def check_time_window(target, now):
    now = pd.Timestamp(now).tz_convert(BANGKOK)
    if now.tz_localize(None).normalize() != target:
        raise GuardError(f"official ต้องรันในวัน target ({target.date()}) ตอนนี้ {now.isoformat()}")
    start = pd.Timestamp(year=target.year, month=target.month, day=target.day, hour=16, tz=BANGKOK)
    end = start.replace(hour=DEADLINE[0], minute=DEADLINE[1])
    if not start <= now < end:
        raise GuardError(f"official ต้องรันระหว่าง 16:00–16:30 ตอนนี้ {now.strftime('%H:%M:%S')}")


def check_no_duplicate(rows, log_path=LOG_PATH):
    if not log_path.exists():
        return
    old = pd.read_csv(log_path)
    old = old[~old["is_dry_run"].astype(str).str.lower().isin(["true", "1"])]
    have = set(map(tuple, old[KEY].astype(str).to_numpy()))
    dup = [tuple(r[k] for k in KEY) for r in rows if tuple(str(r[k]) for k in KEY) in have]
    if dup:
        raise GuardError(f"มีคำทำนาย key นี้ใน log แล้ว: {dup} -- ห้ามเขียนซ้ำ")


def write_official(rows, log_path=LOG_PATH, sidecar_path=SIDECAR_PATH):
    log = pd.DataFrame([{
        "generated_at": r["generated_at"], "prediction_type": PREDICTION_TYPE, "ticker": r["ticker"],
        "target_date": r["target_date"], "data_cutoff": r["feature_window_end"],
        "train_rows": r["train_days"], "half_life": "", "model": r["model"],
        "predicted_return": r["predicted_return"], "prev_close": r["close_bar15"],
        "predicted_close": r["predicted_close_bar16"], "n_features": r["n_features"],
        "dataset_sha256": r["dataset_sha256"], "code_commit": r["code_commit"],
        "config_tag": CONFIG_TAG_1600, "is_dry_run": False} for r in rows])[LOG_COLUMNS]
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
                          "target_bar_start": r["target_bar_start"],
                          "config_id": r["config_id"], "params": r["params"]} for r in rows])[SIDECAR_COLUMNS]
    if sidecar_path.exists() and pd.read_csv(sidecar_path, nrows=0).columns.tolist() != SIDECAR_COLUMNS:
        raise GuardError("schema ของ sidecar ไม่ตรง")
    log.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    side.to_csv(sidecar_path, mode="a", header=not sidecar_path.exists(), index=False)
    print(f"[1600] official: เขียน {len(log)} แถว -> {log_path.name} + {sidecar_path.name}")


def append_dryrun(rows, path=DRYRUN_PATH):
    df = pd.DataFrame(rows)[COLUMNS]
    if path.exists() and pd.read_csv(path, nrows=0).columns.tolist() != COLUMNS:
        raise RuntimeError(f"schema ของ {path.name} ไม่ตรง")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"[1600] dry-run: เขียน {len(df)} แถว -> {path}")


def _parse_snapshot_args(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--snapshot ต้องเป็น TICKER=PATH (ได้ {p!r})")
        t, path = p.split("=", 1)
        out[t.strip()] = load_snapshot(BASE_DIR / path)
    return out


def main(argv=None, now=None, live_root=LIVE_DIR):
    p = argparse.ArgumentParser(description="Phase 1D 16:00 prediction (dry-run / official)")
    p.add_argument("--target-date", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--official", action="store_true")
    p.add_argument("--snapshot", action="append", metavar="TICKER=PATH",
                   help="live snapshot ใน raw_data_intraday_live/ (ระบุทีละหุ้น)")
    p.add_argument("--latest-snapshot", action="store_true",
                   help="ใช้ snapshot ล่าสุดที่ดาวน์โหลดในวัน target ของแต่ละหุ้น")
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

    snaps = _parse_snapshot_args(args.snapshot)
    if args.latest_snapshot:
        for t in TICKERS:
            snaps.setdefault(t, latest_snapshot_for(t, target, live_root))
    if use_live and set(snaps) != set(TICKERS):
        raise SnapshotError(f"ต้องมี snapshot ครบทุกหุ้น {TICKERS} (ได้ {sorted(snaps)})")

    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--", "."))
    winners = load_winners()
    print(f"[1600] {'OFFICIAL' if args.official else 'DRY RUN'} target {target.date()} · "
          f"winners {winners} · ANN seeds {LIVE_ANN_SEEDS}")
    rows = []
    for t in TICKERS:
        bars, info = bars_for_prediction(t, target, snaps.get(t))
        rows += predict_ticker(t, target, winners, bars, info)
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
              "  git add results/prediction_log.csv results/prediction_log_1600_meta.csv "
              "raw_data_intraday_live/\n"
              f"  git commit -m 'prediction log 1600: {target.date()}' && git push")
    else:
        append_dryrun(rows)
    return rows


if __name__ == "__main__":
    main()
