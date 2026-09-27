"""
predict_1600.py — Phase 1D (Mode A) live pipeline · DRY-RUN เท่านั้น
====================================================================

    python predict_1600.py --target-date 2026-09-25 --dry-run

ณ 16:00 ของ target_date ทำนาย close_bar16 ของวันเดียวกัน ด้วยผู้ชนะจาก tune_1600.py --run
(results/phase1d/phase1d_scores.csv) · นิยามทั้งหมดตาม PHASE1D_PLAN.md

*** รอบนี้ไม่มี official prediction ***
- ไม่มี --dry-run -> raise · target_date ในอนาคต -> raise
- เขียนลง results/dryrun/prediction_1600_dryrun.csv เท่านั้น -- ห้ามแตะ results/prediction_log.csv
- predict_live.py ยังไม่มี same_day_1600 ใน choices
- snapshot_downloaded_at: ใช้ไฟล์ historical ที่ล็อก SHA (ไม่ใช่ live snapshot) -> ค่าว่าง

ถ้า target_date เป็น eligible labeled day ในไฟล์ จะตรวจเพิ่มว่า
X_live เท่ากับ X ของ historical builder และ train_days = จำนวนแถว labeled ก่อน target_date
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_1600 import (BANGKOK, BASE_DIR, EXPECTED_SHA256, build_1600_live_feature,
                           build_dataset, intraday_path, load_ticker_bars)
from tune_1600 import GRIDS, LIVE_ANN_SEEDS, TICKERS, fit_predict, make_live_model

SCORES_PATH = BASE_DIR / "results" / "phase1d" / "phase1d_scores.csv"
DRYRUN_PATH = BASE_DIR / "results" / "dryrun" / "prediction_1600_dryrun.csv"
COLUMNS = [
    "generated_at", "prediction_type", "ticker", "target_date", "model", "config_id", "params",
    "predicted_return", "close_bar15", "predicted_close_bar16",
    "last_feature_bar_start", "feature_window_end", "target_bar_start", "snapshot_downloaded_at",
    "train_days", "n_features", "dataset_sha256", "code_commit", "worktree_dirty",
    "historical_x_match", "is_dry_run",
]


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


def predict_ticker(ticker, target, winners):
    bars = load_ticker_bars(ticker)                    # ตรวจ SHA ในตัว
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
            "snapshot_downloaded_at": "",               # historical locked file ไม่ใช่ live snapshot
            "train_days": live["train_days"], "n_features": live["X_live"].shape[1],
            "dataset_sha256": EXPECTED_SHA256[ticker], "historical_x_match": match,
        })
        print(f"    {ticker} {fam:15s} ŷ = {ret:+.6f} -> close_bar16 {rows[-1]['predicted_close_bar16']:.4f}"
              f" (C15 {live['close_bar15']}, train_days {live['train_days']})")
    return rows


def append_dryrun(rows, path=DRYRUN_PATH):
    df = pd.DataFrame(rows)[COLUMNS]
    if path.exists() and pd.read_csv(path, nrows=0).columns.tolist() != COLUMNS:
        raise RuntimeError(f"schema ของ {path.name} ไม่ตรง")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"[1600] เขียน {len(df)} แถว -> {path.relative_to(BASE_DIR)}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Phase 1D 16:00 dry-run (ไม่มี official prediction)")
    p.add_argument("--target-date", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if not args.dry_run:
        raise ValueError("predict_1600.py รอบนี้รองรับ --dry-run เท่านั้น -- official same_day_1600 ยังไม่เปิด")
    target = pd.Timestamp(args.target_date).normalize()
    today = pd.Timestamp(datetime.now(BANGKOK).date())
    if target > today:
        raise ValueError(f"target_date {target.date()} อยู่ในอนาคต -- dry-run ใช้ได้เฉพาะวันในอดีต")

    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--", "."))
    winners = load_winners()
    print(f"[1600] DRY RUN target {target.date()} · winners {winners} · ANN seeds {LIVE_ANN_SEEDS}")
    rows = []
    for t in TICKERS:
        rows += predict_ticker(t, target, winners)
    now = datetime.now(BANGKOK).isoformat(timespec="seconds")
    for r in rows:
        r.update(generated_at=now, prediction_type="same_day_1600", target_date=str(target.date()),
                 code_commit=commit, worktree_dirty=dirty, is_dry_run=True)
    append_dryrun(rows)
    return rows


if __name__ == "__main__":
    main()
