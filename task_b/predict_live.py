"""
predict_live.py — ทำนายราคาปิดของวันข้างหน้า แล้วบันทึกลง log (B3-B5)
====================================================================

    python predict_live.py --target-date 2026-09-28
    python predict_live.py --target-date 2025-02-25 --as-of 2025-02-24 --dry-run

*** ไฟล์นี้ไม่ใช่การเปิด test set ***
main.py = historical evaluation (เทรนถึง train_end ประเมินบน val)
ไฟล์นี้ = prospective prediction (เทรนด้วยข้อมูลที่รู้ผลแล้วทั้งหมด
          เพื่อทำนายวันที่ยังไม่เกิดขึ้น)
ช่วง 2025-02-26 -> 2026-08-28 ไม่ใช่ "อนาคต" ของการทำนายวันพรุ่งนี้อีกแล้ว
มันคือ labeled data ที่รู้ผลแล้ว การเอามาเทรนจึงถูกต้องตามหลักการ
แต่ห้ามเอาผลจากไฟล์นี้ไปรายงานเป็นผล test set เด็ดขาด

*** ห้ามให้สคริปต์เดาวันทำการถัดไปเอง ***
ปฏิทินวันหยุดไทยซับซ้อนเกินกว่าจะเดา -> บังคับระบุ --target-date เสมอ
"""

import argparse
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from config import TICKERS, OUTPUT_DIR, BASE_DIR, CONFIG_TAG
from data_loader import load_stock, raw_data_path, dataset_fingerprint
from features import build_features, build_live_feature
from targets import build_targets
from splits import prepare_xy          # ตัวเดียวกับ main.py (A4) ห้ามเขียนใหม่
from models import fit_live_models

LOG_PATH = OUTPUT_DIR / "prediction_log.csv"

# key ที่ห้ามซ้ำใน log (เฉพาะแถวที่ไม่ใช่ dry-run)
KEY = ["prediction_type", "ticker", "target_date", "model"]

# schema สุดท้ายของ Phase 1C -- ห้ามเปลี่ยนลำดับหลังจากนี้ (§1.5)
# ใส่ทุกคอลัมน์ที่ Phase 1C จะใช้ตั้งแต่ตอนนี้ แม้บางคอลัมน์ยังไม่มีความหมาย
# เพื่อไม่ให้ log append-only ไฟล์เดียวมีแถวคนละ schema
#   half_life  : ค่า half-life ที่ใช้จริง (None -> ค่าว่าง = ไม่ถ่วงน้ำหนัก)
#   config_tag : จาก config.CONFIG_TAG
LOG_COLUMNS = [
    "generated_at", "prediction_type", "ticker", "target_date", "data_cutoff",
    "train_rows", "half_life", "model", "predicted_return", "prev_close",
    "predicted_close", "n_features", "dataset_sha256", "code_commit",
    "config_tag", "is_dry_run",
]

# dry-run แยกไฟล์ออกจาก production log -- production log มีแต่คำทำนายจริง
# (แถว dry-run 18 แถวเดิมใน prediction_log.csv เป็น legacy ห้ามลบ/ย้ายเอง)
# worktree_dirty = True แปลว่าโค้ดที่รันไม่ใช่ code_commit เป๊ะ (ยังมีไฟล์ไม่ commit ใน task_b)
DRYRUN_LOG_PATH = OUTPUT_DIR / "dryrun" / "prediction_log_dryrun.csv"
DRYRUN_LOG_COLUMNS = LOG_COLUMNS + ["worktree_dirty"]

BANGKOK = timezone(timedelta(hours=7))


# ---------------------------------------------------------------
# Guards (B5) -- ทำงานเฉพาะตอนไม่ใช่ dry-run
# ---------------------------------------------------------------

def _git(*args):
    """
    เรียก git แล้ว raise ถ้าล้มเหลว -- ห้ามมี path ที่คืนค่าว่างแบบเงียบ ๆ (§1.4)

    ถ้าคืน string ว่างตอน git พัง (ไม่ใช่ repo / git ไม่อยู่ใน PATH / repo เสีย)
    guard จะตีความว่า "tree สะอาด" และ code_commit จะผิด -- ซึ่ง code_commit
    คือเหตุผลเดียวที่ prediction log มีน้ำหนักเป็นหลักฐาน
    """
    result = subprocess.run(
        ["git", *args],
        capture_output=True, text=True, cwd=BASE_DIR,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"คำสั่ง git ล้มเหลว: git {' '.join(args)}\n"
            f"  returncode = {result.returncode}\n"
            f"  stderr     = {result.stderr.strip()}\n"
            "  ถ้าอ่านสถานะ git ไม่ได้ ห้ามบันทึก official prediction"
        )
    return result.stdout.strip()


def get_code_commit(dry_run):
    """commit hash เต็มของโค้ดที่รันอยู่ -- ห้ามมีค่า 'unknown' เข้า log"""
    commit = _git("rev-parse", "HEAD")
    if len(commit) < 7:
        raise RuntimeError(f"อ่าน commit hash ไม่ได้ (ได้ {commit!r})")
    return commit


def worktree_dirty():
    """True ถ้ามีไฟล์ใดใน task_b ต่างจาก HEAD (รวม untracked) -- บันทึกใน dry-run log"""
    return bool(_git("status", "--porcelain", "--", "."))


def check_clean_tree():
    """
    B5.2 -- ห้ามทำนายตอน source/data ยังไม่ commit

    code_commit จะเชื่อถือได้ก็ต่อเมื่อโค้ดที่รันจริงตรงกับ commit นั้น
    ถ้าแก้ features.py แล้วยังไม่ commit แต่รัน prediction
    log จะบันทึก commit เก่าซึ่งไม่ใช่โค้ดที่ผลิตคำทำนายนั้นจริง
    -- หลักฐานทั้งชุดเสียทันที

    ตรวจเฉพาะ source กับ data ไม่สนไฟล์ผลลัพธ์ที่ generate ใหม่ทุกครั้ง
    """
    out = _git("status", "--porcelain", "--", "*.py", "raw_data/")
    if out:
        raise RuntimeError(
            "source หรือ raw_data ยังไม่ได้ commit:\n" + out + "\n"
            "ต้อง commit ก่อนจึงจะทำนายได้ ไม่งั้น code_commit ใน log "
            "จะไม่ตรงกับโค้ดที่รันจริง"
        )


def check_prediction_log_committed():
    """
    ห้ามทำนายรอบใหม่ถ้า log รอบก่อนยังไม่ได้ commit

    clean-tree guard ตั้งใจไม่สน results/ ซึ่งปกติถูกต้อง แต่เปิดช่องนี้:
        จันทร์ predict -> log เปลี่ยน -> ลืม commit
        อังคาร predict อีก -> append ต่อได้ -> ค่อย commit ทั้งสองวันพร้อมกัน
    duplicate guard ไม่ช่วยเพราะ target_date คนละวัน
    ผลคือหลักฐานของคำทำนายวันจันทร์อ่อนลง เพราะ commit เกิดหลัง
    outcome อาจรู้แล้ว

    ยกเว้นกรณีไฟล์ยังไม่เคยมี (prediction ครั้งแรก)
    """
    if not LOG_PATH.exists():
        return
    rel = LOG_PATH.relative_to(BASE_DIR).as_posix()
    if _git("status", "--porcelain", "--", rel):
        raise RuntimeError(
            f"{LOG_PATH.name} จากรอบก่อนยังไม่ได้ commit/push\n"
            "ให้จัดเก็บ log รอบก่อนให้เรียบร้อยก่อนทำนายรอบใหม่\n"
            "  git add task_b/results/prediction_log.csv\n"
            "  git commit -m 'prediction log: ...' && git push"
        )


def check_no_duplicate(new_rows, log_path=LOG_PATH):
    """
    B5.1 -- ห้ามทำนายซ้ำ key เดิม

    ถ้าเผลอรัน target-date เดิมสองครั้งจะได้ 12 แถว (2 หุ้น x 3 โมเดล x 2 รอบ)
    แล้วทีหลังจะไม่รู้ว่ารอบไหนคือคำทำนายอย่างเป็นทางการก่อนตลาดเปิด
    -- ซึ่งทำลายจุดประสงค์ทั้งหมดของ log
    """
    if not log_path.exists():
        return
    old = pd.read_csv(log_path)
    if old.empty:
        return
    old = old[~old["is_dry_run"].astype(str).str.lower().isin(["true", "1"])]
    if old.empty:
        return
    for k in KEY:                       # เทียบเป็น string กันชนิดข้อมูลเพี้ยน
        old[k] = old[k].astype(str)
    probe = new_rows[KEY].astype(str)
    merged = probe.merge(old[KEY].drop_duplicates(), on=KEY, how="inner")
    if len(merged):
        raise RuntimeError(
            "มีคำทำนายสำหรับ key นี้อยู่แล้วใน log:\n"
            f"{merged.drop_duplicates().to_string(index=False)}\n"
            "ปฏิเสธการเขียนซ้ำ -- คำทำนายที่บันทึกแล้วห้ามเขียนทับหรือเพิ่มซ้ำ"
        )


# ---------------------------------------------------------------
# Logger (B4) -- append-only
# ---------------------------------------------------------------

def append_log(log_path, rows, columns=LOG_COLUMNS):
    """
    เขียนต่อท้ายเท่านั้น ห้าม overwrite ห้ามแก้แถวเก่า
    ตรวจ schema ทุกครั้ง -- ห้าม append ผสม schema (§1.5)
    """
    rows = rows[columns]                           # KeyError = ขาดคอลัมน์ -> ต้องพัง
    if log_path.exists():
        header = pd.read_csv(log_path, nrows=0).columns.tolist()
        if header != columns:
            raise RuntimeError(
                f"schema ของ {log_path.name} ไม่ตรงที่โค้ดกำหนด\n"
                f"  ในไฟล์: {header}\n  ในโค้ด : {columns}\n"
                "  ห้าม append ผสม schema -- ดู §1.5"
            )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    print(f"[log] เขียนต่อท้าย {len(rows)} แถว -> "
          f"{log_path.relative_to(BASE_DIR) if log_path.is_relative_to(BASE_DIR) else log_path}")


def write_prediction_rows(out, dry_run, dirty=None,
                          log_path=LOG_PATH, dryrun_path=DRYRUN_LOG_PATH):
    """
    dry-run -> dryrun_path เท่านั้น (LOG_COLUMNS + worktree_dirty) · ห้ามแตะ production log
    official -> log_path (LOG_COLUMNS เดิม ไม่เปลี่ยน schema)
    คืน path ที่เขียนจริง
    """
    if dry_run:
        out = out.copy()
        out["worktree_dirty"] = bool(dirty)
        append_log(dryrun_path, out, DRYRUN_LOG_COLUMNS)
        return dryrun_path
    append_log(log_path, out, LOG_COLUMNS)
    return log_path


# ---------------------------------------------------------------
# ตัวหลัก
# ---------------------------------------------------------------

def predict_one_ticker(ticker, target_date, as_of=None, expected_cutoff=None,
                       verbose=True):
    """เทรนด้วยข้อมูลที่รู้ผลแล้วทั้งหมด แล้วทำนาย target_date"""
    print("\n" + "#" * 78)
    print(f"#  หุ้น: {ticker}  ->  ทำนาย {pd.Timestamp(target_date).date()}")
    print("#" * 78)

    df = load_stock(ticker)
    sha = dataset_fingerprint(raw_data_path(ticker))

    if as_of is not None:
        n_before = len(df)
        df = df.loc[:pd.Timestamp(as_of)].copy()
        print(f"[live] --as-of {pd.Timestamp(as_of).date()}: "
              f"ตัดข้อมูลเหลือ {len(df)}/{n_before} แถว")

    X = build_features(df)
    targets = build_targets(df)

    # ต้องเรียก prepare_xy เสมอ (A4) ไม่งั้นแถว warm-up 20 แถวหลุดเข้าไปเทรน
    X_prep, y_prep = prepare_xy(X, targets["y_return"])
    print(f"[live] ข้อมูลเทรน {len(X_prep)} แถว "
          f"({X_prep.index[0].date()} -> {X_prep.index[-1].date()})")

    X_live, prev_close, data_cutoff = build_live_feature(df, target_date)

    # ตรวจ "ทุกหุ้น" แยกกัน -- KBANK กับ ADVANC อาจมีวันสุดท้ายไม่เท่ากัน
    # ถ้า download มาไม่พร้อมกัน (§1.3) · ตรวจก่อนเทรน จะได้ fail เร็ว
    if expected_cutoff is not None:
        exp = pd.Timestamp(expected_cutoff)
        if data_cutoff != exp:
            raise RuntimeError(
                f"{ticker}: data_cutoff จริง = {data_cutoff.date()} "
                f"แต่ --expected-cutoff = {exp.date()}\n"
                "  ถ้าข้อมูลเก่ากว่าที่คาด -> ไป update raw_data/ ก่อน\n"
                "  ถ้าข้อมูลใหม่กว่าที่คาด -> แก้ --expected-cutoff ให้ตรงความจริง"
            )

    fitted, half_lives = fit_live_models(X_prep, y_prep)

    rows = []
    for name, model in fitted.items():
        ret = float(model.predict(X_live)[0])
        close = prev_close * (1 + ret)
        rows.append({
            "ticker": ticker,
            "data_cutoff": data_cutoff.date().isoformat(),
            "train_rows": len(X_prep),
            "half_life": half_lives[name],   # ค่าที่ใช้จริงของโมเดลในแถวนี้ (§3.5)
            "model": name,
            "predicted_return": ret,
            "prev_close": prev_close,
            "predicted_close": close,
            "n_features": X_live.shape[1],
            "dataset_sha256": sha,
        })
        print(f"    {name:15s} return = {ret:+.6f}  ->  ราคา {close:.4f} บาท")

    return rows


def parse_args():
    p = argparse.ArgumentParser(
        description="ทำนายราคาปิดของวันข้างหน้า แล้วบันทึกลง prediction_log.csv"
    )
    p.add_argument("--target-date", required=True,
                   help="วันที่จะทำนาย YYYY-MM-DD (ต้องเป็นวันทำการของตลาด)")
    p.add_argument("--as-of", default=None,
                   help="จำลองว่าวันนี้คือวันนี้ (ตัดข้อมูลถึงวันนี้) ใช้ตอน dry-run")
    p.add_argument("--dry-run", action="store_true",
                   help="ทดสอบ: ข้าม guard เรื่อง git และ duplicate, "
                        "เขียนลง results/dryrun/prediction_log_dryrun.csv "
                        "(ไม่แตะ production log)")
    # ค่า "same_day_1600" ถูกจงใจไม่ใส่ใน choices
    # schema ของ prediction_log รองรับค่านี้แล้ว แต่ pipeline ยังไม่มี
    # จะเปิดได้เมื่อ Phase 1D (โมเดล 16:00 จากข้อมูลรายชั่วโมง) เสร็จและ freeze แล้วเท่านั้น
    # -- ดู §6
    p.add_argument("--prediction-type", default="next_day",
                   choices=["next_day"],   # same_day_1600 ยังไม่ implement -- ห้ามเปิด
                   help="ชนิดการทำนาย (ตอนนี้มีแค่ next_day)")
    # ไม่ใช้ "ห่างไม่เกิน N วัน" เพราะวันหยุดตลาดไทยเดาล่วงหน้าไม่ได้
    # บังคับให้คนพิมพ์วันที่คาดหวังเอง -> แรงกว่าและตรวจย้อนหลังได้ (§1.3)
    p.add_argument("--expected-cutoff", default=None,
                   help="YYYY-MM-DD วันสุดท้ายของข้อมูลที่คาดว่าจะมี (บังคับตอน official)")
    return p.parse_args()


def main():
    args = parse_args()

    # ตรวจก่อนโหลดข้อมูลใด ๆ (§1.2)
    if args.as_of is not None and not args.dry_run:
        raise ValueError(
            "--as-of ใช้ได้เฉพาะกับ --dry-run เท่านั้น\n"
            "  --as-of มีไว้ย้อนเวลาเพื่อทดสอบระบบ ไม่ใช่เพื่อทำนายจริง\n"
            "  official prediction ต้องใช้ข้อมูลล่าสุดที่มีเสมอ"
        )

    # บังคับตอน official (§1.3)
    if not args.dry_run and args.expected_cutoff is None:
        raise ValueError(
            "official prediction ต้องใส่ --expected-cutoff YYYY-MM-DD\n"
            "  = วันทำการล่าสุดที่ควรมีใน raw_data/\n"
            "  guard นี้กัน 'ลืม update ข้อมูล' ซึ่งตรวจด้วย target_date > data_cutoff ไม่เจอ"
        )

    print("=" * 78)
    print("  งาน B : Live Prediction")
    print(f"  target_date = {args.target_date}"
          + (f"   as_of = {args.as_of}" if args.as_of else ""))
    if args.dry_run:
        print("  *** DRY RUN — ไม่ใช่คำทำนายอย่างเป็นทางการ ***")
    print("=" * 78)

    if not args.dry_run:
        check_clean_tree()
        check_prediction_log_committed()

    commit = get_code_commit(args.dry_run)
    generated_at = datetime.now(BANGKOK).isoformat(timespec="seconds")

    rows = []
    for ticker in TICKERS:
        rows.extend(predict_one_ticker(ticker, args.target_date,
                                       as_of=args.as_of,
                                       expected_cutoff=args.expected_cutoff))

    out = pd.DataFrame(rows)
    out["generated_at"] = generated_at
    out["prediction_type"] = args.prediction_type
    out["target_date"] = pd.Timestamp(args.target_date).date().isoformat()
    out["code_commit"] = commit
    # None -> ค่าว่างใน CSV · Int64 เพื่อไม่ให้ 1000 กลายเป็น "1000.0"
    out["half_life"] = out["half_life"].astype("Int64")
    out["config_tag"] = CONFIG_TAG
    out["is_dry_run"] = args.dry_run
    out = out[LOG_COLUMNS]

    if not args.dry_run:
        check_no_duplicate(out)

    dirty = worktree_dirty() if args.dry_run else False   # official ผ่าน clean-tree guard แล้ว
    written = write_prediction_rows(out, args.dry_run, dirty)

    print("\n" + "=" * 78)
    print(out[["ticker", "model", "predicted_return", "predicted_close"]]
          .to_string(index=False))
    print("=" * 78)
    if args.dry_run:
        print(f"\n*** DRY RUN — เขียนที่ {written.relative_to(BASE_DIR)} "
              f"(ไม่แตะ production log) · worktree_dirty={dirty} ***")
    else:
        print("\nอย่าลืม commit + push log:")
        print("  git add task_b/results/prediction_log.csv")
        print(f"  git commit -m 'prediction log: {args.target_date}' && git push")
    return out


if __name__ == "__main__":
    main()
