"""
deploy/test_precheck.py -- ทดสอบ deploy/precheck.py (ไม่ใช้อินเทอร์เน็ต: downloader ปลอม, log ปลอมใน temp dir)

    python deploy/test_precheck.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import precheck as P  # noqa: E402

HOLIDAYS = ROOT / "task_b" / "set_holidays.txt"


def _hol():
    return P.RND.read_holidays(HOLIDAYS)


def _fake_root(rows_by_log):
    """temp root มี set_holidays.txt จริง + log ปลอม  rows_by_log = {rel_path: [(target_date, ptype or None), ...]}"""
    d = Path(tempfile.mkdtemp())
    (d / "task_b").mkdir()
    shutil.copy(HOLIDAYS, d / "task_b" / "set_holidays.txt")
    for rel, rows in rows_by_log.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"target_date": t, "is_dry_run": False, "prediction_type": pt or ""} for t, pt in rows]
                     ).to_csv(p, index=False)
    return d


ND_LOGS = ["task_a/results/prediction_log.csv", "task_c/results/prediction_log.csv", "task_b/results/prediction_log.csv"]


def _daily(close, day="2026-10-12"):
    return lambda tk: pd.DataFrame({"Close": [close]}, index=pd.DatetimeIndex([day]))


def test_target_and_cutoff_dates():
    hol, thr = _hol()
    f = lambda mode, now: tuple(None if x is None else str(x.date()) for x in P.target_and_cutoff(mode, pd.Timestamp(now), hol, thr))
    assert f("next_day", "2026-10-14 08:30") == ("2026-10-14", "2026-10-12")    # 10-13 หยุด -> cutoff ข้ามไป 10-12
    assert f("next_day", "2026-10-19 08:30") == ("2026-10-19", "2026-10-15")    # จันทร์: ถอยข้าม ส-อา และวันหยุดพิเศษ 10-16
    assert f("next_day", "2026-10-14 18:30") == ("2026-10-15", "2026-10-14")    # หลังตลาดปิด target = วันถัดไป
    assert f("next_day", "2026-10-23 08:30") == ("2026-10-26", "2026-10-22")    # 10-23 หยุด -> target เลื่อนเป็นจันทร์
    assert f("t1600", "2026-10-14 16:02") == ("2026-10-14", None)


def test_target_matches_run_next_day_logic():
    """target ต้องตรงกับที่ run_next_day.main() คำนวณ (reference = วันนี้-1 ถ้าก่อน 10:00)"""
    hol, thr = _hol()
    for now in ["2026-10-14 08:30", "2026-10-15 09:59", "2026-10-19 08:00", "2026-10-22 18:00"]:
        ts = pd.Timestamp(now)
        today = ts.normalize()
        ref = today - pd.Timedelta(days=1) if ts.hour < P.RND.MARKET_OPEN_HOUR else today
        assert P.target_and_cutoff("next_day", ts, hol, thr)[0] == P.RND.next_trading_day(ref, hol, thr), now


def test_yahoo_ready_cases():
    c = pd.Timestamp("2026-10-12")
    assert P.yahoo_ready(c, _daily(231.0)) == (True, [])
    ok, why = P.yahoo_ready(c, _daily(float("nan")))
    assert not ok and len(why) == 2 and "NaN" in why[0]                       # ทั้งสองหุ้น
    ok, why = P.yahoo_ready(c, _daily(231.0, day="2026-10-09"))
    assert not ok and "ยังไม่มีแถวของ 2026-10-12" in why[0]
    ok, why = P.yahoo_ready(c, lambda tk: (_ for _ in ()).throw(ConnectionError("boom")))
    assert not ok and "ดึงข้อมูลไม่ได้" in why[0]
    ok, why = P.yahoo_ready(c, lambda tk: None)
    assert not ok and "ไม่คืนข้อมูล" in why[0]


def test_log_status_ignores_dry_run_and_other_dates():
    d = _fake_root({ND_LOGS[0]: [("2026-10-14", None), ("2026-10-13", None)]})
    (d / "task_c" / "results").mkdir(parents=True)
    pd.DataFrame([{"target_date": "2026-10-14", "is_dry_run": True, "prediction_type": ""}]
                 ).to_csv(d / ND_LOGS[1], index=False)
    n, total, have = P.log_status(d, "next_day", pd.Timestamp("2026-10-14"))
    assert (n, total) == (1, 3) and have == ["task_a"], (n, have)               # dry-run ของ task_c ไม่นับ


def test_log_status_filters_prediction_type():
    d = _fake_root({ND_LOGS[2]: [("2026-10-14", "same_day_1600")]})            # task_b: ชนิดไม่ตรง ไม่นับเป็น next_day
    assert P.log_status(d, "next_day", pd.Timestamp("2026-10-14"))[0] == 0
    d = _fake_root({ND_LOGS[2]: [("2026-10-14", "next_day")]})
    assert P.log_status(d, "next_day", pd.Timestamp("2026-10-14"))[0] == 1


def test_t1600_task_b_counts_same_day_close_not_old_type():
    t1 = ["task_a2/results/prediction_log.csv", "task_b/results/prediction_log.csv",
          "task_c/results/prediction_log_1600.csv"]
    d = _fake_root({t1[1]: [("2026-10-14", "same_day_close")]})
    n, total, have = P.log_status(d, "t1600", pd.Timestamp("2026-10-14"))
    assert (n, total) == (1, 3) and have == ["task_b[same_day_close]"], (n, have)
    d = _fake_root({t1[1]: [("2026-10-14", "same_day_1600")]})              # แถวรอบ 1 ไม่นับว่าทำรอบ 2 แล้ว
    assert P.log_status(d, "t1600", pd.Timestamp("2026-10-14"))[0] == 0


def test_main_exit_codes():
    dl_ok, dl_nan = _daily(231.0), _daily(float("nan"))
    args = ["next_day", "--now", "2026-10-14 08:30"]
    # ยังไม่มี log เลย + Yahoo พร้อม -> 0 ; Yahoo NaN -> 11
    assert P.main(args, root=_fake_root({}), downloader=dl_ok) == P.EXIT_READY
    assert P.main(args, root=_fake_root({}), downloader=dl_nan) == P.EXIT_NOT_READY
    # ทำเสร็จครบทั้งสาม -> 10 (แม้ Yahoo จะเป็น NaN ก็ไม่สน เพราะไม่ต้องรันแล้ว)
    done = _fake_root({ND_LOGS[0]: [("2026-10-14", None)], ND_LOGS[1]: [("2026-10-14", None)],
                       ND_LOGS[2]: [("2026-10-14", "next_day")]})
    assert P.main(args, root=done, downloader=dl_nan) == P.EXIT_DONE
    # ทำไปแล้วบางส่วน -> 12
    part = _fake_root({ND_LOGS[0]: [("2026-10-14", None)]})
    assert P.main(args, root=part, downloader=dl_ok) == P.EXIT_PARTIAL
    # t1600 ไม่ตรวจ Yahoo
    assert P.main(["t1600", "--now", "2026-10-14 16:02"], root=_fake_root({}), downloader=dl_nan) == P.EXIT_READY
    # วันที่เกิน confirmed_through -> 13 (ตรวจไม่ได้ ห้ามเดา)
    assert P.main(["next_day", "--now", "2027-03-01 08:30"], root=_fake_root({}), downloader=dl_ok) == P.EXIT_ERROR


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
