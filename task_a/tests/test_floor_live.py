"""
tests/test_floor_live.py — ส่วน live ที่เปลี่ยนตอนย้ายไป label floor (2026-10-24)

  - log_schema.migrate_log_columns: เพิ่มคอลัมน์โดยแถวเก่าไม่เปลี่ยนแม้แต่ byte เดียว
  - evaluate_live.score: แถวเก่าใช้กฎเดิม แถว floor ใช้ parity_label + กฎ H1

    python -m pytest -q tests/test_floor_live.py
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from log_schema import migrate_log_columns  # noqa: E402
import evaluate_live as EV  # noqa: E402
import predict_live as PL  # noqa: E402


def _raises(fn, *a, match="", **kw):
    try:
        fn(*a, **kw)
    except RuntimeError as e:
        assert match in str(e), f"ข้อความ error ไม่ตรง: {e}"
        return
    raise AssertionError("ต้อง raise RuntimeError")


def test_migrate_appends_empty_columns_and_keeps_old_bytes():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        old = b"a,b\r\n1,0.480580\r\n2,True\r\n"
        log.write_bytes(old)
        assert migrate_log_columns(log, ["a", "b"], ["c", "d"])
        assert log.read_bytes() == b"a,b,c,d\r\n1,0.480580,,\r\n2,True,,\r\n"
        # แถวเดิมทุกบรรทัด = บรรทัดเดิม + ",," เท่านั้น
        new_lines = log.read_bytes().split(b"\r\n")[1:3]
        assert [l[:-2] for l in new_lines] == old.split(b"\r\n")[1:3]


def test_migrate_is_idempotent_and_rejects_unknown_schema():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        log.write_text("a,b,c\n1,2,\n", encoding="utf-8")
        assert not migrate_log_columns(log, ["a", "b"], ["c"])   # ย้ายแล้ว ไม่ทำซ้ำ
        log.write_text("a,x\n1,2\n", encoding="utf-8")
        _raises(migrate_log_columns, log, ["a", "b"], ["c"], match="schema")


def test_migrate_then_append_log_accepts_new_schema():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        pd.DataFrame([{c: 1 for c in PL.OLD_LOG_COLUMNS}]).to_csv(log, index=False)
        migrate_log_columns(log, PL.OLD_LOG_COLUMNS, PL.NEW_LOG_COLUMNS)
        row = {c: 2 for c in PL.OLD_LOG_COLUMNS} | {"label_rule": "floor"}
        PL.append_log(pd.DataFrame([row]), log)
        out = pd.read_csv(log)
        assert out["label_rule"].isna().iloc[0] and out["label_rule"].iloc[1] == "floor"


def test_real_logs_migrate_without_touching_old_rows():
    for path, mod in ((ROOT / "results" / "prediction_log.csv", PL),):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / "log.csv"
            tmp.write_bytes(path.read_bytes())
            migrate_log_columns(tmp, mod.OLD_LOG_COLUMNS, mod.NEW_LOG_COLUMNS)
            before = pd.read_csv(path, dtype=str)
            after = pd.read_csv(tmp, dtype=str)
            assert after[before.columns].fillna("").equals(before.fillna(""))
            assert after[mod.NEW_LOG_COLUMNS].isna().all().all()


def test_evaluate_scores_old_rows_with_old_rule_and_floor_rows_with_floor():
    # 162.5: กฎเดิม -> 163 คี่ · floor -> 162 คู่
    close = pd.Series([162.5, 162.5], index=pd.DatetimeIndex(["2026-10-14", "2026-10-26"]))
    log = pd.DataFrame({
        "ticker": ["KBANK.BK"] * 2,
        "target_date": ["2026-10-14", "2026-10-26"],
        "model": ["ANN (MLP)"] * 2,
        "parity_1600": [1.0, 0.0],
        "predicted_parity": [1.0, 0.0],
        "is_dry_run": [False, False],
        "label_rule": [np.nan, "floor"],
        "h1_status": [np.nan, "predict"],
        "h1_predicted_parity": [np.nan, 0.0],
    })
    with mock.patch.object(EV, "official_close", return_value=close):
        out = EV.score(log, "parity_1600")
    assert out["actual_parity"].tolist() == [1.0, 1.0]           # กฎเดิมคำนวณทุกแถว
    assert out["correct"].iloc[0] == 1.0 and np.isnan(out["correct"].iloc[1])
    assert np.isnan(out["actual_parity_floor"].iloc[0]) and out["actual_parity_floor"].iloc[1] == 0.0
    assert np.isnan(out["correct_floor"].iloc[0]) and out["correct_floor"].iloc[1] == 1.0
    assert out["baseline_correct_floor"].iloc[1] == 1.0
    assert np.isnan(out["h1_correct"].iloc[0]) and out["h1_correct"].iloc[1] == 1.0


def test_write_outcomes_keeps_existing_column_order_first():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "outcomes.csv"
        pd.DataFrame(columns=["a", "status", "correct"]).to_csv(path, index=False)
        df = pd.DataFrame([{"a": 1, "label_rule": "floor", "status": "scored",
                            "correct": 1, "correct_floor": 1}])
        EV.write_outcomes(df, path)
        assert pd.read_csv(path).columns.tolist() == [
            "a", "status", "correct", "label_rule", "correct_floor"]


def _h1_days(statuses):
    n = len(statuses)
    return pd.DataFrame({
        "ticker": [f"T{i}.BK" for i in range(n)], "target_date": ["2026-10-26"] * n,
        "status": ["scored"] * n, "actual_parity_floor": [0.0] * n,
        "h1_status": statuses, "h1_correct": [1.0 if s == "predict" else np.nan for s in statuses],
    })


def test_h1_abstain_note_shown_when_coverage_zero(capsys):
    EV.summarize_h1(_h1_days(["abstain", "abstain"]))
    out = capsys.readouterr().out
    assert "coverage 0/2" in out and EV.H1_ABSTAIN_NOTE in out


def test_h1_abstain_note_hidden_when_coverage_positive(capsys):
    EV.summarize_h1(_h1_days(["predict", "abstain"]))
    out = capsys.readouterr().out
    assert "coverage 1/2" in out and "ถูก 1/1" in out
    assert EV.H1_ABSTAIN_NOTE not in out
