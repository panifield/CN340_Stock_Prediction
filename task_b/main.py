"""
main.py — งาน B (ราคาปิด / return)
==================================
ไฟล์หลักของงาน B เท่านั้น รันไฟล์นี้ไฟล์เดียวจบ

    cd task_b
    python main.py

โฟลเดอร์นี้เป็นอิสระจาก task_a / task_c โดยสมบูรณ์
(feature / target / โมเดล / config แยกกันคนละชุด)

ลำดับการทำงาน:
  1. โหลดข้อมูล
  2. สร้าง feature (shift แล้ว) + target (return)
  3. ตรวจ leak เชิงโครงสร้าง
  4. วิเคราะห์ข้อมูลก่อนเทรน (ใช้ train เท่านั้น ทุกโหมด)
  5. เทรน 3 โมเดล (ทำนายผ่าน return แล้วแปลงกลับเป็นราคา) + เทียบ baseline
  6. บันทึกผลลง csv
"""

import argparse
import io
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# บน Windows คอนโซลใช้ code page cp1252 เป็นค่าเริ่มต้น พอ print ภาษาไทย
# จะ crash ด้วย UnicodeEncodeError ตั้งแต่บรรทัดแรก บังคับ stdout/stderr
# เป็น UTF-8 เพื่อให้รัน `python main.py` ตรง ๆ ได้โดยไม่ต้องตั้ง env เอง
# (Python 3.7+ มี stream.reconfigure; ห่อ try กันสตรีมที่ไม่รองรับ เช่น
# ตอนถูก redirect เป็นไฟล์/ไปป์บางชนิด)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from config import TICKERS, OUTPUT_DIR, ANN_SEEDS
from data_loader import load_stock
from features import build_features, verify_no_leak
from targets import build_targets
from diagnostics import run_all_diagnostics
from splits import chronological_split
from models import get_regressors
from baselines import get_regression_baselines, always_up_note
from evaluate import (regression_metrics, results_table, print_table,
                      compare_to_baseline, rho_significance_note,
                      diebold_mariano, dm_note)

NAIVE_NAME = "Baseline: Naive (RW)"


def _prepare(X, y, extra=None, verbose=True):
    """
    จัด X และ y ให้ index ตรงกัน แล้วตัดแถวที่ feature ยังคำนวณไม่ครบออก

    *** ทำไมต้องตัดแถวที่มี NaN แม้แต่ตัวเดียว ? (บั๊กที่แก้ในเวอร์ชันนี้) ***

    ของเดิมใช้ `X.isna().mean(axis=1) < 0.5` คือเก็บแถวไว้ตราบใดที่ NaN
    ไม่เกินครึ่ง แล้วปล่อยให้ `SimpleImputer(median)` ใน pipeline เติมให้

    ปัญหาคือ 20 แถวแรกเป็นช่วง "อุ่นเครื่อง" ที่ rolling window ยังไม่ครบ
    (volatility_20d ต้องใช้ 20 วัน, sma20 / bb_position / volume_over_ma20
    ต้องใช้ 19-20 วัน, ret_10d ต้องใช้ 10 วัน) แถวพวกนี้มี NaN ราว 3-44%
    ซึ่งต่ำกว่าเกณฑ์ 0.5 ทุกแถว จึงถูกเก็บไว้ทั้งหมดแล้วโดนเติมด้วย median
    ของทั้งคอลัมน์ = **ยัดค่ากลางที่คำนวณจากทั้งชุดลงไปในแถวที่ ณ เวลานั้น
    ยังไม่มีทางรู้ค่าได้** ทั้งที่ค่าจริงยังไม่เกิดขึ้นด้วยซ้ำ

    ตรวจข้อมูลจริงแล้ว: NaN อยู่แค่ 20 แถวแรกต่อเนื่องกัน (ตำแหน่ง 0-19)
    ทั้งสองหุ้น **ไม่มี NaN กลางชุดเลย** การตัดทิ้งจึงเสียข้อมูลแค่
    20 จาก 2,432 แถว = 0.8% แลกกับการไม่มีค่าที่ถูกเดาขึ้นมาปนในชุดเทรน

    `SimpleImputer` ใน models.py ยังคงไว้เป็นตาข่ายนิรภัย (กลายเป็น no-op)
    เผื่อข้อมูลชุดใหม่มี NaN โผล่มาในอนาคต

    หมายเหตุ: การตัดทำก่อน split เสมอ จึงกระทบแค่ขอบเขตของ train/val/test
    เล็กน้อยตามสัดส่วน ไม่ได้เป็นการเปิดดูหรือใช้ข้อมูล test แต่อย่างใด
    """
    idx = X.index.intersection(y.dropna().index)
    X = X.loc[idx]
    y = y.loc[idx]

    ok = X.notna().all(axis=1)
    n_drop = int((~ok).sum())

    if n_drop and verbose:
        pos = np.where(~ok.values)[0]
        # นับว่าแถวที่ถูกตัดต่อเนื่องจากแถวแรกกี่แถว
        run = 0
        while run < len(pos) and pos[run] == run:
            run += 1

        print(f"[prepare] ตัด {n_drop} แถวที่ feature ยังคำนวณไม่ครบ "
              f"({n_drop / len(ok) * 100:.1f}% ของ {len(ok)} แถว)")
        if run == n_drop:
            print(f"          เป็นช่วงอุ่นเครื่องต้นชุดต่อเนื่องกัน "
                  f"({X.index[0].date()} ถึง {X.index[pos[-1]].date()})")
        else:
            print(f"          !! เตือน: มี {n_drop - run} แถวที่ NaN อยู่กลางชุด "
                  f"(แถวที่ {pos[run:run + 5].tolist()} ...)")
            print(f"          ตรวจ features.py ว่ามีตัวหารเป็นศูนย์หรือไม่ "
                  f"ก่อนเชื่อผลลัพธ์")

    X, y = X[ok], y[ok]

    if extra is not None:
        extra = extra.loc[X.index]
        return X, y, extra
    return X, y


def run_task_b(X, targets, verbose=True, dev=False):
    print("\n" + "=" * 78)
    print("  งาน B : ทำนายราคาปิด (ทำนายผ่าน return แล้วแปลงกลับ)")
    if dev:
        print("  (โหมด dev — ยังไม่แตะ test)")
    print("=" * 78)

    y = targets["y_return"]
    extra = targets[["prev_close", "close"]]
    X_, y_, extra_ = _prepare(X, y, extra)

    parts = chronological_split(X_, y_, verbose=verbose, name="งาน B")
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]
    prev_close_val = extra_.loc[X_val.index, "prev_close"]

    # วิเคราะห์ข้อมูลก่อนเทรน -- ใช้ "train เท่านั้น" ทุกโหมด
    # เดิมโหมดปกติรันบนข้อมูลทั้งหมดรวม test ทำให้สถิติของ test (Return SD,
    # mean return, corr กับ target) หลุดเข้าไฟล์รายงาน ส่วนโหมด dev ตัดด้วย
    # สัดส่วนของ df ดิบ ซึ่งไม่ตรงกับจุดตัดจริงหลัง _prepare (2068 vs 2071)
    diag = run_all_diagnostics(targets.loc[X_train.index], X_train)

    if not dev:
        X_test, y_test = parts["test"]
        prev_close_test = extra_.loc[X_test.index, "prev_close"]

    # ห่อด้วย TransformedTargetRegressor เพื่อ scale target
    # (return มีขนาดเล็กมาก ~0.01 ANN จะเทรนไม่ดีถ้าไม่ scale)
    val_results, val_preds = {}, {}
    test_results, test_preds = {}, {}

    for name, model in get_regressors().items():
        if verbose:
            print(f"    เทรน {name} ...", end=" ", flush=True)
        wrapped = TransformedTargetRegressor(
            regressor=model, transformer=StandardScaler()
        )
        wrapped.fit(X_train, y_train)

        y_val_pred = wrapped.predict(X_val)
        val_results[name] = regression_metrics(y_val, y_val_pred, prev_close_val)
        val_preds[name] = y_val_pred

        if not dev:
            y_test_pred = wrapped.predict(X_test)
            test_results[name] = regression_metrics(y_test, y_test_pred, prev_close_test)
            test_preds[name] = y_test_pred

        if verbose:
            if dev:
                print(f"เสร็จ (val MAE_return={val_results[name]['MAE_return']:.6f}, "
                      f"MAE_baht={val_results[name]['MAE_baht']:.4f} บาท)")
            else:
                print(f"เสร็จ (val MAE_return={val_results[name]['MAE_return']:.6f}, "
                      f"test MAE_return={test_results[name]['MAE_return']:.6f})")

    # เลือกด้วย MAE_return ไม่ใช่ MAE_baht: MAE_baht คูณด้วยระดับราคาซึ่งไต่ขึ้น
    # ตลอดช่วง val ทำให้วันท้ายชุดมีน้ำหนักมากกว่าวันต้นชุดโดยไม่มีเหตุผล
    # (เหตุผลเดียวกับที่ feature_selection.py ใช้ MAE_return)
    best = min(val_results, key=lambda n: val_results[n]["MAE_return"])
    if verbose:
        print(f"\n    เลือกโมเดลที่ดีที่สุดจาก val set: {best} "
              f"(val MAE_return={val_results[best]['MAE_return']:.6f})")

    if dev:
        eval_split, y_eval, eval_results, eval_preds, prev_close_eval = (
            "val", y_val, val_results, val_preds, prev_close_val
        )
    else:
        eval_split, y_eval, eval_results, eval_preds, prev_close_eval = (
            "test", y_test, test_results, test_preds, prev_close_test
        )

    for name, p in get_regression_baselines(y_train, y_eval).items():
        eval_results[name] = regression_metrics(y_eval, p, prev_close_eval)
        eval_preds[name] = p

    # Diebold-Mariano เทียบทุกตัวกับ Naive (RW) -- ใส่เป็นคอลัมน์ใน csv ด้วย
    # (Naive เทียบกับตัวเองได้ NaN)
    naive = eval_preds[NAIVE_NAME]
    for name in eval_results:
        z, p = diebold_mariano(y_eval, eval_preds[name], naive)
        eval_results[name]["DM_z"] = z
        eval_results[name]["DM_p"] = p

    df = results_table(eval_results, sort_by="MAE_baht", ascending=True)
    label = "Val Set (โหมด dev)" if dev else "Test Set"
    print_table(df, f"งาน B : ผลลัพธ์บน {label}")

    print("\n" + compare_to_baseline(df, "MAE_baht", higher_is_better=False,
                                     model_name=best))
    print("\n" + dm_note(eval_results, NAIVE_NAME))

    print("\n  หมายเหตุการอ่านผล:")
    print(f"  - ANN (MLP) = ค่าเฉลี่ยการทำนายของ {len(ANN_SEEDS)} seeds "
          f"{ANN_SEEDS} น้ำหนักเท่ากัน")
    print("    ไม่ได้เลือก seed ที่ดีที่สุด (ดูเหตุผลที่ ANN_SEEDS ใน config.py)")
    print("  - ทุกโมเดลเทรนบน train เท่านั้น ไม่ refit รวม val "
          "(ตัดสินใจไว้ก่อนเปิด test -- ดู config.py)")
    print("  - เลือกโมเดลที่ดีที่สุดด้วย val MAE_return "
          "(MAE_baht ถ่วงน้ำหนักตามระดับราคา)")
    print("  - R2_price ที่สูงมาก (>0.95) ไม่ได้แปลว่าโมเดลเก่ง")
    print("    เพราะมันมาจากการที่ราคาพรุ่งนี้ใกล้เคียงราคาวันนี้อยู่แล้ว")
    print("  - ให้ดู MAE_baht เทียบกับ Baseline: Naive (RW) เป็นหลัก")
    flat = float(eval_results[best]["FlatRate"])
    print(f"  - DirAcc นับเฉพาะวันที่ราคาขยับจริง (ตัดวันราคานิ่ง "
          f"{flat*100:.2f}% ออก)")
    print(f"    เพดานสูงสุดคือ 100% ของวันที่นับ ไม่ใช่ของทั้งชุด")
    print("  - Baseline: Naive (RW) ได้ DirAcc = NaN เพราะทำนาย return = 0")
    print("    เสมอ จึงไม่ได้ให้สัญญาณทิศทาง -> ใช้ Always Up เทียบแทน")
    print("  - StdRatio / Rho / Bias แตก R2_return ออกเป็นส่วน ๆ ได้พอดี:")
    print("        R2 = 2*Rho*StdRatio - StdRatio^2 - Bias^2")
    print("    Rho คือทักษะจริง (เพดานของ R2 หลังปรับเทียบคือ Rho^2)")
    print("    StdRatio ที่เหมาะสมคือเท่ากับ Rho ถ้าสูงกว่ามาก = ทำนายแกว่งเกินจริง")
    print("    Bias สูง = train กับชุดที่ประเมินมี distribution ต่างกัน")
    print(f"  - Rho_lo / Rho_hi คือช่วงความเชื่อมั่น 95% ของ Rho "
          f"(n = {len(y_eval)} วัน)")
    print(rho_significance_note(eval_results, stage=eval_split))
    print(always_up_note(y_train))

    best_rmse = float(eval_results[best]["RMSE_baht"])

    return {"table": df, "preds": eval_preds, "y_test": y_eval,
            "prev_close_test": prev_close_eval,
            "best_rmse_baht": best_rmse,
            "test_index": y_eval.index, "best_model": best, "stage": eval_split,
            "n_features": X_.shape[1], "diag": diag}


def run_one_ticker(ticker, dev=False):
    print("\n\n" + "#" * 78)
    print(f"#  หุ้น: {ticker}  (งาน B)")
    if dev:
        print("#  โหมด dev — ใช้แค่ train/val เพื่อพัฒนา ยังไม่แตะ test")
    print("#" * 78)

    df = load_stock(ticker)
    X = build_features(df)
    targets = build_targets(df)

    verify_no_leak(df, X, sample_idx=100)

    # diagnostics รันใน run_task_b หลัง split เพื่อให้ใช้แถวของ train เท่านั้น
    res = run_task_b(X, targets, dev=dev)
    return {"ticker": ticker, "diag": res["diag"], "b": res}


class _Tee:
    """
    เขียนออกจอตามปกติ และเก็บสำเนาทุกตัวอักษรไว้ด้วย
    ใช้ทำไฟล์รายงาน (.txt) ที่หน้าตาเหมือนตอนรันบนจอทุกประการ --
    ทั้ง diagnostics, การแบ่งข้อมูล, ตาราง, ผลเทียบ baseline และหมายเหตุ
    """

    def __init__(self, stream):
        self.stream = stream
        self.copy = io.StringIO()

    def write(self, s):
        self.stream.write(s)
        self.copy.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        # attribute อื่น (encoding, isatty ฯลฯ) ส่งต่อให้สตรีมจริง
        return getattr(self.stream, name)


def save_results(all_results, timestamp):
    """
    บันทึกผลลัพธ์ 2 แบบ

    1) ตารางตัวเลขของแต่ละหุ้นเป็น csv (เอาไปทำกราฟ/ตารางต่อได้)
       ชื่อไฟล์: {หุ้น}_taskB_{stage}_{n}feat_{timestamp}.csv
    2) รายงานฉบับเต็มแบบเดียวกับที่ขึ้นบนจอ (อ่านง่าย ใช้ทำสไลด์)
       ชื่อไฟล์: taskB_{stage}_{n}feat_{timestamp}_report.txt
       -- ไฟล์นี้ main() เป็นคนเขียนตอนจบ เพื่อให้เก็บได้ครบถึงบรรทัดสุดท้าย

    stage     = "val" (โหมด --dev) หรือ "test"
                อ่านจากชุดข้อมูลที่ใช้ประเมิน "จริง" (res["stage"]) ไม่ได้อ่าน
                จาก flag --dev จึงไม่มีทางที่ผลของ val จะถูกตั้งชื่อเป็น test
                หรือกลับกัน
    n feat    = จำนวน feature ที่ใช้ -- กันสับสนระหว่างผลก่อน/หลัง ablation H1
    timestamp = เวลาที่รัน ไฟล์เก่าไม่ถูกเขียนทับ ย้อนดูได้เสมอว่าตัวเลข
                บนสไลด์มาจากการรันครั้งไหน

    คืน (รายการไฟล์ csv, path ของไฟล์รายงาน)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = []
    for r in all_results:
        t = r["ticker"].replace(".", "_").replace("^", "")
        b = r["b"]
        name = f"{t}_taskB_{b['stage']}_{b['n_features']}feat_{timestamp}.csv"
        path = os.path.join(OUTPUT_DIR, name)
        b["table"].to_csv(path, encoding="utf-8-sig")
        paths.append(path)

    b0 = all_results[0]["b"]
    report_path = os.path.join(
        OUTPUT_DIR,
        f"taskB_{b0['stage']}_{b0['n_features']}feat_{timestamp}_report.txt",
    )

    print(f"\n[main] บันทึกผลลัพธ์ {len(paths) + 1} ไฟล์:")
    for p in paths:
        print(f"         {p}")
    print(f"         {report_path}   <- รายงานฉบับเต็มแบบเดียวกับบนจอ")
    return paths, report_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="รันงาน B (ราคาปิด) — ปกติจะแตะ test ครั้งเดียวตอนจบ"
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="โหมดพัฒนา: เทรน+ประเมินบน train/val เท่านั้น ไม่แตะ test "
             "บันทึกผลเป็น *_taskB_val_*.csv และรายงาน *_report.txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # เก็บสำเนาทุกอย่างที่ print ระหว่างรัน เพื่อเขียนเป็นไฟล์รายงานตอนจบ
    tee = _Tee(sys.stdout)
    sys.stdout = tee
    report_path = None

    try:
        print("=" * 78)
        print("  งาน B : ทำนายราคาปิด (Regression)")
        if args.dev:
            print("  *** โหมด dev: ไม่แตะ test, บันทึกผล val เป็น csv + รายงาน ***")
        print("=" * 78)

        # เวลาเดียวกันทุกหุ้นในการรันครั้งนี้ -> ไฟล์จากรอบเดียวกันจับคู่กันได้
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"  เวลาที่รัน: {stamp}")

        all_results = []
        for ticker in TICKERS:
            try:
                all_results.append(run_one_ticker(ticker, dev=args.dev))
            except Exception as e:
                print(f"\n!! {ticker} รันไม่ผ่าน: {type(e).__name__}: {e}")
                import traceback
                # พิมพ์ลง stdout เพื่อให้ error ติดไปในไฟล์รายงานด้วย
                traceback.print_exc(file=sys.stdout)

        if all_results:
            _, report_path = save_results(all_results, stamp)

        print("\nเสร็จสิ้น")
    finally:
        sys.stdout = tee.stream

    if report_path is not None:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(tee.copy.getvalue())

    return all_results


if __name__ == "__main__":
    main()
