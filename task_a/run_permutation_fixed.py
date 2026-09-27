"""
run_permutation_fixed.py  — แก้ข้อ 3, 4, 5, 6
=============================================
เวอร์ชันแก้บั๊กของ run_permutation.py เดิม  ไม่แตะ test set

สิ่งที่แก้จากของเดิม
--------------------
[ข้อ 3] is_month_end เป็น 1 ทุกแถว
    เดิม: (df.index.to_period("M") != df.index.to_period("M").shift(-1))
    PeriodIndex.shift(-1) เลื่อน *ค่าของ period* ไปหนึ่งเดือน
    ไม่ใช่เลื่อนตำแหน่ง -> ทุกแถวไม่เท่ากันเสมอ -> ได้ 1 ทั้งคอลัมน์
    ตอนนี้ใช้ month_end_flag() ใน tick_utils แทน

[ข้อ 4] target ไม่ตรงกับ pipeline หลัก   <-- ร้ายแรงที่สุด
    เดิม permutation test ใช้ out["y"] = parity (parity ตรง ๆ)
    แต่ pipeline หลักเทรนบน y_flip แล้วค่อย reconstruct
    = กำลังทดสอบคนละอย่างกับที่รายงาน
    ตอนนี้เทรนบน y_flip แล้ว reconstruct กลับเป็น parity
    ด้วย parity_hat = (prev_parity + flip_hat) mod 2 เหมือน pipeline หลัก
    แล้ววัด accuracy บน parity เพื่อให้เทียบกับตารางผลได้ตรง ๆ

    หมายเหตุเรื่องการ permute: สลับ y_flip (target ที่เทรนจริง)
    ไม่ใช่สลับ parity เพราะการสลับต้องทำลายความสัมพันธ์ X-y
    ของสิ่งที่โมเดลเรียน ไม่ใช่ของผลลัพธ์ปลายทาง

[ข้อ 5] ATR ไม่ใช่ ATR
    เดิม: (High - Low).rolling(14).mean()  ไม่นับ gap ระหว่างวัน
    ตอนนี้ใช้ atr_wilder() ตัวเดียวกับ features.py

[ข้อ 6] DEV_FRACTION = 2068/2433 hardcode
    ถ้าโหลดข้อมูลใหม่แล้วจำนวนแถวเปลี่ยน ขอบ test จะเลื่อนโดยไม่รู้ตัว
    ตอนนี้ใช้ dev_cutoff() ที่คำนวณจากสัดส่วนใน config

การอ่านผล
    null mean ~ 0.50        -> pipeline ทำงานถูก ไม่มี leak
    ผลจริงตกกลาง null       -> ไม่มี signal (สรุปได้อย่างมั่นใจ)
    ผลจริง > null p95       -> มี signal หรือมี leak ต้องไปไล่หา

วิธีใช้
    python run_permutation_fixed.py --file KBANK_10Y_Cleaned.csv
    python run_permutation_fixed.py --file ADVANC_10Y_Cleaned.csv --n-perm 200
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from data_adapter import load_investing_csv
from tick_utils import (
    dev_cutoff, tick_size, to_int_baht, parity, atr_wilder, month_end_flag,
    resolve_data_path,
)

warnings.filterwarnings("ignore")
SEED = 42


# =============================================================================
def build_features(df: pd.DataFrame, round_mode: str = "gt") -> pd.DataFrame:
    """
    สร้าง feature หน่วย tick + target

    กติกา: feature ของแถว t ต้องมาจากข้อมูลถึงวัน t-1 เท่านั้น
    วิธี: คำนวณตามปกติ แล้ว shift(1) ทั้งก้อนทีเดียว
    ยกเว้น dow / is_month_end ที่เป็นข้อมูลปฏิทิน รู้ล่วงหน้าได้อยู่แล้ว
    จึงใส่ *หลัง* shift ไม่ต้องเลื่อน (ข้อ 7 ในเช็คลิสต์)
    """
    close = df["Close"].astype(float)
    tick = tick_size(close)
    n_tick = (close / tick).round()
    dtick = n_tick.diff()

    int_price = to_int_baht(close, mode=round_mode)
    y_parity = parity(int_price)

    f = pd.DataFrame(index=df.index)
    f["n_mod2"] = n_tick % 2
    f["n_mod4"] = n_tick % 4
    f["satang"] = (np.rint(close * 100).astype("int64") % 100)
    f["tick_regime"] = tick
    f["parity"] = y_parity
    f["dtick_1"] = dtick
    f["dtick_2"] = dtick.shift(1)
    f["dtick_3"] = dtick.shift(2)
    f["dtick_par"] = dtick.abs() % 2
    f["zero_rate20"] = (dtick == 0).rolling(20).mean()
    f["atr14_ticks"] = atr_wilder(df["High"], df["Low"], close) / tick   # ข้อ 5
    f["std20_ticks"] = dtick.rolling(20).std()
    f["range_ticks"] = (df["High"] - df["Low"]) / tick
    f["vol_ratio"] = np.log(
        df["Volume"].replace(0, np.nan) / df["Volume"].rolling(20).mean()
    )

    out = f.shift(1)
    out["dow"] = df.index.dayofweek                    # ปฏิทิน ไม่ต้อง shift
    out["is_month_end"] = month_end_flag(df.index)     # ข้อ 3
    out["prev_parity"] = y_parity.shift(1)
    out["y_parity"] = y_parity
    out["y_flip"] = (y_parity + y_parity.shift(1)) % 2   # ข้อ 4
    return out.dropna()


def make_model(kind: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    if kind == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=SEED),
        )
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            num_leaves=7, max_depth=3, learning_rate=0.02, n_estimators=400,
            min_child_samples=100, reg_lambda=10.0, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED, verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            max_depth=3, learning_rate=0.02, n_estimators=400,
            random_state=SEED,
        )


def wf_parity_accuracy(X, y_flip, prev_parity, y_parity,
                       kind: str, n_folds: int = 5) -> float:
    """
    walk-forward OOF accuracy — เทรนบน y_flip แล้ว reconstruct เป็น parity

    นี่คือจุดที่แก้ข้อ 4: วัด accuracy บน parity (สิ่งที่รายงาน)
    แต่โมเดลเรียนจาก y_flip (สิ่งที่ pipeline หลักเทรนจริง)
    """
    n = len(y_flip)
    fold_size = n // (n_folds + 1)
    preds, truths = [], []

    for k in range(1, n_folds + 1):
        a, b = fold_size * k, min(fold_size * (k + 1), n)
        if b - a < 30 or len(np.unique(y_flip[:a])) < 2:
            continue
        model = make_model(kind)
        model.fit(X[:a], y_flip[:a])
        flip_hat = model.predict(X[a:b])
        parity_hat = (prev_parity[a:b] + flip_hat) % 2      # reconstruct
        preds.append(parity_hat)
        truths.append(y_parity[a:b])

    if not truths:
        return np.nan
    return float(np.mean(np.concatenate(preds) == np.concatenate(truths)))


# =============================================================================
def permutation_test(data: pd.DataFrame, kind: str, n_perm: int = 100) -> dict:
    feature_cols = [c for c in data.columns
                    if c not in {"y_parity", "y_flip", "prev_parity"}]
    X = data[feature_cols].to_numpy(float)
    y_flip = data["y_flip"].to_numpy(int)
    prev_parity = data["prev_parity"].to_numpy(int)
    y_parity = data["y_parity"].to_numpy(int)

    rng = np.random.default_rng(SEED)

    print(f"\n  [{kind}] ผลจริง ...", end=" ", flush=True)
    observed = wf_parity_accuracy(X, y_flip, prev_parity, y_parity, kind)
    print(f"{observed:.4f}")

    print(f"  [{kind}] สลับ y_flip {n_perm} รอบ ", end="", flush=True)
    null = []
    for i in range(n_perm):
        # สลับ target ที่เทรนจริง แล้วปล่อยให้ reconstruct ตามปกติ
        null.append(wf_parity_accuracy(
            X, rng.permutation(y_flip), prev_parity, y_parity, kind))
        if (i + 1) % 10 == 0:
            print(".", end="", flush=True)
    print(" เสร็จ")

    null = np.array([v for v in null if not np.isnan(v)])
    p_value = float((np.sum(null >= observed) + 1) / (len(null) + 1))

    return {
        "model": kind,
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": p_value,
        "percentile": float(stats.percentileofscore(null, observed)),
        "null": null,
    }


def report(results: list[dict], name: str):
    print("\n" + "=" * 76)
    print(f"  ผล Permutation Test : {name}   (target = y_flip -> parity)")
    print("=" * 76)
    print(f"{'Model':<8}{'Observed':>10}{'Null mean':>12}{'Null SD':>10}"
          f"{'Null p95':>10}{'p-value':>10}{'pctile':>9}")
    for r in results:
        print(f"{r['model']:<8}{r['observed']:>10.4f}{r['null_mean']:>12.4f}"
              f"{r['null_std']:>10.4f}{r['null_p95']:>10.4f}"
              f"{r['p_value']:>10.4f}{r['percentile']:>9.1f}")

    print("\n  การตีความ")
    for r in results:
        if abs(r["null_mean"] - 0.5) > 0.03:
            print(f"    [{r['model']}] !! null mean = {r['null_mean']:.4f} "
                  f"ห่างจาก 0.50 -> ตรวจสอบ pipeline")
        elif r["p_value"] < 0.05:
            print(f"    [{r['model']}] ผลจริงสูงกว่า null อย่างมีนัยสำคัญ "
                  f"(p={r['p_value']:.4f}) -> มี signal หรือมี leak")
        else:
            print(f"    [{r['model']}] ผลจริงอยู่ในช่วงเดียวกับ label สุ่ม "
                  f"(percentile {r['percentile']:.0f}) -> ไม่มี signal, "
                  f"ไม่มี leak")


def plot(results: list[dict], name: str, outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results),
                             figsize=(6 * len(results), 4.5), squeeze=False)
    for ax, r in zip(axes[0], results):
        ax.hist(r["null"], bins=22, color="#9ecae1", edgecolor="white",
                label="Null (shuffled y_flip)")
        ax.axvline(r["observed"], color="#d62728", lw=2.5,
                   label=f"Observed = {r['observed']:.4f}")
        ax.axvline(r["null_p95"], color="#7f7f7f", ls="--", lw=1.5,
                   label=f"Null 95th = {r['null_p95']:.4f}")
        ax.axvline(0.5, color="black", ls=":", lw=1, label="Random = 0.50")
        ax.set_xlabel("Walk-forward OOF parity accuracy")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{name} — {r['model'].upper()}  (p = {r['p_value']:.3f})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"permutation_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] {path}")


# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True,
                    help="เช่น KBANK_10Y_Cleaned.csv")
    ap.add_argument("--out", default="results/permutation")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--models", default="lr,lgbm")
    ap.add_argument("--round-mode", default="gt", choices=["gt", "gte"])
    ap.add_argument("--date-format", default="%m/%d/%Y")
    args = ap.parse_args()

    path = resolve_data_path(args.file)
    name = os.path.basename(path).split("_")[0]
    frame = load_investing_csv(path, date_format=args.date_format,
                               verbose=False)
    data = build_features(frame, round_mode=args.round_mode)

    # ---- ตัด test ออกด้วยสัดส่วนจริง ไม่ hardcode (ข้อ 6) ----
    cut = dev_cutoff(len(data))
    dev = data.iloc[:cut]
    assert dev.index[-1] < data.index[cut], "dev ล้ำเข้า test"

    print("=" * 76)
    print(f"  Permutation Test : {name}")
    print(f"  dev = {len(dev)} จาก {len(data)} แถว "
          f"({dev.index[0].date()} -> {dev.index[-1].date()})")
    print(f"  test ที่กันไว้ = {len(data) - cut} แถว "
          f"(เริ่ม {data.index[cut].date()})  << ไม่แตะ")
    print(f"  round_mode = {args.round_mode} | permutations = {args.n_perm}")
    print("=" * 76)

    res = [permutation_test(dev, k.strip(), args.n_perm)
           for k in args.models.split(",")]
    report(res, name)
    plot(res, name, args.out)

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k != "null"} for r in res]) \
        .to_csv(os.path.join(args.out, f"permutation_{name}.csv"), index=False)
    print(f"[csv] {args.out}/permutation_{name}.csv")
