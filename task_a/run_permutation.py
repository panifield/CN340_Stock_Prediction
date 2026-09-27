"""
run_permutation.py
==================
Permutation test แบบ standalone - รันคำสั่งเดียวจบ ไม่ต้องแก้ pipeline เดิม

ตอบคำถาม: "ถ้าข้อมูลไม่มี signal เลย โมเดลจะได้ accuracy เท่าไหร่?"

วิธีการ:
  1. สลับ label แบบสุ่ม -> ทำลายความสัมพันธ์ X-y ทั้งหมด
  2. เทรนโมเดลใหม่ วัด accuracy
  3. ทำซ้ำ N รอบ -> ได้ null distribution
  4. เอาผลจริงไปเทียบว่าอยู่ตรงไหน

อ่านผล:
  null mean ≈ 0.50           -> pipeline ทำงานถูก
  ผลจริงตกกลาง null          -> ไม่มี signal จริง (สรุปได้มั่นใจ)
  ผลจริง > null 95th pct     -> มี signal หรือมี leak ต้องไปไล่หา

การใช้งาน
--------
    python run_permutation.py --ticker KBANK.BK --cache-dir data_cache
    python run_permutation.py --ticker ADVANC.BK --n-perm 200
    python run_permutation.py --demo
"""

from __future__ import annotations

import argparse
import os
import warnings
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
SEED = 42

# สัดส่วน dev (train+val) เทียบข้อมูลทั้งหมด - กัน test set
DEV_FRACTION = 2068 / 2433


# =============================================================================
def tick_size(p: float) -> float:
    if p < 2:   return 0.01
    if p < 5:   return 0.02
    if p < 10:  return 0.05
    if p < 25:  return 0.10
    if p < 100: return 0.25
    if p < 200: return 0.50
    if p < 400: return 1.00
    return 2.00


def round_half_up(x: float) -> int:
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """16 features หน่วย tick, shift(1) ทั้งหมด (เหมือน pipeline หลัก)"""
    close = df["Close"].astype(float)
    tick = close.apply(tick_size)
    n_tick = (close / tick).round().astype(int)
    dtick = n_tick.diff()
    parity = close.apply(round_half_up) % 2

    f = pd.DataFrame(index=df.index)
    f["n_mod2"] = n_tick % 2
    f["n_mod4"] = n_tick % 4
    f["satang"] = (close * 100).round().astype(int) % 100
    f["tick_regime"] = tick
    f["parity"] = parity
    f["dtick_1"] = dtick
    f["dtick_2"] = dtick.shift(1)
    f["dtick_3"] = dtick.shift(2)
    f["dtick_par"] = dtick.abs() % 2
    f["zero_rate20"] = (dtick == 0).rolling(20).mean()
    f["atr14_ticks"] = (df["High"] - df["Low"]).rolling(14).mean() / tick
    f["std20_ticks"] = dtick.rolling(20).std()
    f["range_ticks"] = (df["High"] - df["Low"]) / tick
    f["vol_ratio"] = np.log(df["Volume"].replace(0, np.nan)
                            / df["Volume"].rolling(20).mean())

    out = f.shift(1)
    out["dow"] = df.index.dayofweek
    out["is_month_end"] = (df.index.to_period("M")
                           != df.index.to_period("M").shift(-1)).astype(int)
    out["y"] = parity
    return out.dropna()


def make_model(kind: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    if kind == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000,
                                                random_state=SEED))
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(num_leaves=7, max_depth=3, learning_rate=0.02,
                              n_estimators=400, min_child_samples=100,
                              reg_lambda=10.0, subsample=0.8,
                              colsample_bytree=0.8, random_state=SEED,
                              verbose=-1)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(max_depth=3, learning_rate=0.02,
                                          n_estimators=400,
                                          random_state=SEED)


def wf_accuracy(X, y, n_folds: int = 5) -> float:
    """walk-forward OOF accuracy (ใช้ทั้งกับ label จริงและ label สลับ)"""
    n = len(y)
    fs = n // (n_folds + 1)
    P, Y = [], []
    for k in range(1, n_folds + 1):
        a, b = fs * k, min(fs * (k + 1), n)
        if b - a < 30 or len(np.unique(y[:a])) < 2:
            continue
        m = make_model(wf_accuracy.kind)
        m.fit(X[:a], y[:a])
        P.append(m.predict(X[a:b])); Y.append(y[a:b])
    if not Y:
        return np.nan
    return float(np.mean(np.concatenate(P) == np.concatenate(Y)))


wf_accuracy.kind = "lgbm"


# =============================================================================
def permutation_test(X, y, kind: str, n_perm: int = 100) -> dict:
    wf_accuracy.kind = kind
    rng = np.random.default_rng(SEED)

    print(f"\n  [{kind}] คำนวณผลจริง ...", end=" ", flush=True)
    observed = wf_accuracy(X, y)
    print(f"{observed:.4f}")

    print(f"  [{kind}] สลับ label {n_perm} รอบ ", end="", flush=True)
    null = []
    for i in range(n_perm):
        null.append(wf_accuracy(X, rng.permutation(y)))
        if (i + 1) % 10 == 0:
            print(".", end="", flush=True)
    print(" เสร็จ")

    null = np.array([v for v in null if not np.isnan(v)])
    p = float((np.sum(null >= observed) + 1) / (len(null) + 1))

    return {"model": kind, "observed": observed,
            "null_mean": float(null.mean()), "null_std": float(null.std()),
            "null_p95": float(np.percentile(null, 95)),
            "p_value": p,
            "percentile": float(stats.percentileofscore(null, observed)),
            "null": null}


def report(results: list[dict], name: str):
    print("\n" + "=" * 74)
    print(f"  ผล Permutation Test : {name}")
    print("=" * 74)
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
                  f"ห่างจาก 0.50 มาก -> ตรวจสอบ pipeline")
        elif r["p_value"] < 0.05:
            print(f"    [{r['model']}] ผลจริงสูงกว่า null อย่างมีนัยสำคัญ "
                  f"(p={r['p_value']:.4f}) -> มี signal หรือมี leak")
        else:
            print(f"    [{r['model']}] ผลจริงอยู่ในช่วงเดียวกับ label สุ่ม "
                  f"(percentile {r['percentile']:.0f}) -> ไม่มี signal, "
                  f"pipeline ไม่มี leak")


def plot(results: list[dict], name: str, outdir: str):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.5),
                             squeeze=False)
    for ax, r in zip(axes[0], results):
        ax.hist(r["null"], bins=22, color="#9ecae1", edgecolor="white",
                label="Null (shuffled labels)")
        ax.axvline(r["observed"], color="#d62728", lw=2.5,
                   label=f"Observed = {r['observed']:.4f}")
        ax.axvline(r["null_p95"], color="#7f7f7f", ls="--", lw=1.5,
                   label=f"Null 95th = {r['null_p95']:.4f}")
        ax.axvline(0.5, color="black", ls=":", lw=1, label="Random = 0.50")
        ax.set_xlabel("Walk-forward OOF Accuracy")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{name} — {r['model'].upper()}  (p = {r['p_value']:.3f})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"permutation_{name.replace('.', '_')}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[plot] บันทึก: {png}")


def make_demo(price=155.0, n=2400, vol=0.015) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2016-08-26", periods=n)
    tk = tick_size(price)
    step = max(1.0, price * vol / tk)
    k = round(price / tk); ks = []
    for _ in range(n):
        k = max(1, k + int(round(rng.normal(0, step)))); ks.append(k)
    c = np.array(ks) * tk
    return pd.DataFrame({"Close": c,
                         "High": c + tk * rng.integers(0, 3, n),
                         "Low": c - tk * rng.integers(0, 3, n),
                         "Volume": rng.integers(1e6, 1e7, n)}, index=dates)


# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="KBANK.BK")
    ap.add_argument("--cache-dir", default="data_cache")
    ap.add_argument("--out", default="results/permutation")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--models", default="lr,lgbm")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if a.demo:
        df, name = make_demo(), "DEMO"
    else:
        p = os.path.join(a.cache_dir,
                         f"{a.ticker.replace('.', '_')}_2016-08-26_2026-08-28.csv")
        if not os.path.exists(p):
            raise SystemExit(f"ไม่พบไฟล์: {p}")
        df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")
        name = a.ticker

    data = build_features(df)
    cut = int(len(data) * DEV_FRACTION)          # <-- กัน test set
    data = data.iloc[:cut]
    X = data.drop(columns="y").to_numpy(float)
    y = data["y"].to_numpy(int)

    print("=" * 74)
    print(f"  Permutation Test : {name}")
    print(f"  n = {len(y)} แถว (dev เท่านั้น, ตัด test ออกแล้ว)")
    print(f"  features = {X.shape[1]}   permutations = {a.n_perm}")
    print("=" * 74)

    res = [permutation_test(X, y, k.strip(), a.n_perm)
           for k in a.models.split(",")]
    report(res, name)
    plot(res, name, a.out)

    pd.DataFrame([{k: v for k, v in r.items() if k != "null"} for r in res]) \
        .to_csv(os.path.join(a.out, f"permutation_{name.replace('.','_')}.csv"),
                index=False)
