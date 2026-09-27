"""
improve_task_a.py
=================
3 วิธีเพิ่มผลลัพธ์งาน A โดยไม่ผิดกติกาและไม่ใช่ p-hacking

  [1.1] intraday horizon  - ใช้ราคาเปิดวัน t (ลดระยะทำนายจาก 24 ชม. เหลือ 6.5 ชม.)
  [1.2] digit clustering  - ทดสอบว่าหลักหน่วยบาทกระจุกที่เลข 0/5 หรือไม่
  [1.3] selective predict - ทำนายเฉพาะวันที่มั่นใจ + risk-coverage curve

ทั้งสามอย่างให้ผลที่เขียนรายงานได้ ไม่ว่าผลจะออกบวกหรือลบ

การใช้งาน
--------
    python improve_task_a.py --ticker KBANK.BK --cache-dir data_cache
    python improve_task_a.py --demo
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from tick_utils import dev_cutoff, month_end_flag, tick_size, to_int_baht

warnings.filterwarnings("ignore")
SEED = 42


# =============================================================================
# helper — tick_size / to_int_baht มาจาก tick_utils.py (แหล่งความจริงเดียว)
# แล้ว re-export ต่อจาก rounding.py จริงๆ
#
# *** เดิมไฟล์นี้นิยาม round_half_up() เองด้วย Decimal ROUND_HALF_UP ***
# ซึ่งปัด 142.50 -> 143 (ปัดขึ้นเสมอเมื่อเจอ .50 พอดี) แต่กฎอาจารย์คือ
# "เศษ > 0.5 ปัดขึ้น" (ต้อง > เท่านั้น) -> 142.50 ต้องได้ 142 ไม่ใช่ 143
# ผิดกับที่ rounding.py ทั้งโปรเจกต์ยึดถือ ราคาหุ้นพวกนี้ลงท้าย .50 บ่อย
# มาก (KBANK ~41% ของวัน) การปัดผิดจุดนี้เปลี่ยน parity label ของเกือบ
# ครึ่งข้อมูลได้เลย แก้โดยใช้ to_int_baht() จาก rounding.py ตรงๆ แทน
# =============================================================================


# =============================================================================
# [1.2] Digit clustering test
# =============================================================================

def digit_analysis(close: pd.Series, label: str = "") -> dict:
    """
    ทดสอบว่าหลักหน่วยบาทของราคาปิดกระจุกที่เลขใดเลขหนึ่งหรือไม่

    ทฤษฎีรองรับ: price clustering / round-number effect
    ถ้าเลขลงท้าย 0 พบบ่อยกว่าเลขลงท้าย 5 อย่างมีนัยสำคัญ
    -> เป็น bias ทางเลข "คู่" ที่มีอยู่จริง ใช้เป็น prior ได้

    ผลลัพธ์มีค่าทั้งสองทาง:
      เจอ     -> ได้ feature ใหม่ที่มีทฤษฎีรองรับ
      ไม่เจอ  -> ได้ประโยคในรายงานว่าทดสอบสมมติฐานนี้แล้ว
    """
    ip = to_int_baht(close).astype(int)
    digit = (ip % 10).to_numpy()
    n = len(digit)

    counts = np.bincount(digit, minlength=10)
    expected = n / 10

    # chi-square goodness of fit: หลักหน่วยกระจายสม่ำเสมอไหม
    chi2, p_uniform = stats.chisquare(counts)

    # เลขคู่ vs คี่
    n_even = int(np.sum(digit % 2 == 0))
    p_even = stats.binomtest(n_even, n, 0.5).pvalue

    # เลขลงท้าย 0/5 (round number) เทียบค่าคาดหวัง 0.20
    n_round = int(np.sum(np.isin(digit, [0, 5])))
    p_round = stats.binomtest(n_round, n, 0.2).pvalue

    # 0 vs 5 โดยเฉพาะ (ถ้า 0 มากกว่า 5 -> bias ทางคู่)
    n0, n5 = int(counts[0]), int(counts[5])
    p_0v5 = (stats.binomtest(n0, n0 + n5, 0.5).pvalue
             if (n0 + n5) > 0 else np.nan)

    print(f"\n--- [1.2] Digit clustering {label} ---")
    print("  หลักหน่วยบาท : ", end="")
    for d in range(10):
        print(f"{d}={counts[d]/n*100:.1f}% ", end="")
    print(f"\n  (คาดหวังถ้าสม่ำเสมอ = 10.0% ทุกตัว, n={n})")
    print(f"  chi-square uniform : chi2={chi2:.1f}  p={p_uniform:.4f} "
          f"{'<-- ไม่สม่ำเสมอ' if p_uniform < 0.05 else '(สม่ำเสมอ)'}")
    print(f"  เลขคู่             : {n_even/n:.4f}  p={p_even:.4f}")
    print(f"  ลงท้าย 0 หรือ 5    : {n_round/n:.4f}  (คาดหวัง 0.2000)  p={p_round:.4f}")
    print(f"  ลงท้าย 0 vs 5      : {n0} vs {n5}  p={p_0v5:.4f}")

    if p_uniform < 0.05 and p_even < 0.05:
        print("  => พบ clustering ที่ทำให้เกิด bias ทางคู่/คี่ -> ใช้เป็น feature ได้")
    elif p_uniform < 0.05:
        print("  => พบ clustering แต่ไม่ทำให้เกิด bias คู่/คี่")
    else:
        print("  => ไม่พบ clustering ที่หลักหน่วยบาท")

    return {"counts": counts, "p_uniform": float(p_uniform),
            "even_rate": n_even / n, "p_even": float(p_even),
            "round_rate": n_round / n, "p_round": float(p_round),
            "p_0v5": float(p_0v5)}


# =============================================================================
# [1.1] Feature builder - เทียบ 2 horizon
# =============================================================================

def build_features(df: pd.DataFrame, use_open: bool = False) -> pd.DataFrame:
    """
    use_open=False : ใช้ข้อมูลถึงวัน t-1 (เหมือน pipeline เดิม, horizon ~24 ชม.)
    use_open=True  : เพิ่มราคาเปิดวัน t (horizon ~6.5 ชม.)

    *** ราคาเปิดวัน t ไม่ใช่ leakage ***
    ถ้ากรอบโจทย์คือ "ทำนายตอน 10:00 น. ของวันนั้น"
    ราคาเปิดเป็นข้อมูลที่มีจริงแล้ว ณ เวลานั้น
    ต้องเขียนกรอบเวลาให้ชัดในรายงาน
    """
    close = df["Close"].astype(float)
    tick = tick_size(close)
    n_tick = (close / tick).round().astype(int)
    dtick = n_tick.diff()
    ip = to_int_baht(close).astype(int)
    parity = ip % 2

    f = pd.DataFrame(index=df.index)
    f["n_mod2"] = n_tick % 2
    f["n_mod4"] = n_tick % 4
    f["satang"] = ((close * 100).round().astype(int) % 100)
    f["last_digit"] = ip % 10                         # <-- จาก [1.2]
    f["is_round_digit"] = f["last_digit"].isin([0, 5]).astype(int)
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

    out = f.shift(1)                                  # <-- ข้อมูลถึง t-1
    out["dow"] = df.index.dayofweek
    # เดิมใช้ PeriodIndex.shift(-1) ซึ่งเลื่อน "ค่า" period ไม่ใช่ตำแหน่ง
    # ทำให้ทุกแถวไม่เท่ากันเสมอ -> ได้ 1 ทั้งคอลัมน์ (บั๊ก) แก้ด้วย
    # month_end_flag() ที่ tick_utils.py ซึ่งพิสูจน์แล้วว่าถูกต้อง
    out["is_month_end"] = month_end_flag(df.index)

    if use_open and "Open" in df.columns:
        op = df["Open"].astype(float)
        op_tick = tick_size(op)
        op_n = (op / op_tick).round().astype(int)
        # ไม่ shift - ข้อมูลของวัน t เอง
        out["open_n_mod2"] = op_n % 2
        out["open_parity"] = to_int_baht(op).astype(int) % 2
        out["open_satang"] = ((op * 100).round().astype(int) % 100)
        out["gap_ticks"] = (op - close.shift(1)) / op_tick

    out["y"] = parity
    out["_y_prev"] = parity.shift(1)
    out["_close"] = close
    return out.dropna()


# =============================================================================
# Walk-forward ที่คืนค่า probability ด้วย (จำเป็นสำหรับ [1.3])
# =============================================================================

def walk_forward_proba(data: pd.DataFrame, n_folds: int = 5) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    cols = [c for c in data.columns if not c.startswith("_") and c != "y"]
    X = data[cols].to_numpy(float)
    y = data["y"].to_numpy(int)
    y_prev = data["_y_prev"].to_numpy(int)
    n = len(y)
    fs = n // (n_folds + 1)

    P, PRED, YT, YP = [], [], [], []
    for k in range(1, n_folds + 1):
        a, b = fs * k, min(fs * (k + 1), n)
        if b - a < 30 or len(np.unique(y[:a])) < 2:
            continue
        try:
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(num_leaves=7, max_depth=3, learning_rate=0.02,
                               n_estimators=400, min_child_samples=100,
                               reg_lambda=10.0, subsample=0.8,
                               colsample_bytree=0.8, random_state=SEED,
                               verbose=-1)
        except ImportError:
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000,
                                                 random_state=SEED))
        m.fit(X[:a], y[:a])
        P.append(m.predict_proba(X[a:b])[:, 1])
        PRED.append(m.predict(X[a:b]))
        YT.append(y[a:b]); YP.append(y_prev[a:b])

    yt = np.concatenate(YT)
    return {"y": yt, "proba": np.concatenate(P), "pred": np.concatenate(PRED),
            "y_prev": np.concatenate(YP), "n": len(yt),
            "acc": float(np.mean(np.concatenate(PRED) == yt)),
            "acc_persistence": float(np.mean(np.concatenate(YP) == yt))}


# =============================================================================
# [1.3] Selective prediction / risk-coverage
# =============================================================================

def risk_coverage(y, proba, label: str = "") -> pd.DataFrame:
    """
    ทำนายเฉพาะวันที่โมเดลมั่นใจที่สุด แล้วดูว่า accuracy ขึ้นไหม

    ถ้ามี signal แม้น้อย -> accuracy จะไต่ขึ้นเมื่อ coverage ลดลง
    ถ้าเส้นแบนที่ 0.50   -> ยืนยันว่า probability ไม่มีความหมายเลย

    เป็นกราฟที่ใช้ได้ในรายงานไม่ว่าผลจะออกทางไหน
    """
    y = np.asarray(y); proba = np.asarray(proba)
    conf = np.abs(proba - 0.5)
    pred = (proba >= 0.5).astype(int)

    rows = []
    for cov in [1.00, 0.75, 0.50, 0.30, 0.20, 0.10, 0.05, 0.02]:
        thr = np.quantile(conf, 1 - cov)
        m = conf >= thr
        if m.sum() < 10:
            continue
        k = int(np.sum(pred[m] == y[m])); nn = int(m.sum())
        acc = k / nn
        # 95% CI แบบ Wilson
        z = 1.96
        den = 1 + z**2 / nn
        c = (acc + z**2 / (2 * nn)) / den
        h = z * np.sqrt(acc * (1 - acc) / nn + z**2 / (4 * nn**2)) / den
        rows.append({"coverage": cov, "n": nn, "accuracy": round(acc, 4),
                     "ci_low": round(max(0, c - h), 4),
                     "ci_high": round(min(1, c + h), 4),
                     "p_vs_0.50": round(
                         float(stats.binomtest(k, nn, 0.5).pvalue), 4)})

    out = pd.DataFrame(rows)
    print(f"\n--- [1.3] Selective prediction {label} ---")
    print(out.to_string(index=False))

    # แยกทิศทาง — ห้ามใช้ idxmax(accuracy) เฉยๆ เพราะจะหยิบแถวที่
    # accuracy สูงสุด "ไม่ว่าจะนัยสำคัญหรือไม่" มารายงานเป็น "ดีสุด"
    # ทั้งที่แถวที่ significant จริงอาจเป็นแถวอื่นที่ accuracy ต่ำกว่า 0.50
    sig = out[out["p_vs_0.50"] < 0.05]
    sig_above = sig[sig["accuracy"] > 0.5]
    sig_below = sig[sig["accuracy"] < 0.5]

    if len(sig_above):
        print("  => พบ coverage ที่ 'ดีกว่าสุ่ม' อย่างมีนัยสำคัญ:")
        print("     " + sig_above[["coverage", "n", "accuracy", "p_vs_0.50"]]
             .to_string(index=False).replace("\n", "\n     "))
        print("     *** ระวัง: ทดสอบหลาย coverage = multiple comparison")
        print("         ต้องแก้ด้วย Bonferroni/FDR ก่อนสรุป ***")
    if len(sig_below):
        print("  => พบ coverage ที่ 'แย่กว่าสุ่มอย่างเป็นระบบ' อย่างมีนัยสำคัญ:")
        print("     " + sig_below[["coverage", "n", "accuracy", "p_vs_0.50"]]
             .to_string(index=False).replace("\n", "\n     "))
        print("     *** ระวัง: ทดสอบหลาย coverage = multiple comparison "
              "ต้องแก้ก่อนสรุปเช่นกัน ***")
    if not len(sig_above) and not len(sig_below):
        print("  => ทุกระดับ coverage ไม่ต่างจากการเดา "
              "-> ยืนยันว่า probability ไม่มีข้อมูล")
    return out


def plot_risk_coverage(tables: dict, out_png: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2b6cb0", "#d62728", "#2ca02c", "#ff7f0e"]
    for (name, t), c in zip(tables.items(), colors):
        ax.plot(t["coverage"] * 100, t["accuracy"], "o-", color=c,
                lw=2, ms=6, label=name)
        ax.fill_between(t["coverage"] * 100, t["ci_low"], t["ci_high"],
                        color=c, alpha=0.15)
    ax.axhline(0.5, color="black", ls=":", lw=1, label="Random = 0.50")
    ax.invert_xaxis()
    ax.set_xlabel("Coverage (% of days predicted)")
    ax.set_ylabel("Accuracy on predicted days")
    ax.set_title("Risk-Coverage Curve\n(if signal exists, accuracy rises as coverage falls)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n[plot] บันทึก: {out_png}")
    return fig


# =============================================================================
# main
# =============================================================================

def run(df: pd.DataFrame, name: str, outdir: str):
    print("=" * 78)
    print(f"  {name}")
    print("=" * 78)

    # *** ตัด test set ออกก่อนทำอะไรทั้งหมด ***
    # เดิมไฟล์นี้ส่ง df เต็ม (รวม test) เข้า digit_analysis และ
    # walk_forward_proba ตรงๆ ทำให้ผลทั้งหมดปนข้อมูล test ตัดที่นี่จุด
    # เดียวให้ทุกอย่างข้างล่าง (digit_analysis, build_features,
    # walk_forward_proba) เห็นแค่ dev เท่านั้น เหมือน run_permutation.py
    n_total = len(df)
    cutoff = dev_cutoff(n_total)
    df = df.iloc[:cutoff]
    print(f"  [dev-only] ตัด test ออก: ใช้ {cutoff}/{n_total} แถวแรก "
          f"({df.index[0].date()} -> {df.index[-1].date()})")

    close = df["Close"].astype(float)
    digit_analysis(close, f"({name}, dev only)")

    tables = {}
    for use_open, tag in [(False, "T+1 (24h horizon)"),
                          (True, "Open-to-Close (6.5h horizon)")]:
        if use_open and "Open" not in df.columns:
            continue
        data = build_features(df, use_open=use_open)
        r = walk_forward_proba(data)
        flip = float(np.mean(r["y"] != r["y_prev"]))
        print(f"\n--- [1.1] {tag} ---")
        print(f"  n_oof={r['n']}  flip_rate={flip:.4f}")
        print(f"  model accuracy      = {r['acc']:.4f}")
        print(f"  persistence baseline= {r['acc_persistence']:.4f}")
        print(f"  edge                = {r['acc'] - r['acc_persistence']:+.4f}")
        tables[tag] = risk_coverage(r["y"], r["proba"], f"[{tag}]")

    if tables:
        os.makedirs(outdir, exist_ok=True)
        png = os.path.join(outdir, f"risk_coverage_{name.replace('.','_')}.png")
        plot_risk_coverage(tables, png)


def make_demo(price=155.0, n=2400, vol=0.015, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-08-26", periods=n)
    tk = tick_size(price)
    step = max(1.0, price * vol / tk)
    k = round(price / tk); ks = []
    for _ in range(n):
        k = max(1, k + int(round(rng.normal(0, step)))); ks.append(k)
    c = np.array(ks) * tk
    return pd.DataFrame({
        "Open": c + tk * rng.integers(-2, 3, n),
        "Close": c,
        "High": c + tk * rng.integers(0, 3, n),
        "Low": c - tk * rng.integers(0, 3, n),
        "Volume": rng.integers(1e6, 1e7, n)}, index=dates)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="KBANK.BK")
    ap.add_argument("--cache-dir", default="data_cache")
    ap.add_argument("--out", default="results/improve")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if a.demo:
        run(make_demo(), "DEMO_randomwalk", a.out)
    else:
        safe = a.ticker.replace(".", "_")
        path = os.path.join(a.cache_dir,
                            f"{safe}_2016-08-26_2026-08-28.csv")
        if not os.path.exists(path):
            raise SystemExit(f"ไม่พบไฟล์: {path}")
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        run(df, a.ticker, a.out)
