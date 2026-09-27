"""
scan_universe.py
================
[ขั้น 3] สแกนหุ้นหลายตัวเพื่อพิสูจน์ว่า accuracy ของงาน parity
        ถูกกำหนดโดยโครงสร้าง (volatility / tick size) ไม่ใช่โดยอัลกอริทึม

แนวคิด
------
รัน pipeline เดียวกันกับหุ้นที่กระจายช่วงราคา 5-300 บาท แล้วพล็อต
    แกน X : การขยับเฉลี่ยต่อวัน (บาท)  [log scale]
    แกน Y : accuracy
    2 เส้น: persistence baseline  vs  โมเดลที่ดีที่สุด

ผลที่คาดหวัง: baseline ลาดลงจาก ~0.95 ไปหา ~0.50 อย่างต่อเนื่อง
             ส่วนเส้นโมเดลแนบไปกับ baseline ตลอด ไม่เคยแยกออก
             -> กราฟใบเดียวพิสูจน์ว่า ML ไม่มีอำนาจทำนายเหนือกฎง่ายๆ

การใช้งาน
--------
    python scan_universe.py --cache-dir data_cache --out results/scan
    python scan_universe.py --demo          # รันด้วยข้อมูลจำลอง ไม่ต้องต่อเน็ต

หมายเหตุ: สคริปต์นี้ standalone ไม่ต้อง import จาก pipeline เดิม
         แต่ใช้ logic เดียวกัน (tick size SET, ROUND_HALF_UP, walk-forward)
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from tick_utils import dev_cutoff, month_end_flag, tick_size, to_int_baht

warnings.filterwarnings("ignore")

SEED = 42


# =============================================================================
# 1) Universe - หุ้น SET50/100 กระจายช่วงราคา
# =============================================================================
# จัดกลุ่มตามช่วงราคาโดยประมาณ เพื่อให้แกน X กระจายทั่วถึง
# ปรับรายชื่อได้ตามต้องการ ยิ่งกระจายช่วงราคายิ่งดี

UNIVERSE = [
    # ราคาต่ำ (~1-10 บาท) -> คาดว่า baseline สูงมาก
    "TRUE.BK", "JAS.BK", "BANPU.BK", "SAWAD.BK", "TTB.BK",
    "KTC.BK", "AOT.BK", "IVL.BK", "PTT.BK", "OSP.BK",
    # ราคากลาง (~10-60 บาท)
    "BBL.BK", "SCC.BK", "CPALL.BK", "MINT.BK", "BDMS.BK",
    "CENTEL.BK", "HMPRO.BK", "TU.BK", "EGCO.BK", "GPSC.BK",
    "BH.BK", "TISCO.BK", "BEM.BK", "CRC.BK", "GULF.BK",
    # ราคาสูง (~60-300 บาท) -> คาดว่า baseline ~0.50
    "KBANK.BK", "ADVANC.BK", "SCB.BK", "PTTEP.BK", "DELTA.BK",
]


# =============================================================================
# 2) Tick size / to_int_baht — มาจาก tick_utils.py (แหล่งความจริงเดียว)
#
# *** เดิมไฟล์นี้นิยาม round_half_up() เองด้วย Decimal ROUND_HALF_UP ***
# คอมเมนต์เดิมอ้างว่า "ตามกฎอาจารย์" แต่จริงๆ ไม่ใช่ — ROUND_HALF_UP ปัด
# 142.50 -> 143 เสมอ แต่กฎอาจารย์คือ "เศษ > 0.5 ปัดขึ้น" (ต้อง > เท่านั้น)
# -> 142.50 ต้องได้ 142 ราคาหุ้นกลุ่มนี้ลงท้าย .50 บ่อยมาก (บาง ticker
# เกือบ 30-40% ของวัน) จุดปัดผิดนี้เปลี่ยน parity label ไปเยอะพอสมควร
# แก้โดยใช้ to_int_baht() จาก rounding.py ตรงๆ แทน
# =============================================================================


# =============================================================================
# 3) โหลดข้อมูล
# =============================================================================

def load_prices(ticker: str, cache_dir: str,
                start: str = "2016-08-26", end: str = "2026-08-28") -> pd.DataFrame | None:
    """
    อ่านจาก cache ก่อน ถ้าไม่มีค่อยดึงจาก yfinance
    *** ต้องใช้ราคา unadjusted เท่านั้น (auto_adjust=False) ***
    """
    safe = ticker.replace(".", "_")
    path = os.path.join(cache_dir, f"{safe}_{start}_{end}.csv")

    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    else:
        try:
            import yfinance as yf
        except ImportError:
            print(f"    [skip] {ticker}: ไม่มี cache และไม่มี yfinance")
            return None
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=False, progress=False)
        if df is None or len(df) == 0:
            print(f"    [skip] {ticker}: ดึงข้อมูลไม่ได้")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(path)

    need = {"Close", "High", "Low", "Volume"}
    if not need.issubset(df.columns) or len(df) < 500:
        print(f"    [skip] {ticker}: ข้อมูลไม่ครบ")
        return None
    return df.dropna(subset=["Close"])


# =============================================================================
# 4) สร้าง feature + target  (ทุก feature shift(1) แล้ว)
# =============================================================================

def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    close = df["Close"].astype(float)

    d["close"] = close
    d["tick"] = tick_size(close)
    d["n"] = (close / d["tick"]).round().astype(int)          # ราคาเป็นจำนวน tick
    d["dtick"] = d["n"].diff()

    # ---- target: parity ของราคาปิดปัดเป็นจำนวนเต็มบาท ----
    d["int_price"] = to_int_baht(close).astype(int)
    d["parity"] = d["int_price"] % 2                           # 1 = คี่

    # ---- features (ทั้งหมด shift(1) = ใช้ข้อมูลถึงวัน t-1) ----
    f = pd.DataFrame(index=df.index)
    f["n_mod2"] = d["n"] % 2
    f["n_mod4"] = d["n"] % 4
    f["dec_flag"] = ((close * 100).round().astype(int) % 100)  # สตางค์
    f["tick_regime"] = d["tick"]
    f["parity_lag"] = d["parity"]
    f["dtick_1"] = d["dtick"]
    f["dtick_2"] = d["dtick"].shift(1)
    f["dtick_3"] = d["dtick"].shift(2)
    f["dtick_par"] = d["dtick"].abs() % 2
    f["zero_rate20"] = (d["dtick"] == 0).rolling(20).mean()
    f["atr14_ticks"] = ((df["High"] - df["Low"]).rolling(14).mean()
                        / d["tick"])
    f["std20_ticks"] = d["dtick"].rolling(20).std()
    f["range_ticks"] = (df["High"] - df["Low"]) / d["tick"]
    f["vol_ratio"] = np.log(df["Volume"].replace(0, np.nan)
                            / df["Volume"].rolling(20).mean())
    f["dow"] = df.index.dayofweek
    # เดิมใช้ PeriodIndex.shift(-1) ซึ่งเลื่อน "ค่า" period ไม่ใช่ตำแหน่ง
    # ทำให้ทุกแถวไม่เท่ากันเสมอ -> ได้ 1 ทั้งคอลัมน์ (บั๊กเดียวกับ
    # improve_task_a.py เดิม) แก้ด้วย month_end_flag() จาก tick_utils.py
    f["is_month_end"] = month_end_flag(df.index)

    out = f.shift(1)                       # <-- กันรั่วสำคัญที่สุด
    out["y"] = d["parity"]
    out["y_prev"] = d["parity"].shift(1)
    out["_close"] = close
    out["_abs_move"] = close.diff().abs()
    out["_int_changed"] = (d["int_price"].diff() != 0).astype(int)
    return out.dropna()


# =============================================================================
# 5) Walk-forward evaluation
# =============================================================================

def walk_forward(data: pd.DataFrame, n_folds: int = 5) -> dict | None:
    """
    รัน walk-forward CV แล้วคืน out-of-fold accuracy ของแต่ละโมเดล
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    feat_cols = [c for c in data.columns
                 if not c.startswith("_") and c not in ("y", "y_prev")]
    X = data[feat_cols].to_numpy(dtype=float)
    y = data["y"].to_numpy(dtype=int)
    y_prev = data["y_prev"].to_numpy(dtype=int)
    n = len(y)

    if n < 800:
        return None

    fold_size = n // (n_folds + 1)
    oof = {m: [] for m in ("lr", "lgbm")}
    oof_true, oof_prev = [], []

    for k in range(1, n_folds + 1):
        tr_end = fold_size * k
        va_end = min(fold_size * (k + 1), n)
        Xtr, ytr = X[:tr_end], y[:tr_end]
        Xva, yva = X[tr_end:va_end], y[tr_end:va_end]
        if len(yva) < 30 or len(np.unique(ytr)) < 2:
            continue

        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=1000, C=1.0,
                                              random_state=SEED))
        lr.fit(Xtr, ytr)
        oof["lr"].append(lr.predict(Xva))

        try:
            from lightgbm import LGBMClassifier
            gb = LGBMClassifier(num_leaves=7, max_depth=3, learning_rate=0.02,
                                n_estimators=400, min_child_samples=100,
                                reg_lambda=10.0, subsample=0.8,
                                colsample_bytree=0.8, random_state=SEED,
                                verbose=-1)
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            gb = GradientBoostingClassifier(max_depth=3, learning_rate=0.02,
                                            n_estimators=400,
                                            random_state=SEED)
        gb.fit(Xtr, ytr)
        oof["lgbm"].append(gb.predict(Xva))

        oof_true.append(yva)
        oof_prev.append(y_prev[tr_end:va_end])

    if not oof_true:
        return None

    yt = np.concatenate(oof_true)
    yp = np.concatenate(oof_prev)
    maj = int(np.bincount(y[:fold_size]).argmax())

    acc = lambda p: float(np.mean(p == yt))
    return {
        "n_oof": len(yt),
        "acc_lr": acc(np.concatenate(oof["lr"])),
        "acc_lgbm": acc(np.concatenate(oof["lgbm"])),
        "acc_persistence": acc(yp),
        "acc_majority": acc(np.full(len(yt), maj)),
    }


# =============================================================================
# 6) สแกนทั้ง universe
# =============================================================================

def scan(tickers: list[str], cache_dir: str) -> pd.DataFrame:
    rows = []
    for i, tk in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {tk} ...", end=" ", flush=True)
        df = load_prices(tk, cache_dir)
        if df is None:
            continue
        # *** ตัด test set ออกก่อนสร้าง feature/เทรน ***
        # เดิมส่ง df เต็มเข้า build_dataset -> walk_forward ตรงๆ ทำให้
        # ผลปนข้อมูล test เหมือน improve_task_a.py ที่เจอบั๊กเดียวกัน
        df = df.iloc[:dev_cutoff(len(df))]
        data = build_dataset(df)
        res = walk_forward(data)
        if res is None:
            print("ข้อมูลน้อยเกินไป")
            continue

        best_model = max(res["acc_lr"], res["acc_lgbm"])
        best_base = max(res["acc_persistence"], res["acc_majority"])
        rows.append({
            "ticker": tk.replace(".BK", ""),
            "mean_price": round(float(data["_close"].mean()), 2),
            "abs_move": round(float(data["_abs_move"].mean()), 4),
            "int_change_rate": round(float(data["_int_changed"].mean()), 4),
            "acc_persistence": round(res["acc_persistence"], 4),
            "acc_majority": round(res["acc_majority"], 4),
            "acc_lr": round(res["acc_lr"], 4),
            "acc_lgbm": round(res["acc_lgbm"], 4),
            "best_model": round(best_model, 4),
            "best_baseline": round(best_base, 4),
            "edge": round(best_model - best_base, 4),
            "n_oof": res["n_oof"],
        })
        print(f"move={rows[-1]['abs_move']:.3f}฿  "
              f"base={best_base:.3f}  model={best_model:.3f}  "
              f"edge={rows[-1]['edge']:+.4f}")

    return pd.DataFrame(rows).sort_values("abs_move").reset_index(drop=True)


# =============================================================================
# 7) กราฟหลัก  <-- นี่คือรูปที่จะใส่รายงาน
# =============================================================================

def plot_scan(res: pd.DataFrame, out_png: str = "scan_result.png"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- ซ้าย: accuracy vs การขยับต่อวัน ---
    ax = axes[0]
    ax.semilogx(res["abs_move"], res["best_baseline"], "o-",
                color="#2b6cb0", lw=2, ms=6, label="Persistence / Majority baseline")
    ax.semilogx(res["abs_move"], res["best_model"], "s--",
                color="#d62728", lw=2, ms=6, label="Best ML model (LR / LightGBM)")
    ax.axhline(0.5, color="black", ls=":", lw=1, label="Random guess = 0.50")

    for tk in ("KBANK", "ADVANC"):
        r = res[res["ticker"] == tk]
        if len(r):
            ax.annotate(tk, (r["abs_move"].iloc[0], r["best_baseline"].iloc[0]),
                        textcoords="offset points", xytext=(6, 8),
                        fontsize=9, fontweight="bold")

    ax.set_xlabel("Mean absolute daily price move (THB, log scale)")
    ax.set_ylabel("Out-of-fold Accuracy")
    ax.set_title("Parity accuracy is set by price movement,\nnot by the algorithm")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3, which="both")

    # --- ขวา: edge ของโมเดลเหนือ baseline ---
    ax = axes[1]
    colors = ["#2ca02c" if e > 0 else "#d62728" for e in res["edge"]]
    ax.barh(res["ticker"], res["edge"], color=colors, alpha=0.75)
    ax.axvline(0, color="black", lw=1)
    # แถบนัยสำคัญโดยประมาณ (n_oof เฉลี่ย)
    n_avg = res["n_oof"].mean()
    band = 1.96 * np.sqrt(0.25 / n_avg)
    ax.axvspan(-band, band, color="grey", alpha=0.18,
               label=f"Not significant (±{band*100:.1f} pp)")
    ax.set_xlabel("Best model − Best baseline (accuracy)")
    ax.set_title("ML edge over trivial baseline")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n[plot] บันทึก: {out_png}")
    return fig


def summarize(res: pd.DataFrame):
    n_avg = res["n_oof"].mean()
    band = 1.96 * np.sqrt(0.25 / n_avg)
    sig = res[res["edge"].abs() > band]

    print("\n" + "=" * 78)
    print("  สรุปผลการสแกน")
    print("=" * 78)
    print(res.to_string(index=False))
    print(f"\n  จำนวนหุ้นที่สแกน            : {len(res)}")
    print(f"  ช่วงการขยับต่อวัน            : {res['abs_move'].min():.3f} - "
          f"{res['abs_move'].max():.3f} บาท")
    print(f"  ช่วง baseline accuracy      : {res['best_baseline'].min():.3f} - "
          f"{res['best_baseline'].max():.3f}")
    print(f"  ค่าเฉลี่ย edge ของโมเดล      : {res['edge'].mean():+.4f}")
    print(f"  เกณฑ์นัยสำคัญ (n≈{n_avg:.0f})   : ±{band:.4f}")
    print(f"  หุ้นที่โมเดลชนะอย่างมีนัยสำคัญ : {len(sig[sig['edge'] > 0])} / {len(res)}")

    corr = res[["abs_move", "best_baseline"]].corr().iloc[0, 1]
    print(f"\n  correlation(การขยับต่อวัน, baseline accuracy) = {corr:.3f}")
    print("  -> ยิ่งราคาขยับมาก accuracy ยิ่งเข้าใกล้ 0.50")


# =============================================================================
# 8) โหมด demo - ข้อมูลจำลอง ใช้ทดสอบว่าโค้ดทำงานถูก
# =============================================================================

def make_demo_data(price_level: float, n_days: int = 2400,
                   vol_pct: float = 0.015, seed: int = 0) -> pd.DataFrame:
    """สร้าง random walk บน tick grid -> ไม่มี signal โดยนิยาม"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-08-26", periods=n_days)
    tick = tick_size(price_level)
    step_ticks = max(1, price_level * vol_pct / tick)

    n = round(price_level / tick)
    ns = []
    for _ in range(n_days):
        n = max(1, n + int(round(rng.normal(0, step_ticks))))
        ns.append(n)
    close = np.array(ns) * tick

    return pd.DataFrame({
        "Close": close,
        "High": close + tick * rng.integers(0, 3, n_days),
        "Low": close - tick * rng.integers(0, 3, n_days),
        "Volume": rng.integers(1e6, 1e7, n_days),
    }, index=dates)


def run_demo():
    print("=" * 78)
    print("  โหมด DEMO - random walk บน tick grid (ไม่มี signal โดยนิยาม)")
    print("  ใช้ยืนยันว่ากราฟที่ได้สะท้อนโครงสร้าง ไม่ใช่ signal จริง")
    print("=" * 78)

    levels = [3, 6, 12, 22, 35, 60, 95, 140, 180, 230, 290]
    rows = []
    for i, lv in enumerate(levels):
        df = make_demo_data(lv, seed=i)
        data = build_dataset(df)
        res = walk_forward(data)
        if res is None:
            continue
        bm = max(res["acc_lr"], res["acc_lgbm"])
        bb = max(res["acc_persistence"], res["acc_majority"])
        rows.append({
            "ticker": f"SIM{lv}",
            "mean_price": round(float(data["_close"].mean()), 2),
            "abs_move": round(float(data["_abs_move"].mean()), 4),
            "int_change_rate": round(float(data["_int_changed"].mean()), 4),
            "acc_persistence": round(res["acc_persistence"], 4),
            "acc_majority": round(res["acc_majority"], 4),
            "acc_lr": round(res["acc_lr"], 4),
            "acc_lgbm": round(res["acc_lgbm"], 4),
            "best_model": round(bm, 4),
            "best_baseline": round(bb, 4),
            "edge": round(bm - bb, 4),
            "n_oof": res["n_oof"],
        })
        print(f"  price≈{lv:>4}฿  move={rows[-1]['abs_move']:.3f}฿  "
              f"base={bb:.3f}  model={bm:.3f}  edge={bm-bb:+.4f}")

    return pd.DataFrame(rows).sort_values("abs_move").reset_index(drop=True)


# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data_cache")
    ap.add_argument("--out", default="results/scan")
    ap.add_argument("--demo", action="store_true",
                    help="รันด้วยข้อมูลจำลอง ไม่ต้องต่อเน็ต")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    res = run_demo() if args.demo else scan(UNIVERSE, args.cache_dir)

    if len(res) == 0:
        raise SystemExit("ไม่มีข้อมูลให้สแกน")

    summarize(res)
    res.to_csv(os.path.join(args.out, "scan_results.csv"), index=False)
    plot_scan(res, os.path.join(args.out, "scan_result.png"))
    print(f"[csv] บันทึก: {os.path.join(args.out, 'scan_results.csv')}")
