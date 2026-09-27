"""
report_live.py — รายงานผล live test จาก outcomes.csv (+ prediction log สำหรับความครบ/ตรงเวลา)
============================================================================================

    python report_live.py
    python report_live.py --outcomes results/dryrun/outcomes_dryrun.csv \\
                          --log results/dryrun/prediction_log_dryrun.csv \\
                          --out-dir results/dryrun/live_report --include-dry-run

เขียน: <out-dir>/live_summary.csv · live_report.md · live_cumulative.png (ถ้ามี matplotlib)

แยกตาม prediction_type × ticker × model เสมอ -- ห้ามรวมแบบ next_day กับ same_day_1600
ห้ามรวมกับผล validation / walk-forward เดิม · n < 20 วัน = ผลยังแกว่งมาก ห้ามสรุป
"""

import argparse
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import BASE_DIR, OUTPUT_DIR
from data_loader import load_stock
from trading_costs import cost_round_trip

BANGKOK = timezone(timedelta(hours=7))
MIN_N = 20
DEADLINE_HOUR = {"next_day": 9}          # ต้องออกคำทำนายก่อน 09:00 ของ target_date
GROUP = ["prediction_type", "ticker", "model"]


def summarize(o, cost=None):
    """metric ต่อกลุ่ม · o = DataFrame จาก record_outcomes (หนึ่งแถว/คำทำนาย)"""
    cost = cost_round_trip() if cost is None else cost
    rows = []
    for key, g in o.groupby(GROUP, sort=True):
        y, p, e = g["actual_return"].to_numpy(), g["predicted_return"].to_numpy(), g["error_return"].to_numpy()
        n = len(g)
        mae, mae_naive = np.mean(np.abs(e)), np.mean(np.abs(y))
        sse, sse_naive = np.sum(e ** 2), np.sum(y ** 2)
        wins, losses = int((g["beat_naive"] == "win").sum()), int((g["beat_naive"] == "lose").sum())
        moved = y != 0
        long_sig = p > cost
        rows.append({
            **dict(zip(GROUP, key)), "n": n,
            "first_date": g["target_date"].min(), "last_date": g["target_date"].max(),
            "MAE_return": mae, "RMSE_return": float(np.sqrt(np.mean(e ** 2))),
            "MAE_baht": float(g["abs_error_baht"].mean()),
            "MAE_naive": mae_naive,
            "relMAE": mae / mae_naive if mae_naive > 0 else np.nan,
            "R2_OOS": 1 - sse / sse_naive if sse_naive > 0 else np.nan,
            "wins": wins, "losses": losses, "ties": n - wins - losses,
            "win_rate_ex_ties": wins / (wins + losses) if wins + losses else np.nan,
            "Bias": float(np.mean(e)),
            "StdRatio": float(p.std() / y.std()) if n > 1 and np.ptp(y) > 0 else np.nan,
            "Rho": float(np.corrcoef(p, y)[0, 1]) if n > 2 and np.ptp(y) > 0 and np.ptp(p) > 0 else np.nan,
            "zero_move_days": int((~moved).sum()),
            "dir_hit_rate_moved_days": float(np.mean(np.sign(p[moved]) == np.sign(y[moved])))
                                       if moved.any() else np.nan,
            "long_signal_days": int(long_sig.sum()),
            "long_signal_up_rate": float(np.mean(y[long_sig] > 0)) if long_sig.any() else np.nan,
            "prev_close_mismatch": int((~g["prev_close_match"].astype(bool)).sum()),
            "enough_data": n >= MIN_N,
        })
    return pd.DataFrame(rows)


def coverage(log, closes):
    """ความครบ/ตรงเวลาจาก log: วันทำการที่ขาดคำทำนาย + คำทำนายที่ออกหลัง deadline"""
    rows = []
    for (ptype, ticker, model), g in log.groupby(GROUP):
        dates = pd.to_datetime(g["target_date"])
        c = closes.get(ticker)
        missing = []
        if c is not None:
            span = c.index[(c.index >= dates.min()) & (c.index <= dates.max())]
            missing = [str(d.date()) for d in span if d not in set(dates)]
        hour = DEADLINE_HOUR.get(ptype)
        late = 0
        if hour is not None:
            gen = pd.to_datetime(g["generated_at"])
            dl = dates.dt.tz_localize(BANGKOK) + pd.Timedelta(hours=hour)
            late = int((gen.to_numpy() >= dl.to_numpy()).sum())
        rows.append({"prediction_type": ptype, "ticker": ticker, "model": model,
                     "predictions": len(g), "missing_trading_days": len(missing),
                     "missing_dates": " ".join(missing), "late_predictions": late})
    return pd.DataFrame(rows)


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "NaN" if np.isnan(v) else f"{v:.6f}"
    return str(v)


def md_table(df, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return out


def render(summary, cov, label):
    L = [f"# Live test report — {label}", "",
         "> ผลของ **prospective prediction** เท่านั้น · ห้ามรวมกับผล validation / walk-forward เดิม",
         "> แยกตาม prediction_type × หุ้น × โมเดล · ผลจริงจาก raw_data/ (investing.com) แหล่งเดียวกับที่เทรน",
         f"> **n < {MIN_N} วัน = ผลยังแกว่งมาก อย่าสรุปว่าชนะหรือแพ้**", "",
         "อ่านอย่างไร: relMAE < 1 = ดีกว่า Naive (ทายว่าราคาไม่เปลี่ยน) · R2_OOS > 0 = MSE ดีกว่า Naive · "
         "win_rate = สัดส่วนวันที่พลาดน้อยกว่า Naive (ไม่นับวันเสมอ) · "
         "dir_hit_rate นับเฉพาะวันที่ราคาขยับจริง · long_signal = วันที่ทำนาย return เกินต้นทุนไป-กลับ "
         "(diagnostic ไม่ใช่กำไร)", ""]
    for ptype, s in (summary.groupby("prediction_type") if len(summary) else []):
        L += [f"## {ptype}", "", "### ความแม่นเทียบ Naive", ""]
        L += md_table(s, ["ticker", "model", "n", "first_date", "last_date", "MAE_return", "MAE_naive",
                          "relMAE", "R2_OOS", "RMSE_return", "MAE_baht", "wins", "losses", "ties",
                          "win_rate_ex_ties"])
        L += ["", "### รูปร่างคำทำนาย ทิศทาง และสัญญาณ", ""]
        L += md_table(s, ["ticker", "model", "Bias", "StdRatio", "Rho", "zero_move_days",
                          "dir_hit_rate_moved_days", "long_signal_days", "long_signal_up_rate",
                          "prev_close_mismatch", "enough_data"])
        c = cov[cov["prediction_type"] == ptype]
        if len(c):
            L += ["", "### ความครบและตรงเวลา", ""]
            L += md_table(c, ["ticker", "model", "predictions", "missing_trading_days",
                              "missing_dates", "late_predictions"])
        L.append("")
    if summary.empty:
        L += ["ยังไม่มีคำทำนายที่มีผลจริง", ""]
    return "\n".join(L)


def plot_cumulative(o, path):
    """เส้นสะสม |error model| − |error Naive| ตามเวลา (ต่ำกว่า 0 = ชนะ Naive สะสม)"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[report] ไม่มี matplotlib -- ข้ามกราฟ")
        return False
    keys = sorted(o[["prediction_type", "ticker"]].drop_duplicates().itertuples(index=False))
    fig, axes = plt.subplots(len(keys), 1, figsize=(9, 3.2 * len(keys)), squeeze=False)
    for ax, (ptype, ticker) in zip(axes[:, 0], keys):
        g = o[(o["prediction_type"] == ptype) & (o["ticker"] == ticker)]
        for model, m in g.groupby("model"):
            m = m.sort_values("target_date")
            ax.plot(pd.to_datetime(m["target_date"]),
                    (m["abs_error"] - m["naive_abs_error"]).cumsum(), marker="o", label=model)
        ax.axhline(0, color="gray", lw=1)
        ax.set_title(f"{ptype} · {ticker} · cumulative |err| − |err Naive| (below 0 = beats Naive)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description="รายงานผล live test")
    p.add_argument("--outcomes", default=str(OUTPUT_DIR / "outcomes.csv"))
    p.add_argument("--log", default=str(OUTPUT_DIR / "prediction_log.csv"))
    p.add_argument("--out-dir", default=str(OUTPUT_DIR / "live_report"))
    p.add_argument("--include-dry-run", action="store_true")
    args = p.parse_args(argv)
    out_dir = BASE_DIR / args.out_dir
    if args.include_dry_run and out_dir.resolve() == (OUTPUT_DIR / "live_report").resolve():
        raise ValueError("รายงานที่รวม dry-run ต้องเขียนลงโฟลเดอร์อื่น (--out-dir)")
    out_dir.mkdir(parents=True, exist_ok=True)

    opath = BASE_DIR / args.outcomes
    o = pd.read_csv(opath) if opath.exists() else pd.DataFrame(columns=GROUP)
    log = pd.read_csv(BASE_DIR / args.log)
    if not args.include_dry_run:
        dry = log["is_dry_run"].astype(str).str.lower().isin(["true", "1"])
        log = log[~dry]
        if len(o):
            o = o[~o["is_dry_run"].astype(str).str.lower().isin(["true", "1"])]
    closes = {t: load_stock(t, verbose=False)["Close"] for t in log["ticker"].unique()}

    summary = summarize(o) if len(o) else pd.DataFrame()
    cov = coverage(log, closes) if len(log) else pd.DataFrame(columns=["prediction_type"])
    summary.to_csv(out_dir / "live_summary.csv", index=False)
    label = "รวม dry-run (ทดสอบระบบ ไม่ใช่ผลจริง)" if args.include_dry_run else "official predictions"
    (out_dir / "live_report.md").write_text(render(summary, cov, label), encoding="utf-8")
    if len(o):
        plot_cumulative(o, out_dir / "live_cumulative.png")
    print(f"[report] {len(o)} แถวผลจริง · {len(summary)} กลุ่ม -> {out_dir}")


if __name__ == "__main__":
    main()
