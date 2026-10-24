"""
split_profile.py — เทียบ distribution ของ train / val / test (feedback ข้อ 5)
=============================================================================

*** ไฟล์นี้ทำอะไร ***
อ่านค่า return รายวัน (`targets.build_targets`) ของทั้งสามชุด แล้วรายงานว่าสามชุด
"หน้าตาเหมือนกันแค่ไหน": จำนวนแถว ช่วงวันที่ โมเมนต์ของ return (mean / SD / skew /
kurtosis) สัดส่วนวันราคานิ่ง สัดส่วนขึ้น/ลง MAE ของ Naive ราคาเฉลี่ย tick size ที่พบ
บ่อยสุด และสัดส่วนวันที่ราคาขยับน้อยกว่า 1 tick
แล้วทดสอบความต่างของ distribution ด้วย Kolmogorov-Smirnov 2 กลุ่ม และ Levene

*** ทำไมต้องมี / ตอบ feedback ข้อไหน ***
feedback ข้อ 5 (14 ก.ย. 2026) ถามว่า "test set ควร balance ไหม / ทำไมให้เท่า ๆ กัน"
ก่อนหน้านี้ repo ตอบได้แค่ด้วยคอมเมนต์ใน `config.py` ว่าตั้งวันที่ให้ขนาดเท่าเดิม
แต่ไม่มีตัวเลขยืนยันว่าสามชุด "เทียบกันได้" จริงหรือไม่ ไฟล์นี้สร้างตัวเลขนั้น

*** ขอบเขต -- สำคัญมาก ***
ไฟล์นี้อ่านค่า return ของช่วง historical test เพื่ออธิบายข้อมูลเท่านั้น
ไม่มีโมเดลใดถูกประเมินบน test (ราคาช่วงนี้เป็นข้อมูลสาธารณะและถูกใช้เทรนโมเดล live
อยู่แล้ว) -- ไม่มีการ fit / predict / สร้าง feature ในไฟล์นี้เลย ไม่มีการเลือกอะไร
จากตัวเลขเหล่านี้ ทุกค่าเป็น descriptive statistics

    python split_profile.py

ผลลัพธ์: results/split_profile.csv · results/split_profile_tests.csv · results/split_profile.md
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    TICKERS, OUTPUT_DIR, SPLIT_BY_DATE, HISTORICAL_TEST_END, SET_COSTS,
)
from data_loader import load_stock
from targets import build_targets
from trading_costs import tick_size, cost_round_trip

PROFILE_CSV = OUTPUT_DIR / "split_profile.csv"
TESTS_CSV = OUTPUT_DIR / "split_profile_tests.csv"
PROFILE_MD = OUTPUT_DIR / "split_profile.md"

SPLIT_NAMES = ["train", "val", "test"]


def load_returns(ticker):
    """
    คืน DataFrame คอลัมน์ y_return / close / prev_close ของทุกแถวถึง HISTORICAL_TEST_END

    ตัดที่ HISTORICAL_TEST_END เหมือน `main.truncate_for_mode(use_test=True)` เพื่อให้
    "test" ที่รายงานที่นี่เป็นช่วง 362 แถวเดิมเป๊ะ ไม่ใช่รวมแถวที่ append เข้ามาทีหลัง
    (ถ้าไม่ตัด ตัวเลขของไฟล์นี้จะเลื่อนทุกวันที่มีข้อมูลใหม่)
    """
    df = load_stock(ticker, verbose=False).loc[:HISTORICAL_TEST_END]
    t = build_targets(df, verbose=False)
    # แถวแรกของ dataset ไม่มี prev_close -> y_return เป็น NaN (นิยามยังไม่ได้ ไม่ใช่ missing)
    return t.dropna(subset=["y_return"])


def split_slices(t):
    """
    แบ่งตามวันที่ชุดเดียวกับ `splits.chronological_split` (config.SPLIT_BY_DATE)

    ไม่เรียก chronological_split() ตรง ๆ เพราะฟังก์ชันนั้นรับ (X, y) ที่ผ่าน
    prepare_xy มาแล้ว (ตัดแถว warm-up ของ feature ทิ้งต้นชุด) ส่วนไฟล์นี้ต้องอธิบาย
    "ข้อมูลราคา" ไม่ใช่ "ข้อมูลที่โมเดลเห็น" จึงใช้ทุกแถวที่คำนวณ return ได้
    ขอบวันที่เป็นค่าเดียวกันทั้งสองทาง -> ชุด val/test ได้แถวเท่ากันเป๊ะ
    """
    train_end = pd.Timestamp(SPLIT_BY_DATE["train_end"])
    val_end = pd.Timestamp(SPLIT_BY_DATE["val_end"])
    return {
        "train": t.loc[:train_end],
        "val": t.loc[t.index > train_end].loc[:val_end],
        "test": t.loc[t.index > val_end],
    }


def profile_split(ticker, split, part):
    """ตัวเลข descriptive ของหนึ่งชุด -- ไม่มีโมเดล ไม่มีการเลือกอะไร"""
    r = part["y_return"].to_numpy(dtype=float)
    prev = part["prev_close"].to_numpy(dtype=float)
    ticks = np.array([tick_size(p) for p in prev])
    # 1 tick คิดเป็น return เท่าไหร่ของวันนั้น (tick เปลี่ยนตามระดับราคา -> คิดรายวัน)
    tick_return = ticks / prev

    counts = pd.Series(ticks).value_counts()
    return {
        "ticker": ticker,
        "split": split,
        "n": len(part),
        "date_start": part.index[0].date().isoformat(),
        "date_end": part.index[-1].date().isoformat(),
        "ret_mean": float(r.mean()),
        "ret_sd": float(r.std(ddof=1)),
        # skew/kurtosis แบบ sample (bias=False) · kurtosis เป็น excess (ปกติ = 0)
        "ret_skew": float(stats.skew(r, bias=False)),
        "ret_kurtosis_excess": float(stats.kurtosis(r, bias=False)),
        "ret_min": float(r.min()),
        "ret_max": float(r.max()),
        # ราคานิ่งเป๊ะ: close == prev_close -> return = 0 เป๊ะ
        "frac_zero_move": float(np.mean(r == 0.0)),
        "frac_up": float(np.mean(r > 0.0)),
        "frac_down": float(np.mean(r < 0.0)),
        # Naive (random walk) = ทำนาย return 0 ทุกวัน -> MAE = mean(|r|)
        "naive_mae_return": float(np.mean(np.abs(r))),
        "price_mean": float(part["close"].mean()),
        "price_min": float(part["close"].min()),
        "price_max": float(part["close"].max()),
        "tick_mode": float(counts.index[0]),
        "tick_mode_frac": float(counts.iloc[0] / len(ticks)),
        "frac_move_lt_1_tick": float(np.mean(np.abs(r) < tick_return)),
        "frac_move_gt_cost_rt": float(np.mean(np.abs(r) > cost_round_trip())),
    }


def distribution_tests(ticker, slices):
    """
    KS 2 กลุ่ม + Levene — descriptive เท่านั้น

    *** ป้ายกำกับ: descriptive ***
    - ไม่ได้ประกาศล่วงหน้าในแผน pre-register ใด และไม่ได้ใช้ตัดสินใจอะไรเลย
    - return รายวันมี volatility clustering (ไม่ i.i.d.) ซึ่งละเมิดสมมติฐานของ
      ทั้ง KS และ Levene -> p-value ที่ได้มองว่า "เล็ก/ไม่เล็ก" แบบหยาบได้เท่านั้น
      ห้ามรายงานเป็นหลักฐานทางสถิติที่เข้มงวด
    """
    rows = []
    train = slices["train"]["y_return"].to_numpy(dtype=float)
    for other in ("val", "test"):
        x = slices[other]["y_return"].to_numpy(dtype=float)
        ks = stats.ks_2samp(train, x)
        lev = stats.levene(train, x, center="median")   # Brown-Forsythe: ทนหางหนา
        rows.append({
            "ticker": ticker,
            "comparison": f"train vs {other}",
            "n_train": len(train),
            "n_other": len(x),
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "levene_stat": float(lev.statistic),
            "levene_pvalue": float(lev.pvalue),
            "sd_train": float(train.std(ddof=1)),
            "sd_other": float(x.std(ddof=1)),
            "sd_ratio_other_over_train": float(x.std(ddof=1) / train.std(ddof=1)),
            "label": "descriptive",
        })
    return rows


def build_tables():
    prof_rows, test_rows = [], []
    for ticker in TICKERS:
        t = load_returns(ticker)
        slices = split_slices(t)
        # assert เชิงโครงสร้าง: ไม่มีวันซ้อนทับ และครบทุกแถว
        assert sum(len(s) for s in slices.values()) == len(t)
        assert slices["train"].index[-1] < slices["val"].index[0]
        assert slices["val"].index[-1] < slices["test"].index[0]
        assert slices["test"].index[-1] <= pd.Timestamp(HISTORICAL_TEST_END)
        for name in SPLIT_NAMES:
            prof_rows.append(profile_split(ticker, name, slices[name]))
        test_rows += distribution_tests(ticker, slices)
        print(f"[profile] {ticker}: " + " · ".join(
            f"{k} {len(v)} แถว" for k, v in slices.items()))
    return pd.DataFrame(prof_rows), pd.DataFrame(test_rows)


# ---------------------------------------------------------------
# รายงาน
# ---------------------------------------------------------------
def _md_table(df, cols=None, fmt=None):
    """fmt = {ชื่อคอลัมน์: callable ที่รับค่าแล้วคืน str} (เช่น "{:.4f}".format)"""
    cols = cols or df.columns.tolist()
    fmt = fmt or {}
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        d = dict(zip(df.columns, r))
        L.append("| " + " | ".join(
            (fmt[c](d[c]) if c in fmt else str(d[c])) for c in cols) + " |")
    return L


def write_markdown(prof, tests):
    pct = "{:.1%}".format
    f4 = "{:.4f}".format
    f6 = "{:.6f}".format
    f2 = "{:.2f}".format
    sci = "{:.3g}".format
    cmt = SPLIT_BY_DATE["train_end"]
    vend = SPLIT_BY_DATE["val_end"]

    L = [
        "# split_profile — distribution ของ train / val / test (feedback ข้อ 5)",
        "",
        "> สร้างโดย `split_profile.py` · ทุกตัวเลขในไฟล์นี้มาจาก `results/split_profile.csv`",
        "> และ `results/split_profile_tests.csv` ที่รันในรอบเดียวกัน",
        "",
        "## ขอบเขตของไฟล์นี้ (อ่านก่อน)",
        "",
        "ไฟล์นี้อ่านค่า return ของช่วง historical test เพื่ออธิบายข้อมูลเท่านั้น "
        "**ไม่มีโมเดลใดถูกประเมินบน test** (ราคาช่วงนี้เป็นข้อมูลสาธารณะและถูกใช้เทรน"
        "โมเดล live อยู่แล้ว)",
        "",
        "- `split_profile.py` ไม่เรียก `build_features` ไม่ fit ไม่ predict อะไรเลย",
        "- ไม่มีการเลือกโมเดล hyperparameter หรือ feature จากตัวเลขในไฟล์นี้",
        "- `PRE_TEST_LOCK.md` ยังไม่มีในโปรเจกต์ -> historical test **ยังไม่ถูกเปิด** "
        "เป็นผลการประเมินของโมเดล และการเปิดเป็นการตัดสินใจของผู้ใช้",
        "",
        "## 1. คำถามของอาจารย์: test set ควร balance / ทำไมให้เท่า ๆ กัน",
        "",
        "### (1) ทำไม val กับ test มีจำนวนแถวเท่ากัน",
        "",
        "เพราะ **ตั้งใจเลือกวันที่ให้เท่ากัน เพื่อให้เทียบผลกับเวอร์ชันก่อนหน้าได้** "
        "ไม่ใช่เพราะต้อง balance อะไร · คอมเมนต์ของ `SPLIT_BY_DATE` ใน `config.py` "
        "เขียนไว้ตรง ๆ ว่า:",
        "",
        "> *วันที่ชุดนี้ทำให้ train/val/test มีขนาดเท่าเวอร์ชันก่อนหน้า "
        "(1,688 / 362 / 362) จึงเทียบผลได้ว่าอะไรเปลี่ยนเพราะการแก้บั๊ก "
        "ไม่ใช่เพราะ split เลื่อน*",
        "",
        f"การแบ่งจึง freeze ด้วย **วันที่** (`train_end` {cmt}, `val_end` {vend}, "
        f"ขอบปลาย test `HISTORICAL_TEST_END` {HISTORICAL_TEST_END}) "
        "ไม่ใช่ด้วยสัดส่วน (`TRAIN_RATIO` / `VAL_RATIO` ถูกตั้งเป็น `None` เลิกใช้แล้ว) "
        "และห้าม shuffle",
        "",
        "### (2) งาน B เป็น regression ไม่มี class ให้ balance",
        "",
        "คำว่า balance ใช้กับงานจำแนกประเภทที่มี class ไม่เท่ากัน (เช่น Task A / C "
        "ที่ทำนายขึ้น/ลง แล้วต้องดู base rate) แต่งาน B ทำนาย **ค่าต่อเนื่อง** "
        "(return รายวัน) จึงไม่มี class ให้นับ",
        "",
        "สิ่งที่เทียบกันได้จริงสำหรับ regression และรายงานไว้ข้างล่างคือ:",
        "",
        "- **volatility** (`ret_sd`) — ถ้า test ผันผวนกว่า train มาก MAE ทุกตัวจะสูงขึ้นทั้งแถว",
        "- **สัดส่วนวันราคานิ่ง** (`frac_zero_move`) — ยิ่งเยอะ Naive (ทาย 0) ยิ่งได้เปรียบ",
        "- **ทิศทาง** (`frac_up` / `frac_down`) — ใกล้ 50/50 แค่ไหน",
        "- **ระดับราคาและ tick** — ราคาต่างระดับมี tick ต่างกัน -> ขยับ 1 tick "
        "คิดเป็น return ไม่เท่ากัน",
        "",
        "### (3) สามชุดต่างกันแค่ไหน และผลต่อการตีความ MAE vs Naive",
        "",
    ]

    # --- สรุปเชิงตัวเลขอัตโนมัติ (ห้ามพิมพ์เลขจากความจำ) ---
    for ticker in prof["ticker"].unique():
        sub = prof[prof["ticker"] == ticker].set_index("split")
        tr, va, te = sub.loc["train"], sub.loc["val"], sub.loc["test"]
        ks = tests[tests["ticker"] == ticker].set_index("comparison")
        kv, kt = ks.loc["train vs val"], ks.loc["train vs test"]
        L += [
            f"**{ticker}**",
            "",
            f"- ขนาด: train {tr['n']:,} / val {va['n']:,} / test {te['n']:,} แถว",
            f"- SD ของ return: train {tr['ret_sd']:.6f} · val {va['ret_sd']:.6f} "
            f"({va['ret_sd'] / tr['ret_sd']:.2f}x ของ train) · test {te['ret_sd']:.6f} "
            f"({te['ret_sd'] / tr['ret_sd']:.2f}x ของ train)",
            f"- วันราคานิ่งเป๊ะ (return = 0): train {tr['frac_zero_move']:.1%} · "
            f"val {va['frac_zero_move']:.1%} · test {te['frac_zero_move']:.1%}",
            f"- Naive MAE_return: train {tr['naive_mae_return']:.6f} · "
            f"val {va['naive_mae_return']:.6f} · test {te['naive_mae_return']:.6f}",
            f"- ราคาเฉลี่ย: train {tr['price_mean']:.2f} · val {va['price_mean']:.2f} · "
            f"test {te['price_mean']:.2f} บาท · tick ที่พบบ่อยสุด "
            f"{tr['tick_mode']:g} / {va['tick_mode']:g} / {te['tick_mode']:g}",
            f"- วันที่ราคาขยับน้อยกว่า 1 tick: train {tr['frac_move_lt_1_tick']:.1%} · "
            f"val {va['frac_move_lt_1_tick']:.1%} · test {te['frac_move_lt_1_tick']:.1%}",
            f"- KS train vs val: D = {kv['ks_stat']:.4f} (p = {kv['ks_pvalue']:.3g}) · "
            f"train vs test: D = {kt['ks_stat']:.4f} (p = {kt['ks_pvalue']:.3g})",
            f"- Levene train vs val: p = {kv['levene_pvalue']:.3g} · "
            f"train vs test: p = {kt['levene_pvalue']:.3g}",
            "",
        ]

    L += [
        "**ผลต่อการตีความ MAE vs Naive**",
        "",
        "1. `naive_mae_return` ของแต่ละชุด **ไม่เท่ากัน** -> ห้ามเทียบ MAE ข้ามชุด "
        "(เช่น เอา MAE บน test ไปเทียบกับ MAE บน val แล้วสรุปว่าโมเดลดีขึ้น/แย่ลง) "
        "ต้องเทียบกับ Naive **ของชุดเดียวกัน** เสมอ ซึ่งเป็นสิ่งที่ "
        "`evaluate.compare_to_baseline` บังคับอยู่แล้ว (baseline อยู่ในตารางเดียวกับโมเดล)",
        "2. ชุดที่ `frac_zero_move` สูงกว่า จะเอื้อ Naive มากกว่าโดยอัตโนมัติ เพราะ Naive "
        "ทายค่า 0 แล้วถูกเป๊ะในวันเหล่านั้น (error = 0) ส่วนโมเดล ML ที่ทายค่าไม่เป็นศูนย์ "
        "จะเสีย error ทุกวันนิ่ง -- ดู `results/feedback/WHY_NAIVE_WINS.md`",
        "3. ชุดที่ SD สูงกว่าจะมี MAE สูงกว่าทั้งแถว (ทั้งโมเดลและ Naive) -> ตัวเลขที่ "
        "ความหมายคงที่ข้ามชุดคือ **relMAE = MAE_model / MAE_naive** ไม่ใช่ MAE ดิบ",
        "",
        "## 2. ตารางเต็ม (ต่อหุ้น ต่อชุด)",
        "",
    ]
    cols = ["ticker", "split", "n", "date_start", "date_end", "ret_mean", "ret_sd",
            "ret_skew", "ret_kurtosis_excess", "frac_zero_move", "frac_up",
            "frac_down", "naive_mae_return", "price_mean", "tick_mode",
            "frac_move_lt_1_tick", "frac_move_gt_cost_rt"]
    L += _md_table(prof, cols, fmt={
        "ret_mean": f6, "ret_sd": f6, "ret_skew": f4, "ret_kurtosis_excess": f4,
        "frac_zero_move": pct, "frac_up": pct, "frac_down": pct,
        "naive_mae_return": f6, "price_mean": f2, "tick_mode": "{:g}".format,
        "frac_move_lt_1_tick": pct, "frac_move_gt_cost_rt": pct,
        "n": "{:,}".format})
    L += [
        "",
        f"- `frac_move_gt_cost_rt` ใช้ต้นทุนไป-กลับ {cost_round_trip() * 1e4:.1f} bps "
        f"(commission {SET_COSTS['commission_per_side']:.4%} ต่อขา -- ดู `REFERENCES.md`)",
        "- `tick_mode` = tick size ที่พบบ่อยสุดในชุดนั้น (จาก `trading_costs.tick_size`) · "
        "`frac_move_lt_1_tick` คิด tick รายวันจาก `prev_close` ของวันนั้นก่อนเฉลี่ย",
        "- `ret_kurtosis_excess` เป็น excess kurtosis (การแจกแจงปกติ = 0)",
        "- คอลัมน์ที่ไม่ได้แสดงในตารางนี้แต่มีใน CSV: `ret_min`, `ret_max`, `price_min`, "
        "`price_max`, `tick_mode_frac`",
        "",
        "## 3. การทดสอบความต่างของ distribution (descriptive)",
        "",
        "> **ป้ายกำกับ: descriptive เท่านั้น** ไม่ได้ประกาศล่วงหน้าในแผนใด "
        "และไม่ได้ใช้ตัดสินใจอะไร",
        "> return รายวันมี volatility clustering (ไม่ i.i.d.) ซึ่งละเมิดสมมติฐานของทั้ง KS "
        "และ Levene -> อ่าน p-value ได้แบบหยาบว่าเล็ก/ไม่เล็กเท่านั้น "
        "ห้ามรายงานเป็นหลักฐานทางสถิติที่เข้มงวด",
        "",
        "- **Kolmogorov-Smirnov 2 กลุ่ม** (`scipy.stats.ks_2samp`): ทดสอบว่า distribution "
        "ทั้งก้อนต่างกันไหม",
        "- **Levene** (`scipy.stats.levene` + `center=median` = Brown-Forsythe ซึ่งทนหางหนา): "
        "ทดสอบว่าความแปรปรวนต่างกันไหม",
        "",
    ]
    tcols = ["ticker", "comparison", "n_train", "n_other", "ks_stat", "ks_pvalue",
             "levene_stat", "levene_pvalue", "sd_train", "sd_other",
             "sd_ratio_other_over_train", "label"]
    L += _md_table(tests, tcols, fmt={
        "ks_stat": f4, "ks_pvalue": sci, "levene_stat": f4, "levene_pvalue": sci,
        "sd_train": f6, "sd_other": f6, "sd_ratio_other_over_train": f4})
    L += [
        "",
        "## 4. ข้อจำกัด",
        "",
        "- ตัวเลขทั้งหมดเป็น descriptive ของข้อมูลราคา ไม่ใช่ผลของโมเดล",
        "- ชุด train ที่รายงานที่นี่มีทุกแถวที่คำนวณ return ได้ ส่วน train ที่โมเดลเห็นจริง "
        "สั้นกว่าเล็กน้อย เพราะ `splits.prepare_xy` ตัดแถว warm-up ของ rolling indicator "
        "ทิ้ง (เหลือ 1,688 แถว) -- ขอบวันที่เป็นชุดเดียวกัน ทำให้ val/test ได้แถวเท่ากันเป๊ะ",
        "- ชุด test ถูกตัดที่ `HISTORICAL_TEST_END` จึงเป็นช่วง 362 แถวเดิมเสมอ "
        "ไม่เลื่อนตามข้อมูลที่ append เข้ามาใหม่",
        "",
    ]
    PROFILE_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[profile] บันทึก {PROFILE_MD.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("  split_profile.py -- descriptive เท่านั้น ไม่มีการเทรน/ทำนายบน test")
    print("=" * 78)
    prof, tests = build_tables()
    prof.to_csv(PROFILE_CSV, index=False, encoding="utf-8-sig")
    tests.to_csv(TESTS_CSV, index=False, encoding="utf-8-sig")
    print(f"[profile] บันทึก {PROFILE_CSV.name}, {TESTS_CSV.name}")
    write_markdown(prof, tests)
    print(prof.to_string(index=False))
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
