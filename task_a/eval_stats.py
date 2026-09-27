"""
eval_stats.py
=============
โมดูลทดสอบทางสถิติสำหรับงานทำนาย parity (คู่/คี่)

ใช้กับ pipeline เดิม โดย import เข้าไป:
    from eval_stats import verdict, full_report, permutation_test, plot_null_distribution

ครอบคลุม:
  [ขั้น 1] verdict()            - แก้บั๊กบรรทัดสรุปผล "ชนะ baseline"
  [ขั้น 2] binom_ci()           - 95% CI ของ accuracy (Wilson)
           binom_test_vs()      - p-value เทียบกับค่าอ้างอิง (เช่น 0.50)
           mcnemar_test()       - เทียบโมเดล vs baseline แบบ paired
           prob_metrics()       - Brier score + log loss + calibration
           permutation_test()   - null distribution จากการสลับ label
           full_report()        - รวมทุกอย่างเป็นตารางเดียว
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# [ขั้น 1] แก้บั๊กบรรทัดสรุปผล
# =============================================================================

def verdict(model_acc: float,
            baseline_acc: float,
            n: int,
            alpha: float = 0.05) -> str:
    """
    สรุปผลว่าโมเดลชนะ baseline จริงหรือไม่ โดยเทียบกับเกณฑ์นัยสำคัญ

    บั๊กเดิม: สคริปต์พิมพ์ "โมเดล ML ชนะ baseline (ต่างกัน 0.0055)"
              ทั้งที่บรรทัดถัดไปบอกว่าต้องนำ >= 0.0513 ถึงจะนับ
              -> ขัดแย้งกันเอง

    Parameters
    ----------
    model_acc    : accuracy ของโมเดล
    baseline_acc : accuracy ของ baseline ที่ดีที่สุด
    n            : จำนวนตัวอย่างใน val/test set
    alpha        : ระดับนัยสำคัญ (default 0.05 = 95%)
    """
    diff = model_acc - baseline_acc
    z = stats.norm.ppf(1 - alpha / 2)
    # margin of error แบบ conservative (p=0.5 ให้ variance สูงสุด)
    threshold = z * np.sqrt(0.25 / n)

    if diff >= threshold:
        tag = "ชนะ baseline อย่างมีนัยสำคัญ"
    elif diff > 0:
        tag = (f"สูงกว่า baseline {diff:+.4f} แต่ต่ำกว่าเกณฑ์ {threshold:.4f} "
               f"-> ถือว่าไม่ต่างกันทางสถิติ")
    elif diff == 0:
        tag = "เท่ากับ baseline"
    else:
        tag = f"ไม่ชนะ baseline (ต่ำกว่า {abs(diff):.4f})"

    return (f"  ผลสรุป (Accuracy) : {tag}\n"
            f"  เกณฑ์นัยสำคัญ (n={n}, {int((1-alpha)*100)}%) = "
            f"{threshold*100:.2f} percentage point")


# =============================================================================
# [ขั้น 2.1] Confidence interval + binomial test
# =============================================================================

def binom_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    95% CI ของสัดส่วน ด้วยวิธี Wilson score
    (ดีกว่า normal approximation เมื่อ p ใกล้ 0 หรือ 1)

    k = จำนวนที่ทายถูก, n = จำนวนทั้งหมด
    """
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def binom_test_vs(k: int, n: int, p0: float = 0.5) -> float:
    """
    p-value สองทาง ทดสอบว่า accuracy ต่างจาก p0 หรือไม่
    ใช้ exact binomial test (ไม่ใช่ normal approximation)
    """
    if n == 0:
        return np.nan
    return float(stats.binomtest(k, n, p0, alternative="two-sided").pvalue)


# =============================================================================
# [ขั้น 2.2] McNemar's test - เทียบโมเดล vs baseline แบบ paired
# =============================================================================

def mcnemar_test(y_true, pred_a, pred_b, exact: bool = True) -> dict:
    """
    McNemar's test สำหรับเทียบ classifier สองตัวบนข้อมูลชุดเดียวกัน

    เหมาะกว่าการเทียบ accuracy ตรงๆ เพราะเป็น paired test
    -> ใช้ข้อมูลว่า "วันไหนที่ทายไม่ตรงกัน" ซึ่งมี power สูงกว่า

    ตาราง 2x2:
                    B ถูก    B ผิด
        A ถูก        n00      n01   <- b
        A ผิด        n10      n11   <- c
                      ^c

    H0: โมเดลทั้งสองมี error rate เท่ากัน

    Returns
    -------
    dict: b, c, statistic, pvalue, interpretation
    """
    y_true = np.asarray(y_true)
    a_correct = (np.asarray(pred_a) == y_true)
    b_correct = (np.asarray(pred_b) == y_true)

    b = int(np.sum(a_correct & ~b_correct))   # A ถูก B ผิด
    c = int(np.sum(~a_correct & b_correct))   # A ผิด B ถูก
    n_disc = b + c

    if n_disc == 0:
        return {"b": b, "c": c, "statistic": np.nan, "pvalue": 1.0,
                "interpretation": "ทายเหมือนกันทุกแถว เทียบไม่ได้"}

    if exact or n_disc < 25:
        # exact binomial บนจำนวน discordant pairs
        pval = float(stats.binomtest(b, n_disc, 0.5,
                                     alternative="two-sided").pvalue)
        statistic = np.nan
        method = "exact"
    else:
        # chi-square with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / n_disc
        pval = float(stats.chi2.sf(statistic, df=1))
        method = "chi2"

    if pval < 0.05:
        interp = "แตกต่างกันอย่างมีนัยสำคัญ (p < 0.05)"
    else:
        interp = "ไม่ต่างกันทางสถิติ (p >= 0.05)"

    return {"b": b, "c": c, "n_discordant": n_disc, "method": method,
            "statistic": statistic, "pvalue": pval, "interpretation": interp}


# =============================================================================
# [ขั้น 2.3] Probabilistic metrics
# =============================================================================

def prob_metrics(y_true, y_prob, n_bins: int = 10) -> dict:
    """
    Brier score, log loss, และ calibration error

    Brier   : mean squared error ของ probability (ต่ำ = ดี)
              ค่าอ้างอิง: ทำนาย 0.5 ทุกครั้ง -> Brier = 0.25 พอดี
    LogLoss : ค่าอ้างอิง: ทำนาย 0.5 ทุกครั้ง -> log(2) = 0.6931
    ECE     : Expected Calibration Error (ต่ำ = probability น่าเชื่อถือ)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1 - 1e-15)

    brier = float(np.mean((y_prob - y_true) ** 2))
    logloss = float(-np.mean(y_true * np.log(y_prob) +
                             (1 - y_true) * np.log(1 - y_prob)))

    # Expected Calibration Error
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = (idx == b)
        if mask.sum() > 0:
            ece += (mask.sum() / len(y_prob)) * abs(
                y_prob[mask].mean() - y_true[mask].mean())

    return {"brier": brier,
            "brier_ref_0.5": 0.25,
            "log_loss": logloss,
            "log_loss_ref_0.5": float(np.log(2)),
            "ece": float(ece),
            "skill_score": float(1 - brier / 0.25)}  # >0 = ดีกว่าเดา 50/50


# =============================================================================
# [ขั้น 2.4] Permutation test  <-- หลักฐานที่แข็งที่สุด
# =============================================================================

def permutation_test(fit_predict_fn,
                     X_train, y_train, X_val, y_val,
                     n_permutations: int = 100,
                     random_state: int = 42,
                     verbose: bool = True) -> dict:
    """
    สลับ label แบบสุ่มแล้วเทรนใหม่ n รอบ เพื่อสร้าง null distribution

    ตอบคำถาม: "ถ้าไม่มี signal เลย โมเดลจะได้ accuracy เท่าไหร่?"
    ถ้าผลจริงตกอยู่กลาง distribution -> ยืนยันว่าไม่มี signal
    ถ้าผลจริงสูงกว่า 95th percentile -> มี signal จริง (หรือมี leak)

    ข้อดีเพิ่มเติม: ถ้า pipeline มี leakage ผลจริงจะโดดออกมาผิดปกติ
    -> เป็นการตรวจ leak ไปในตัว

    Parameters
    ----------
    fit_predict_fn : callable(X_train, y_train, X_val) -> y_pred
        ฟังก์ชันที่เทรนโมเดลแล้วคืนค่าทำนายบน val
        ตัวอย่าง:
            def fn(Xtr, ytr, Xva):
                m = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
                return m.predict(Xva)

    Returns
    -------
    dict: observed_acc, null_scores, p_value, percentile, null_mean, null_std
    """
    rng = np.random.default_rng(random_state)

    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    # 1) ผลจริง
    observed = float(np.mean(fit_predict_fn(X_train, y_train, X_val) == y_val))

    # 2) null distribution
    null_scores = []
    for i in range(n_permutations):
        y_tr_perm = rng.permutation(y_train)
        y_va_perm = rng.permutation(y_val)
        pred = fit_predict_fn(X_train, y_tr_perm, X_val)
        null_scores.append(float(np.mean(pred == y_va_perm)))

        if verbose and (i + 1) % 20 == 0:
            print(f"    permutation {i+1}/{n_permutations} ...")

    null_scores = np.array(null_scores)

    # p-value: สัดส่วนของ null ที่ >= ผลจริง (+1 correction)
    p_value = float((np.sum(null_scores >= observed) + 1) /
                    (n_permutations + 1))
    percentile = float(stats.percentileofscore(null_scores, observed))

    if p_value < 0.05:
        interp = ("ผลจริงสูงกว่า null อย่างมีนัยสำคัญ "
                  "-> มี signal จริง หรือ ตรวจสอบ leakage")
    else:
        interp = ("ผลจริงอยู่ในช่วงเดียวกับ label สุ่ม "
                  "-> ยืนยันว่าไม่มี signal และ pipeline ไม่มี leak")

    return {"observed_acc": observed,
            "null_scores": null_scores,
            "null_mean": float(null_scores.mean()),
            "null_std": float(null_scores.std()),
            "null_p95": float(np.percentile(null_scores, 95)),
            "p_value": p_value,
            "percentile": percentile,
            "interpretation": interp}


def plot_null_distribution(perm_result: dict,
                           title: str = "Permutation Test",
                           save_path: str | None = None):
    """พล็อต histogram ของ null distribution พร้อมเส้นผลจริง (ใส่รายงานได้)"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(perm_result["null_scores"], bins=25,
            color="#9ecae1", edgecolor="white",
            label="Null (shuffled labels)")
    ax.axvline(perm_result["observed_acc"], color="#d62728", lw=2.5,
               label=f"Observed = {perm_result['observed_acc']:.4f}")
    ax.axvline(perm_result["null_p95"], color="#7f7f7f", ls="--", lw=1.5,
               label=f"Null 95th pct = {perm_result['null_p95']:.4f}")
    ax.axvline(0.5, color="black", ls=":", lw=1, label="Random = 0.50")
    ax.set_xlabel("Validation Accuracy")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{title}  (p = {perm_result['p_value']:.3f})")
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    บันทึกกราฟ: {save_path}")
    return fig


# =============================================================================
# [ขั้น 2.5] รายงานรวม
# =============================================================================

def full_report(y_true,
                predictions: dict,
                probabilities: dict | None = None,
                baseline_name: str | None = None,
                alpha: float = 0.05) -> pd.DataFrame:
    """
    สร้างตารางสรุปพร้อม CI, p-value, McNemar เทียบ baseline

    Parameters
    ----------
    y_true        : ค่าจริง
    predictions   : {"ชื่อโมเดล": array ของค่าทำนาย, ...}
                    ควรใส่ baseline เข้ามาด้วย
    probabilities : {"ชื่อโมเดล": array ของ P(y=1)} (optional)
    baseline_name : key ของ baseline ที่จะใช้เทียบใน McNemar
                    ถ้าไม่ระบุ จะเลือก baseline ที่ accuracy สูงสุด
                    จาก key ที่ขึ้นต้นด้วย "Baseline"
    """
    y_true = np.asarray(y_true)
    n = len(y_true)

    # หา baseline ที่ดีที่สุดอัตโนมัติ
    if baseline_name is None:
        cands = {k: np.mean(np.asarray(v) == y_true)
                 for k, v in predictions.items()
                 if k.lower().startswith("baseline")}
        baseline_name = max(cands, key=cands.get) if cands else None

    rows = []
    for name, pred in predictions.items():
        pred = np.asarray(pred)
        k = int(np.sum(pred == y_true))
        acc = k / n
        lo, hi = binom_ci(k, n, alpha)

        row = {
            "Model": name,
            "Accuracy": round(acc, 4),
            "CI_low": round(lo, 4),
            "CI_high": round(hi, 4),
            "p_vs_0.50": round(binom_test_vs(k, n, 0.5), 4),
        }

        # McNemar เทียบ baseline
        if baseline_name and name != baseline_name:
            mc = mcnemar_test(y_true, pred, predictions[baseline_name])
            row["McNemar_p"] = round(mc["pvalue"], 4)
            row["b/c"] = f"{mc['b']}/{mc['c']}"
        else:
            row["McNemar_p"] = np.nan
            row["b/c"] = "-"

        # probabilistic metrics
        if probabilities and name in probabilities:
            pm = prob_metrics(y_true, probabilities[name])
            row["Brier"] = round(pm["brier"], 4)
            row["LogLoss"] = round(pm["log_loss"], 4)
            row["ECE"] = round(pm["ece"], 4)
            row["Skill"] = round(pm["skill_score"], 4)

        rows.append(row)

    df = (pd.DataFrame(rows)
          .sort_values("Accuracy", ascending=False)
          .reset_index(drop=True))

    print("\n" + "=" * 78)
    print(f"  รายงานสถิติ (n = {n}, baseline อ้างอิง = {baseline_name})")
    print("=" * 78)
    print(df.to_string(index=False))
    print("\n  วิธีอ่าน:")
    print("    CI_low/CI_high : ถ้าช่วงคร่อม 0.50 = ไม่ต่างจากการเดา")
    print("    p_vs_0.50      : > 0.05 = ทายได้ไม่ต่างจากโยนเหรียญ")
    print("    McNemar_p      : > 0.05 = ไม่ต่างจาก baseline อย่างมีนัยสำคัญ")
    print("    Skill          : <= 0 = ไม่ดีกว่าการทำนาย 0.5 ทุกครั้ง")
    print("    b/c            : b = โมเดลถูก/baseline ผิด, c = กลับกัน")

    return df


# =============================================================================
# ตัวอย่างการใช้งาน
# =============================================================================

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    print(__doc__)
    print("\n--- ตัวอย่างการใช้งานกับข้อมูลสุ่ม (จำลองกรณีไม่มี signal) ---\n")

    rng = np.random.default_rng(0)
    n_tr, n_va, n_feat = 1701, 365, 16

    X_tr = rng.normal(size=(n_tr, n_feat))
    y_tr = rng.integers(0, 2, n_tr)
    X_va = rng.normal(size=(n_va, n_feat))
    y_va = rng.integers(0, 2, n_va)

    model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    pred = model.predict(X_va)
    prob = model.predict_proba(X_va)[:, 1]

    preds = {
        "Logistic Regression": pred,
        "Baseline: Majority": np.full(n_va, int(np.bincount(y_tr).argmax())),
        "Baseline: Persistence": np.concatenate([[y_va[0]], y_va[:-1]]),
    }

    # ขั้น 1
    print(verdict(np.mean(pred == y_va),
                  np.mean(preds["Baseline: Majority"] == y_va), n_va))

    # ขั้น 2
    full_report(y_va, preds, probabilities={"Logistic Regression": prob})

    print("\n--- Permutation test (100 รอบ) ---")

    def fit_fn(Xtr, ytr, Xva):
        return LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict(Xva)

    res = permutation_test(fit_fn, X_tr, y_tr, X_va, y_va,
                           n_permutations=100, verbose=True)
    print(f"\n    ผลจริง        : {res['observed_acc']:.4f}")
    print(f"    Null mean±std : {res['null_mean']:.4f} ± {res['null_std']:.4f}")
    print(f"    Null 95th pct : {res['null_p95']:.4f}")
    print(f"    p-value       : {res['p_value']:.4f}")
    print(f"    percentile    : {res['percentile']:.1f}")
    print(f"    => {res['interpretation']}")
