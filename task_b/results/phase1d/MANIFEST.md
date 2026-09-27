# results/phase1d — MANIFEST

> Phase 1D (Mode A 16:00) · development / walk-forward — **not a test**
> prereg_commit = `0b09905ea182edcaace2be641b529403290df2c7` (PHASE1D_PLAN.md + implementation, = HEAD ตอนรัน)
> `tune_1600.py --run` รันครั้งเดียว 2026-09-27 23:38–23:40 (+07:00) · ไม่มีการแก้โค้ด/แผนระหว่างหรือหลังรัน
> environment: Python 3.12.5 · numpy 2.1.1 · pandas 2.3.3 · scikit-learn 1.9.1 · xgboost 3.4.1
> input: raw_data_intraday/KBANK_BK_1h_730d.csv `b192ad2c…2060` · ADVANC_BK_1h_730d.csv `2097bb53…e79c`

| path | SHA-256 | สร้างด้วย |
|---|---|---|
| `results/phase1d/phase1d_folds.csv` | `6c2a0b0a564a7e775d461912091a73f29f6c9989522143562e2b11abebba4260` | python tune_1600.py --run |
| `results/phase1d/phase1d_evals.csv` | `0a72a35dc32a6628c400dcc15ea0f6fca823999f84487d152c01e43c3c94d97f` | python tune_1600.py --run |
| `results/phase1d/phase1d_scores.csv` | `72e96422c0b422ea2746ba07771b1cbc64de3eccd5d6e4e1bcfc25866d2f53db` | python tune_1600.py --run |
| `results/phase1d/phase1d_baselines.csv` | `ad58f75920719962c0866764ef0a61cce99befd1425498670224725dc593120e` | python tune_1600.py --run |
| `results/phase1d/phase1d_econ.csv` | `4753e9b101ea4a442194953df8376d7a0523b9023d71ca79d01fd355950a67ca` | python tune_1600.py --run |
| `results/phase1d/phase1d_summary.md` | `1a029bd0958fe7446fda020bb9346410a51400874a0e2cf88d9079698cee568e` | python tune_1600.py --run |
| `results/phase1d/tuning_1600_evals.csv` | `760ffd4a9eb817815a9cd3c401d1acbb62d8fc47532b2c32af643d3347b0c0bf` | python report_1600.py (อ่าน phase1d_*.csv อย่างเดียว ไม่ fit) |
| `results/phase1d/tuning_1600_scores.csv` | `3f9560de5ec1114a852a8fea2444d519d77208620029c596ef8527d63e048373` | python report_1600.py (อ่าน phase1d_*.csv อย่างเดียว ไม่ fit) |
| `results/phase1d/summary_1600.md` | `780785fa0c84126d65cfbe03a40878733c09e7619bd88a3613646f0e8f9022a2` | python report_1600.py (อ่าน phase1d_*.csv อย่างเดียว ไม่ fit) |

Dry-run ที่เกี่ยวข้อง (นอกโฟลเดอร์นี้):

| `results/dryrun/prediction_1600_dryrun.csv` | `feacfa091cb86120d83b9736b6a84367345de09e52984792e2d427d506e40174` | `python predict_1600.py --target-date 2026-09-2{3,4,5} --dry-run` (worktree_dirty=True) |

หมายเหตุ: `phase1d_summary.md` แสดงตัวเลขแบบ `np.float64(...)` (cosmetic ของโค้ดที่ล็อก ไม่แก้) · `tuning_1600_evals.csv` คอลัมน์ `n_nonfinite_pred` = 0 เพราะ `pred_finite` = True ทุกแถว
