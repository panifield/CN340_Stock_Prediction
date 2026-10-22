# Coverage ของ log ตั้งแต่ 2026-09-28

L = log จริงครบ (6 แถว: 2 หุ้น × 3 โมเดล -- task_a/task_a2 เป็น 8 แถวตั้งแต่ 2026-10-20 ที่เพิ่ม baseline) · B = มีเฉพาะ backfill · L+B = log จริงมีไม่ครบโมเดล backfill เติมให้ครบ · -- = ยังขาด · n/a = ชุดนั้นยังไม่เริ่ม/เลิกรันแล้ว (รอบ 16:00 ของ task_b เปลี่ยนจาก task_b_1600 เป็น task_b_close วันที่ 2026-10-19)

| วันที่ | วัน | task_a<br>(next_day) | task_c<br>(next_day) | task_b<br>(next_day) | task_a2<br>(t1600) | task_b_1600<br>(t1600) | task_b_close<br>(t1600) | task_c_1600<br>(t1600) |
|---|---|---|---|---|---|---|---|---|
| 2026-09-28 | Mon | B | B | B | B | B | n/a | B |
| 2026-09-29 | Tue | B | B | B | B | B | n/a | B |
| 2026-09-30 | Wed | L+B | L+B | B | L+B | L | n/a | L+B |
| 2026-10-01 | Thu | L | L | L | L | L | n/a | L |
| 2026-10-02 | Fri | L | L | L | L | L | n/a | L |
| 2026-10-05 | Mon | L | L | L | L | L | n/a | L |
| 2026-10-06 | Tue | L | L | L | L | B | n/a | B |
| 2026-10-07 | Wed | B | B | B | B | B | n/a | B |
| 2026-10-08 | Thu | B | B | B | B | B | n/a | B |
| 2026-10-09 | Fri | B | B | B | B | B | n/a | B |
| 2026-10-12 | Mon | L | B | B | L | L | n/a | L |
| 2026-10-14 | Wed | L | L | L | L | L | n/a | L |
| 2026-10-15 | Thu | L | L | L | L | L | n/a | L |
| 2026-10-19 | Mon | L | L | L | L | n/a | L | L |
| 2026-10-20 | Tue | L | L | L | L | n/a | L | L |
| 2026-10-21 | Wed | L | L | L | B | n/a | B | B |
| 2026-10-22 | Thu | L | L | L | L | n/a | L | L |

ช่องที่ยังขาด: **0** จาก 102 ช่องที่ต้องมี (วันทำการ 17 วัน × 7 ชุด = 119 ช่อง หัก n/a แล้ว)
