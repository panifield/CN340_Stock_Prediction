# Dockerfile -- รันงาน cron ของ Railway (ดู deploy/README.md)
#
# image นี้มีแค่ "สภาพแวดล้อม" (Python + library ที่ pin ไว้ + git) ไม่ได้ฝังโค้ดของโปรเจกต์ไว้ใช้รัน
# ทุกครั้งที่ cron ทำงาน deploy/railway_run.sh จะ git clone repo ล่าสุดมาใหม่ แล้วรันสคริปต์จากสำเนานั้น
# เพราะ filesystem ของ Railway ไม่ถาวร และ log ที่ระบบเขียนต้อง push กลับ GitHub ถึงจะไม่หาย
# (ถ้าใช้โค้ดที่ฝังใน image จะ push ไม่ผ่านตั้งแต่วันที่ 2 เพราะ log ของวันก่อนอยู่บน GitHub ไม่ใช่ใน image)
#
# Python 3.12.5 = เวอร์ชันที่ task_b/requirements.txt ทดสอบ (guard ของ task_b ตรวจเวอร์ชัน library -- ห้ามเปลี่ยนเวอร์ชันเอง)
FROM python:3.12.5-slim

ENV PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Bangkok

# git: clone/commit/push · libgomp1: OpenMP ที่ lightgbm/xgboost ต้องใช้ · tzdata: ให้ TZ=Asia/Bangkok มีผลจริง
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates tzdata libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# ติดตั้ง library ตามลำดับเดียวกับ .github/workflows/next_day.yml
# (task_b/requirements-live.txt มี "-r requirements.txt" -> ต้องวางสองไฟล์ไว้โฟลเดอร์เดียวกัน)
WORKDIR /opt/deps
COPY requirements.txt ./requirements.txt
COPY task_b/requirements.txt task_b/requirements-live.txt ./task_b/
RUN pip install -r requirements.txt \
 && pip install -r task_b/requirements-live.txt

COPY deploy/railway_run.sh /usr/local/bin/railway_run.sh
RUN sed -i 's/\r$//' /usr/local/bin/railway_run.sh && chmod +x /usr/local/bin/railway_run.sh

WORKDIR /app
# ค่าเริ่มต้นที่ปลอดภัย: smoke test อย่างเดียว (ไม่ทำนาย ไม่ commit) -- railway.toml จะ override ด้วย startCommand
CMD ["bash", "/usr/local/bin/railway_run.sh", "smoke"]
