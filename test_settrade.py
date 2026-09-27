import os

from settrade_v2.user import Investor

# *** ห้าม hardcode secret ในโค้ดเด็ดขาด — อ่านจาก environment variable เสมอ ***
# ตั้งค่าก่อนรันไฟล์นี้ เช่น (PowerShell):
#   $env:SETTRADE_APP_ID = "xxxx"
#   $env:SETTRADE_APP_SECRET = "xxxx"
#   $env:SETTRADE_BROKER_ID = "023"
#   $env:SETTRADE_APP_CODE = "ALGO_EQ"
investor = Investor(
    app_id=os.environ["SETTRADE_APP_ID"],
    app_secret=os.environ["SETTRADE_APP_SECRET"],
    broker_id=os.environ.get("SETTRADE_BROKER_ID", "023"),
    app_code=os.environ.get("SETTRADE_APP_CODE", "ALGO_EQ"),
)

market = investor.MarketData()

data = market.get_candlestick(
    symbol="KBANK",
    interval="1d",
    limit=10
)
import pandas as pd
from datetime import datetime

market = investor.MarketData()

def get_10_years(symbol):
    all_data = []

    for year in range(2016, 2027):
        start = f"{year}-01-01T00:00:00+07:00"

        if year == 2026:
            end = "2026-09-11T23:59:59+07:00"
        else:
            end = f"{year}-12-31T23:59:59+07:00"

        print(f"Downloading {symbol}: {year}")

        data = market.get_candlestick(
            symbol=symbol,
            interval="1d",
            start=start,
            end=end,
            limit=500
        )

        if len(data.get("time", [])) == 0:
            continue

        df = pd.DataFrame({
            "time": data["time"],
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
            "value": data["value"]
        })

        df["symbol"] = symbol

        all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)

    # Unix timestamp -> วันที่ไทย
    df["date"] = (
        pd.to_datetime(df["time"], unit="s", utc=True)
        .dt.tz_convert("Asia/Bangkok")
        .dt.tz_localize(None)
    )

    # เรียงวันที่ + ลบข้อมูลซ้ำ
    df = (
        df.drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    df = df[
        ["date", "symbol", "open", "high", "low",
         "close", "volume", "value"]
    ]

    return df


# ========================
# KBANK
# ========================
kbank = get_10_years("KBANK")
kbank.to_csv("KBANK_10Y.csv", index=False)

print("\nKBANK")
print(kbank.head())
print(kbank.tail())
print("จำนวนวัน =", len(kbank))


# ========================
# ADVANC
# ========================
advanc = get_10_years("ADVANC")
advanc.to_csv("ADVANC_10Y.csv", index=False)

print("\nADVANC")
print(advanc.head())
print(advanc.tail())
print("จำนวนวัน =", len(advanc))


# ========================
# รวมสองหุ้น
# ========================
stocks = pd.concat([kbank, advanc], ignore_index=True)
stocks.to_csv("KBANK_ADVANC_10Y.csv", index=False)

print("\nDONE")
print("KBANK:", len(kbank))
print("ADVANC:", len(advanc))