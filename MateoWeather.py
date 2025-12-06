import requests
import pandas as pd
import os
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================
#       CONFIG PROVINSI DAN PARAMETER
# ============================================

PROVINCE_COORDS = {
    "Aceh": (5.5483, 95.3238),
    "Sumatera Utara": (3.5852, 98.6756),
    "Sumatera Barat": (-0.9492, 100.3543),
    "Riau": (0.5071, 101.4478),
    "Jambi": (-1.6100, 103.6131),
    "Sumatera Selatan": (-3.3194, 104.9140),
    "Bengkulu": (-3.8004, 102.2655),
    "Lampung": (-5.4295, 105.2626),
    "Kepulauan Bangka Belitung": (-2.1291, 106.1071),
    "Kepulauan Riau": (0.9130, 104.4667),
    "DKI Jakarta": (-6.2088, 106.8456),
    "Jawa Barat": (-6.9147, 107.6098),
    "Jawa Tengah": (-6.9667, 110.4167),
    "DI Yogyakarta": (-7.7956, 110.3695),
    "Jawa Timur": (-7.2504, 112.7688),
    "Banten": (-6.1200, 106.1500),
    "Bali": (-8.6500, 115.2167),
    "Nusa Tenggara Barat": (-8.5833, 116.1167),
    "Nusa Tenggara Timur": (-10.1833, 123.5833),
    "Kalimantan Barat": (-0.0225, 109.3324),
    "Kalimantan Tengah": (-2.2100, 113.9200),
    "Kalimantan Selatan": (-3.3186, 114.5944),
    "Kalimantan Timur": (0.5022, 117.1536),
    "Kalimantan Utara": (3.3267, 117.5786),
    "Sulawesi Utara": (1.4748, 124.8421),
    "Sulawesi Tengah": (-0.9000, 119.8700),
    "Sulawesi Selatan": (-5.1400, 119.4221),
    "Sulawesi Tenggara": (-4.0167, 122.5167),
    "Gorontalo": (0.5372, 123.0667),
    "Sulawesi Barat": (-2.8500, 119.2333),
    "Maluku": (-3.6964, 128.1814),
    "Maluku Utara": (0.7833, 127.3833),
    "Papua": (-2.5333, 140.7000),
    "Papua Barat": (-0.8667, 134.0833),
    "Papua Tengah": (-3.8100, 138.0800),
    "Papua Pegunungan": (-3.9000, 139.0333),
    "Papua Selatan": (-7.9900, 139.3300)
}

DAILY_PARAMS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "shortwave_radiation_sum"
]


# ============================================
#           SETUP SESSION DENGAN RETRY
# ============================================

def create_session():
    session = requests.Session()

    retries = Retry(
        total=5,
        backoff_factor=2,  # 1s → 2s → 4s → 8s → 16s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session

session = create_session()


# ============================================
#       FUNGSI AMBIL DATA (AMAN DARI ERROR)
# ============================================

def fetch_meteo(lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "daily": ",".join(DAILY_PARAMS),
        "timezone": "Asia/Jakarta"
    }

    try:
        res = session.get(
            url,
            params=params,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        res.raise_for_status()
        return res.json()

    except Exception as e:
        print(f"[ERROR] gagal ambil lat={lat}, lon={lon}")
        print("        alasan:", e)
        return None


# ============================================
#           PROSES DAN SIMPAN JSON
# ============================================

OUTPUT_DIR = "weather_json_indonesia_2024"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_data = []

print("\n=== MENGAMBIL DATA CUACA BULANAN 2024 (OPEN-METEO, JSON FORMAT) ===\n")

for prov, (lat, lon) in PROVINCE_COORDS.items():
    print(f"Mengambil data: {prov} ...")

    data = fetch_meteo(lat, lon)

    if not data or "daily" not in data:
        print(f"[!] GAGAL mengambil: {prov}\n")
        continue

    d = data["daily"]

    df = pd.DataFrame({
        "tanggal": d["time"],
        "suhu_rata_rata_c": d["temperature_2m_mean"],
        "suhu_maksimum_c": d["temperature_2m_max"],
        "suhu_minimum_c": d["temperature_2m_min"],
        "kelembapan_rata_rata_persen": d["relative_humidity_2m_mean"],
        "curah_hujan_mm": d["precipitation_sum"],
        "kecepatan_angin_maks_kmh": d["wind_speed_10m_max"],
        "radiasi_matahari_mj": d["shortwave_radiation_sum"]
    })

    # konversi tanggal → bulan
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df["bulan"] = df["tanggal"].dt.to_period("M").astype(str)

    # bulanan → mean kecuali curah hujan (SUM)
    df_monthly = df.groupby("bulan").agg({
        "suhu_rata_rata_c": "mean",
        "suhu_maksimum_c": "mean",
        "suhu_minimum_c": "mean",
        "kelembapan_rata_rata_persen": "mean",
        "curah_hujan_mm": "sum",
        "kecepatan_angin_maks_kmh": "mean",
        "radiasi_matahari_mj": "mean"
    }).reset_index()

    df_monthly.insert(0, "provinsi", prov)

    # simpan JSON per provinsi
    df_monthly.to_json(f"{OUTPUT_DIR}/{prov.replace(' ', '_')}.json",
                       orient="records",
                       indent=4)

    all_data.append(df_monthly)

    print(f"✔ selesai: {prov}\n")
    time.sleep(1)


# ============================================
#               GABUNG NASIONAL
# ============================================

df_final = pd.concat(all_data, ignore_index=True)
df_final.to_json(f"{OUTPUT_DIR}/weather_indonesia_2024.json",
                 orient="records",
                 indent=4)

print("\n=== SELESAI! Semua provinsi berhasil dibuat JSON bulanan + digabung nasional ===\n")
