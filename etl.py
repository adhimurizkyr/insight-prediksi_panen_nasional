import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split # <-- Import untuk pemisahan data
import joblib 

# Definisikan kolom cuaca yang akan dijadikan fitur lagging
WEATHER_COLS = [
    'suhu_rata_rata_c', 'curah_hujan_mm', 
    'kelembapan_rata_rata_persen', 'kecepatan_angin_maks_kmh', 
    'radiasi_matahari_mj'
]

# =======================================
#           LOAD DATA CUACA (JSON)
# =======================================
def load_weather_json(json_path):
    df_list = [] 
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Folder tidak ditemukan: {json_path}")
    for file in os.listdir(json_path):
        if file.endswith(".json") and file != "weather_indonesia_2024.json":
            path = os.path.join(json_path, file)
            df_list.append(pd.read_json(path))
    if not df_list:
        raise FileNotFoundError(f"Tidak ada file JSON ditemukan di {json_path}")
    df_weather = pd.concat(df_list, ignore_index=True)
    return df_weather

# =======================================
#           LOAD DATA PANEN (CSV)
# =======================================
def load_crop_csv(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder tidak ditemukan: {folder_path}")
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"Tidak ada file CSV ditemukan di {folder_path}")
    csv_path = os.path.join(folder_path, csv_files[0])
    print(f"    -> Membaca file: {csv_files[0]}")
    df_crop = pd.read_csv(csv_path, header=2) 
    df_crop = df_crop.replace('-', pd.NA) 
    
    new_columns = []
    for i, c in enumerate(df_crop.columns):
        if i == 0:
            new_columns.append("provinsi")
        else:
            cleaned = str(c).strip().lower()
            cleaned = cleaned.replace(" ", "_")
            cleaned = cleaned.replace("(", "").replace(")", "").replace("/", "")
            new_columns.append(cleaned)
    df_crop.columns = new_columns
    
    rename_map = {
        "luas_panenha": "luas_panen", "luas_panen_ha": "luas_panen", 
        "produktivitaskuha": "produktivitas", "produktivitas_kuha": "produktivitas", 
        "produksi_ton": "produksi",
    }
    df_crop.rename(columns=rename_map, inplace=True)
    print(f"    -> Kolom final: {df_crop.columns.tolist()}")

    df_crop = df_crop.dropna(subset=['provinsi'])
    return df_crop

# =======================================
#    CLEANING DATA CUACA (BULAN)
# =======================================
def clean_weather(df):
    df.rename(columns={"bulan": "month", "provinsi": "province"}, inplace=True)
    df = df.sort_values(["province", "month"])
    return df

# =======================================
#    CLEANING DATA PANEN (TAHUNAN)
# =======================================
def clean_crop(df):
    df.rename(columns={"provinsi": "province"}, inplace=True)
    df['province'] = df['province'].str.title() 
    df['luas_panen'] = pd.to_numeric(df['luas_panen'], errors='coerce')
    df['produksi'] = pd.to_numeric(df['produksi'], errors='coerce')
    df['produktivitas'] = pd.to_numeric(df['produktivitas'], errors='coerce')
    
    df_month = pd.DataFrame()
    df_valid = df.dropna(subset=['luas_panen', 'produksi', 'produktivitas'])
    
    for _, row in df_valid.iterrows():
        for month in range(1, 13):
            df_month = pd.concat([
                df_month,
                pd.DataFrame({
                    "province": [row["province"]],
                    "month": [f"2024-{month:02d}"],
                    "luas_panen": [row["luas_panen"]/12],
                    "produksi": [row["produksi"]/12],
                    "produktivitas": [row["produktivitas"]]
                })
                ], ignore_index=True)
    return df_month

# =======================================
#           MERGE CUACA + PANEN
# =======================================
def merge_weather_crop(df_weather, df_crop):
    df_merge = pd.merge(df_weather, df_crop, on=["province", "month"], how="inner")
    return df_merge

# =======================================
#           FEATURE ENGINEERING
# =======================================
def create_lag_features(df, lag=3):
    """Membuat fitur lagging untuk data cuaca dan mengembalikan DataFrame baru."""
    df = df.copy()
    
    df['month_dt'] = pd.to_datetime(df['month'])
    df = df.sort_values(['province', 'month_dt']).reset_index(drop=True)
    
    for col in WEATHER_COLS:
        for i in range(1, lag + 1):
            df[f'{col}_lag_{i}'] = df.groupby('province')[col].shift(i)

    df.drop(columns=['month_dt'], inplace=True)
    return df

# =======================================
#           TRAIN MODEL (PERFECT)
# =======================================
def train_and_save_model(df_features, model_path, accuracy_path):
    """Menerima DataFrame yang sudah punya fitur lagging, melakukan OHE, train, dan save."""
    
    df_model = df_features.copy()
    
    # Drop baris dengan nilai NaN yang dihasilkan dari lagging (3 bulan pertama)
    df_model.dropna(inplace=True) 

    # One-Hot Encoding untuk kolom 'province'
    df_model = pd.get_dummies(df_model, columns=['province'], drop_first=False)
    
    # Definisikan Features (X) dan Target (y)
    TARGET = 'produksi'
    
    # Features adalah semua kolom lag cuaca + kolom OHE provinsi
    feature_cols = [col for col in df_model.columns if '_lag_' in col]
    ohe_cols = sorted([col for col in df_model.columns if col.startswith('province_')])
    
    X_cols = feature_cols + ohe_cols
    
    X = df_model[X_cols]
    y = df_model[TARGET]
    
    # --- 1. TRAIN-TEST SPLIT ---
    # Memisahkan data menjadi 80% Training dan 20% Testing (Data yang belum pernah dilihat)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"    -> Jumlah Fitur (Lagging + OHE): {X_train.shape[1]}")
    print(f"    -> Ukuran Training Set: {X_train.shape[0]} baris")
    print(f"    -> Ukuran Testing Set: {X_test.shape[0]} baris")
    
    # Training Model (Latih hanya dengan data training)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train) 
    
    # --- 2. EVALUASI ---
    
    # Evaluasi pada data Training (Harusnya tinggi karena overfitting)
    r2_score_train = model.score(X_train, y_train)
    
    # Evaluasi pada data Testing (AKURASI RIIL untuk laporan)
    y_pred_test = model.predict(X_test) 
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_score_test = model.score(X_test, y_test) 
    
    print(f"    -> Model Training R2 Score (Indikasi Overfitting): {r2_score_train:.4f}")
    print(f"    -> Model Testing R2 Score (AKURASI KREDIBEL): {r2_score_test:.4f}")
    print(f"    -> Model Testing RMSE: {rmse_test:.2f} ton")
    
    # Simpan R2 Score TESTING ke file
    try:
        with open(accuracy_path, 'w') as f:
            f.write(f"{r2_score_test:.4f}") 
        print("    -> Akurasi model (R2 Test) berhasil disimpan:", accuracy_path)
    except Exception as e:
        print(f"    -> GAGAL menyimpan akurasi model: {e}")
        
    # Simpan Model
    joblib.dump(model, model_path)
    print("    -> Model prediksi berhasil disimpan:", model_path)
    
    return model

# =======================================
#           PIPELINE UTAMA ETL + TRAIN (FINAL)
# =======================================
def run_etl_and_train(weather_dir, crop_folder, etl_output_path, model_output_path, accuracy_output_path):
    print("--- 1. ETL PROCESS ---")
    
    # Load and Clean
    df_weather = load_weather_json(weather_dir)
    df_weather = clean_weather(df_weather)
    df_crop = load_crop_csv(crop_folder)
    df_crop = clean_crop(df_crop)

    # Merge
    print("Merging...")
    df_final = merge_weather_crop(df_weather, df_crop)
    
    # FEATURE ENGINEERING: Buat fitur lagging pada DataFrame yang akan disimpan ke CSV
    print("Creating Lagging Features (3 months)...")
    df_final = create_lag_features(df_final, lag=3)

    # Save CSV
    df_final.to_csv(etl_output_path, index=False)
    print("\n=== ETL SELESAI! ===")
    print("Output CSV saved to:", etl_output_path)
    
    # --- 2. MODEL TRAINING PROCESS ---
    print("\n--- 2. MODEL TRAINING PROCESS ---")
    train_and_save_model(df_final.copy(), model_output_path, accuracy_output_path)
    print("=== MODEL TRAINING SELESAI! ===")

# =======================================
#               RUN SCRIPT
# =======================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    weather_path = os.path.join(base_dir, "weather_json_indonesia_2024")
    crop_path = os.path.join(base_dir, "bps_padi_provinsi")
    
    etl_output_file = os.path.join(base_dir, "etl_panen_cuaca_2024.csv")
    model_output_file = os.path.join(base_dir, "panen_predictor_model.joblib")
    accuracy_output_file = os.path.join(base_dir, "model_accuracy.txt")

    try:
        run_etl_and_train(
            weather_dir=weather_path,
            crop_folder=crop_path,
            etl_output_path=etl_output_file,
            model_output_path=model_output_file,
            accuracy_output_path=accuracy_output_file
        )
    except FileNotFoundError as e:
        print(f"\n[FATAL ERROR]: {e}")
        print("Pastikan Anda memiliki folder 'weather_json_indonesia_2024' dan 'bps_padi_provinsi' di direktori yang sama.")